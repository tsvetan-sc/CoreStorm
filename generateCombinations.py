"""Generate fixed-width decimal combinations safely and efficiently."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO

DEFAULT_BATCH_SIZE = 50_000
DEFAULT_GPU_CHUNK = 10_000_000
DEFAULT_PROGRESS_INTERVAL = 5.0
MAX_LENGTH = 18
CONFIRMATION_THRESHOLD = 1_000_000
CHECKPOINT_VERSION = 1


class GenerationError(RuntimeError):
    """An expected, user-facing generation error."""


ProgressCallback = Callable[[int], None]


def durable_fsync(fd: int) -> None:
    """Flush a file descriptor all the way to stable storage.

    On macOS (including Apple silicon), ``fsync`` only hands data to the drive,
    which may keep it in a volatile cache; ``F_FULLFSYNC`` is required for the
    power-loss guarantee that checkpoints rely on. Filesystems that reject
    ``F_FULLFSYNC`` (some network or FUSE mounts) fall back to ``fsync``.
    """
    if sys.platform == "darwin":
        import fcntl

        full_fsync = getattr(fcntl, "F_FULLFSYNC", None)
        if full_fsync is not None:
            try:
                fcntl.fcntl(fd, full_fsync)
                return
            except OSError:
                pass
    os.fsync(fd)


def iter_cpu_batches(
    length: int, start: int, stop: int, batch_size: int
) -> Iterator[tuple[int, bytes]]:
    """Yield (next index, encoded output) batches without retaining the full set."""
    for batch_start in range(start, stop, batch_size):
        batch_stop = min(batch_start + batch_size, stop)
        data = "".join(
            str(number).zfill(length) + "\n"
            for number in range(batch_start, batch_stop)
        ).encode("ascii")
        yield batch_stop, data


def generate_combinations_cpu(
    length: int, start: int, stop: int, batch_size: int
) -> Iterator[tuple[int, bytes]]:
    """Generate encoded batches on the CPU."""
    yield from iter_cpu_batches(length, start, stop, batch_size)


def generate_combinations_gpu(
    length: int, start: int, stop: int, max_gpu_chunk: int
) -> Iterator[tuple[int, bytes]]:
    """Generate encoded batches with CUDA, importing GPU dependencies lazily."""
    try:
        import numpy as np
        from numba import cuda
    except ImportError as exc:
        raise GenerationError(
            "GPU mode requires optional dependencies; "
            "install with: pip install -e .[gpu]"
        ) from exc

    if not cuda.is_available():
        if sys.platform == "darwin":
            raise GenerationError(
                "GPU mode uses CUDA, which requires an NVIDIA GPU and is not "
                "available on macOS (including Apple silicon). Run without --gpu."
            )
        raise GenerationError("CUDA is not available on this system.")
    if stop > np.iinfo(np.int64).max:
        raise GenerationError("The requested GPU range exceeds CUDA's int64 limit.")

    @cuda.jit
    def kernel(result, start_index, width):  # pragma: no cover - requires CUDA
        index = cuda.grid(1)
        if index < result.shape[0]:
            number = start_index + index
            for position in range(width):
                result[index, width - position - 1] = number % 10
                number //= 10

    try:
        free_memory, _ = cuda.current_context().get_memory_info()
    except Exception as exc:  # pragma: no cover - requires CUDA
        raise GenerationError(f"Unable to query CUDA memory: {exc}") from exc

    # Digits use one byte on the device. Keep half of free VRAM in reserve.
    memory_limited_chunk = max(1, int((free_memory * 0.5) // length))
    chunk_size = min(max_gpu_chunk, memory_limited_chunk, stop - start)
    threads_per_block = 256

    for chunk_start in range(start, stop, chunk_size):
        chunk_stop = min(chunk_start + chunk_size, stop)
        count = chunk_stop - chunk_start
        try:
            device_result = cuda.device_array((count, length), dtype=np.uint8)
            blocks_per_grid = (count + threads_per_block - 1) // threads_per_block
            kernel[blocks_per_grid, threads_per_block](
                device_result, np.int64(chunk_start), length
            )
            host_digits = device_result.copy_to_host()
            cuda.synchronize()

            # Serialize the matrix in native NumPy operations, not a Python row loop.
            output = np.empty((count, length + 1), dtype=np.uint8)
            output[:, :length] = host_digits + ord("0")
            output[:, length] = ord("\n")
            yield chunk_stop, output.tobytes()
        except Exception as exc:  # pragma: no cover - requires CUDA
            raise GenerationError(f"CUDA generation failed: {exc}") from exc


def checkpoint_paths(output_path: Path) -> tuple[Path, Path]:
    return (
        output_path.with_name(output_path.name + ".part"),
        output_path.with_name(output_path.name + ".checkpoint.json"),
    )


def write_checkpoint(path: Path, state: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        durable_fsync(handle.fileno())
    os.replace(temporary, path)


def load_checkpoint(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise GenerationError(f"Cannot read checkpoint {path}: {exc}") from exc
    if state.get("version") != CHECKPOINT_VERSION:
        raise GenerationError(f"Unsupported checkpoint version in {path}.")
    return state


def remove_if_present(path: Path) -> None:
    with suppress(FileNotFoundError):
        path.unlink()


def confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        return input(f"{prompt} [y/N]: ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024 or unit == "PiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} PiB"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate fixed-width decimal combinations."
    )
    parser.add_argument("length", type=int, help="combination width")
    parser.add_argument("--gpu", action="store_true", help="use CUDA generation")
    parser.add_argument(
        "--gpu-fallback",
        action="store_true",
        help="restart with CPU if explicitly requested GPU generation fails",
    )
    parser.add_argument(
        "--output", default="combinations.txt", help="output path, or - for stdout"
    )
    parser.add_argument(
        "--gzip",
        action="store_true",
        help="write concatenated gzip members (also enabled by a .gz suffix)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="CPU batch size"
    )
    parser.add_argument(
        "--max-gpu-chunk",
        type=int,
        default=DEFAULT_GPU_CHUNK,
        help="maximum combinations held in a GPU chunk",
    )
    parser.add_argument("--start", type=int, default=0, help="first value, inclusive")
    parser.add_argument("--stop", type=int, help="last value, exclusive")
    parser.add_argument("--resume", action="store_true", help="resume a checkpoint")
    parser.add_argument("--force", action="store_true", help="replace output/work")
    parser.add_argument("--yes", action="store_true", help="accept large output")
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=DEFAULT_PROGRESS_INTERVAL,
        help="seconds between progress updates; 0 disables them",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not 1 <= args.length <= MAX_LENGTH:
        parser.error(f"length must be between 1 and {MAX_LENGTH}")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if args.max_gpu_chunk <= 0:
        parser.error("--max-gpu-chunk must be greater than zero")
    if args.progress_interval < 0:
        parser.error("--progress-interval cannot be negative")

    total = 10**args.length
    if args.start < 0 or args.start >= total:
        parser.error(f"--start must be between 0 and {total - 1}")
    if args.stop is not None and not args.start < args.stop <= total:
        parser.error(f"--stop must be greater than --start and at most {total}")
    if args.output == "-" and args.resume:
        parser.error("--resume cannot be used with stdout")
    if args.output == "-" and args.force:
        parser.error("--force is not meaningful with stdout")
    if args.gpu_fallback and not args.gpu:
        parser.error("--gpu-fallback requires --gpu")


def checkpoint_signature(
    args: argparse.Namespace, stop: int, compressed: bool
) -> dict[str, object]:
    return {
        "version": CHECKPOINT_VERSION,
        "length": args.length,
        "start": args.start,
        "stop": stop,
        "compressed": compressed,
        "gpu": args.gpu,
    }


def prepare_output(
    args: argparse.Namespace, stop: int, compressed: bool
) -> tuple[Path, Path, Path, int]:
    output = Path(args.output).expanduser().resolve()
    parent = output.parent
    if not parent.exists() or not parent.is_dir():
        raise GenerationError(f"Output directory does not exist: {parent}")

    part, checkpoint = checkpoint_paths(output)
    signature = checkpoint_signature(args, stop, compressed)
    if args.resume:
        if not part.exists() or not checkpoint.exists():
            raise GenerationError("No matching incomplete output and checkpoint found.")
        state = load_checkpoint(checkpoint)
        for key, expected in signature.items():
            if state.get(key) != expected:
                raise GenerationError(
                    f"Checkpoint mismatch for {key}: expected {expected!r}, "
                    f"found {state.get(key)!r}."
                )
        try:
            next_index = int(state["next_index"])
            part_size = int(state["part_size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GenerationError("Checkpoint lacks valid progress data.") from exc
        actual_size = part.stat().st_size
        if not args.start <= next_index <= stop or not 0 <= part_size <= actual_size:
            raise GenerationError("Checkpoint progress is outside the expected range.")
        with part.open("r+b") as handle:
            handle.truncate(part_size)
        return output, part, checkpoint, next_index

    incomplete_exists = part.exists() or checkpoint.exists()
    if (
        incomplete_exists
        and not args.force
        and not confirm("Discard existing incomplete output?", False)
    ):
        raise GenerationError("Existing incomplete output was left unchanged.")
    if (
        output.exists()
        and not args.force
        and not confirm(f"Replace existing file {output}?", False)
    ):
        raise GenerationError("Existing output was left unchanged.")
    remove_if_present(part)
    remove_if_present(checkpoint)
    return output, part, checkpoint, args.start


def ensure_disk_space(path: Path, required: int) -> None:
    try:
        free = shutil.disk_usage(path.parent).free
    except OSError as exc:
        raise GenerationError(f"Unable to check free disk space: {exc}") from exc
    if required > free:
        raise GenerationError(
            f"Insufficient disk space: need up to {format_bytes(required)}, "
            f"but only {format_bytes(free)} is free."
        )


def make_progress_reporter(
    start: int, stop: int, interval: float, started_at: float
) -> ProgressCallback:
    last_report = [started_at]

    def report(next_index: int) -> None:
        now = time.perf_counter()
        if interval == 0 or (next_index < stop and now - last_report[0] < interval):
            return
        completed = next_index - start
        count = stop - start
        print(
            f"Progress: {completed:,}/{count:,} ({completed * 100 / count:.1f}%), "
            f"{completed / max(now - started_at, 1e-9):,.0f} combinations/s",
            file=sys.stderr,
        )
        last_report[0] = now

    return report


def write_batches(
    batches: Iterator[tuple[int, bytes]],
    handle: BinaryIO,
    compressed: bool,
    checkpoint: Path | None,
    signature: dict[str, object] | None,
    progress: ProgressCallback,
) -> None:
    for next_index, data in batches:
        # Each compressed batch is a complete gzip member, making truncation and
        # append-based resume reliable after an interruption.
        handle.write(gzip.compress(data, mtime=0) if compressed else data)
        handle.flush()
        if checkpoint is not None and signature is not None:
            # Persist output before advancing the durable checkpoint. This ordering
            # prevents a power loss from making the checkpoint point past data.
            durable_fsync(handle.fileno())
            state = dict(signature)
            state.update(next_index=next_index, part_size=handle.tell())
            write_checkpoint(checkpoint, state)
        progress(next_index)


def choose_batches(
    args: argparse.Namespace, start: int, stop: int
) -> Iterator[tuple[int, bytes]]:
    if not args.gpu:
        return generate_combinations_cpu(args.length, start, stop, args.batch_size)

    def gpu_with_optional_fallback() -> Iterator[tuple[int, bytes]]:
        produced_output = False
        try:
            for batch in generate_combinations_gpu(
                args.length, start, stop, args.max_gpu_chunk
            ):
                produced_output = True
                yield batch
        except GenerationError:
            if not args.gpu_fallback or produced_output:
                raise
            print("GPU failed; restarting the current range on CPU.", file=sys.stderr)
            yield from generate_combinations_cpu(
                args.length, start, stop, args.batch_size
            )

    return gpu_with_optional_fallback()


def run_generation(args: argparse.Namespace) -> None:
    stop = args.stop if args.stop is not None else 10**args.length
    compressed = args.gzip or args.output.lower().endswith(".gz")
    count = stop - args.start
    estimated_size = count * (args.length + 1)
    if count >= CONFIRMATION_THRESHOLD and not confirm(
        f"Generate {count:,} combinations (up to {format_bytes(estimated_size)})?",
        args.yes,
    ):
        raise GenerationError("Generation cancelled.")

    started_at = time.perf_counter()
    if args.output == "-":
        reporter = make_progress_reporter(
            args.start, stop, args.progress_interval, started_at
        )
        write_batches(
            choose_batches(args, args.start, stop),
            sys.stdout.buffer,
            compressed,
            None,
            None,
            reporter,
        )
        return

    output, part, checkpoint, next_index = prepare_output(args, stop, compressed)
    # Conservative for gzip, whose actual output is normally smaller.
    ensure_disk_space(part, (stop - next_index) * (args.length + 1))
    signature = checkpoint_signature(args, stop, compressed)
    reporter = make_progress_reporter(
        args.start, stop, args.progress_interval, started_at
    )
    with part.open("ab") as handle:
        # Create a resumable checkpoint before generation initializes (CUDA setup
        # can fail before producing its first batch).
        initial_state = dict(signature)
        initial_state.update(next_index=next_index, part_size=handle.tell())
        write_checkpoint(checkpoint, initial_state)
        write_batches(
            choose_batches(args, next_index, stop),
            handle,
            compressed,
            checkpoint,
            signature,
            reporter,
        )
        handle.flush()
        durable_fsync(handle.fileno())
    os.replace(part, output)
    remove_if_present(checkpoint)

    duration = time.perf_counter() - started_at
    print(
        f"Generated {count:,} combinations in {duration:.2f}s. Output: {output}",
        file=sys.stderr,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        validate_args(args, parser)
        run_generation(args)
        return 0
    except GenerationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Incomplete output was preserved; use --resume to continue.",
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
