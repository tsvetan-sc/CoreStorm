from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import generateCombinations as generator


def run_cli(output: Path, *arguments: str) -> int:
    return generator.main(
        [
            "3",
            "--output",
            str(output),
            "--force",
            "--progress-interval",
            "0",
            *arguments,
        ]
    )


def test_cpu_generation_preserves_width_order_and_boundaries(tmp_path: Path) -> None:
    output = tmp_path / "values.txt"
    assert run_cli(output, "--start", "7", "--stop", "12", "--batch-size", "2") == 0
    assert output.read_bytes() == b"007\n008\n009\n010\n011\n"
    assert not Path(f"{output}.part").exists()
    assert not Path(f"{output}.checkpoint.json").exists()


def test_gzip_output_is_complete_and_readable(tmp_path: Path) -> None:
    output = tmp_path / "values.txt.gz"
    assert run_cli(output, "--start", "98", "--stop", "102", "--batch-size", "2") == 0
    with gzip.open(output, "rb") as handle:
        assert handle.read() == b"098\n099\n100\n101\n"


def test_gzip_resume_truncates_to_last_complete_member(tmp_path: Path) -> None:
    output = tmp_path / "values.txt.gz"
    part = Path(f"{output}.part")
    checkpoint = Path(f"{output}.checkpoint.json")
    committed = gzip.compress(b"000\n001\n", mtime=0)
    part.write_bytes(committed + b"incomplete member")
    checkpoint.write_text(
        json.dumps(
            {
                "version": 1,
                "length": 3,
                "start": 0,
                "stop": 5,
                "compressed": True,
                "gpu": False,
                "next_index": 2,
                "part_size": len(committed),
            }
        ),
        encoding="utf-8",
    )

    assert (
        generator.main(
            [
                "3",
                "--output",
                str(output),
                "--start",
                "0",
                "--stop",
                "5",
                "--resume",
                "--progress-interval",
                "0",
            ]
        )
        == 0
    )
    with gzip.open(output, "rb") as handle:
        assert handle.read() == b"000\n001\n002\n003\n004\n"


def test_failure_preserves_existing_destination_and_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "values.txt"
    output.write_text("original\n", encoding="ascii")

    def fail_after_one_batch(length: int, start: int, stop: int, batch_size: int):
        del length, stop, batch_size
        yield start + 1, b"000\n"
        raise generator.GenerationError("injected failure")

    monkeypatch.setattr(generator, "generate_combinations_cpu", fail_after_one_batch)
    assert run_cli(output, "--start", "0", "--stop", "5") == 1
    assert output.read_text(encoding="ascii") == "original\n"
    assert Path(f"{output}.part").read_bytes() == b"000\n"
    state = json.loads(Path(f"{output}.checkpoint.json").read_text(encoding="utf-8"))
    assert state["next_index"] == 1


def test_resume_truncates_uncheckpointed_bytes_and_finishes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "values.txt"
    part = Path(f"{output}.part")
    checkpoint = Path(f"{output}.checkpoint.json")
    part.write_bytes(b"000\n001\npartial")
    checkpoint.write_text(
        json.dumps(
            {
                "version": 1,
                "length": 3,
                "start": 0,
                "stop": 5,
                "compressed": False,
                "gpu": False,
                "next_index": 2,
                "part_size": 8,
            }
        ),
        encoding="utf-8",
    )

    assert (
        generator.main(
            [
                "3",
                "--output",
                str(output),
                "--start",
                "0",
                "--stop",
                "5",
                "--resume",
                "--progress-interval",
                "0",
            ]
        )
        == 0
    )
    assert output.read_bytes() == b"000\n001\n002\n003\n004\n"


def test_gpu_chunk_argument_is_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "values.txt"
    observed: list[int] = []

    def fake_gpu(length: int, start: int, stop: int, max_chunk: int):
        observed.append(max_chunk)
        yield from generator.generate_combinations_cpu(length, start, stop, 100)

    monkeypatch.setattr(generator, "generate_combinations_gpu", fake_gpu)
    assert (
        run_cli(output, "--start", "0", "--stop", "2", "--gpu", "--max-gpu-chunk", "17")
        == 0
    )
    assert observed == [17]


def test_gpu_does_not_fallback_after_writing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "values.txt"

    def partial_gpu(length: int, start: int, stop: int, max_chunk: int):
        del length, stop, max_chunk
        yield start + 1, b"000\n"
        raise generator.GenerationError("injected CUDA failure")

    monkeypatch.setattr(generator, "generate_combinations_gpu", partial_gpu)
    assert (
        run_cli(output, "--start", "0", "--stop", "3", "--gpu", "--gpu-fallback") == 1
    )
    assert not output.exists()
    assert Path(f"{output}.part").read_bytes() == b"000\n"


def test_invalid_batch_size_returns_usage_error() -> None:
    with pytest.raises(SystemExit) as exc_info:
        generator.main(["3", "--batch-size", "0"])
    assert exc_info.value.code == 2


def test_yes_does_not_authorize_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "values.txt"
    output.write_text("keep me\n", encoding="ascii")
    observed_assume_yes: list[bool] = []

    def reject(prompt: str, assume_yes: bool) -> bool:
        del prompt
        observed_assume_yes.append(assume_yes)
        return False

    monkeypatch.setattr(generator, "confirm", reject)
    assert generator.main(["3", "--output", str(output), "--stop", "2", "--yes"]) == 1
    assert observed_assume_yes == [False]
    assert output.read_text(encoding="ascii") == "keep me\n"


def test_help_works_without_importing_optional_gpu_dependencies() -> None:
    result = subprocess.run(
        [sys.executable, "generateCombinations.py", "--help"],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--max-gpu-chunk" in result.stdout


def test_stdout_contains_only_generated_data() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "generateCombinations.py",
            "2",
            "--start",
            "8",
            "--stop",
            "12",
            "--output",
            "-",
            "--progress-interval",
            "0",
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == b"08\n09\n10\n11\n"
    assert result.stderr == b""


def test_disk_preflight_fails_before_creating_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "values.txt"
    monkeypatch.setattr(
        generator.shutil, "disk_usage", lambda path: SimpleNamespace(free=1)
    )
    assert run_cli(output, "--start", "0", "--stop", "2") == 1
    assert not output.exists()
    assert not Path(f"{output}.part").exists()
    assert not Path(f"{output}.checkpoint.json").exists()
