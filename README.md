# CoreStorm Combination Generator

CoreStorm writes fixed-width decimal combinations in ascending order. It supports
dependency-free CPU generation, optional CUDA acceleration, atomic output,
checkpoints, resumable ranges, gzip, stdout, and automation-friendly exit codes.

Use this software only for systems and data you own or are explicitly authorized
to test.

## Installation

Python 3.10 or newer is required. CoreStorm runs natively on Linux, Windows,
and macOS, including Apple silicon (arm64); use an arm64 Python build there rather
than one running under Rosetta.

```bash
python -m pip install -e .
```

CUDA mode is optional. It requires an NVIDIA CUDA environment plus NumPy and
Numba:

```bash
python -m pip install -e ".[gpu]"
```

CPU mode and even `--help` do not import those GPU dependencies.

CUDA requires an NVIDIA GPU, so `--gpu` is not available on macOS, including
Apple silicon. Use the default CPU mode there.

## Usage

Generate `0000` through `9999`:

```bash
python generateCombinations.py 4
```

Choose an output file and suppress the large-job prompt in automation:

```bash
python generateCombinations.py 6 --output values.txt --yes
```

Generate an inclusive/exclusive range:

```bash
python generateCombinations.py 8 --start 10000000 --stop 20000000 --output part-1.txt --yes
```

Stream to another program rather than creating a file:

```bash
python generateCombinations.py 4 --output - --progress-interval 0 | your-program
```

Progress and errors use stderr, so stdout contains only generated data.

Write compressed output by using `--gzip` or a `.gz` suffix:

```bash
python generateCombinations.py 7 --output values.txt.gz --yes
```

The compressed file consists of standard concatenated gzip members and can be
read normally by gzip-compatible tools.

## Safe interruption and resume

File output is first written beside the destination as:

- `<output>.part`
- `<output>.checkpoint.json`

The completed destination is replaced atomically only after generation and a
final disk flush succeed. If a run is interrupted or fails, the previous output
remains untouched and the sidecar files remain available:

```bash
python generateCombinations.py 8 --output values.txt --yes
# After interruption, use exactly the same range/mode options:
python generateCombinations.py 8 --output values.txt --resume --yes
```

On macOS, durable flushes use `F_FULLFSYNC`, because a plain `fsync` there
does not guarantee data has reached stable storage. Filesystems that reject it
fall back to `fsync`.

Resume validates the width, range, compression, and CPU/GPU mode. It truncates
any bytes written after the last committed checkpoint before continuing.

Use `--force` to intentionally replace completed or incomplete output without a
prompt. `--yes` accepts only the capacity warning and never implies overwrite
permission.

## CUDA mode

```bash
python generateCombinations.py 8 --gpu --max-gpu-chunk 5000000 --output values.txt --yes
```

`--max-gpu-chunk` is a hard upper limit. CoreStorm also caps the allocation to
half of currently free GPU memory. Digits are held as bytes and serialized to
ASCII with NumPy rather than converted row-by-row in Python.

GPU failures return a nonzero status by default. To permit CPU fallback before
the GPU has produced any output, opt in explicitly:

```bash
python generateCombinations.py 6 --gpu --gpu-fallback --output values.txt --yes
```

CoreStorm will not silently fall back after a GPU batch has been written, because
restarting at that point could duplicate output.

## Capacity planning

Uncompressed output requires exactly:

```text
number of combinations × (width + 1) bytes
```

The extra byte is the LF newline. For example, all eight-digit values require
900,000,000 bytes (about 858.31 MiB). CoreStorm checks available disk space before
starting file output. The same conservative uncompressed estimate is used for
gzip jobs.

Jobs of at least one million combinations require confirmation unless `--yes` is
provided. Width is limited to 18 digits to reject accidental, impractical jobs.

## Options

Run the authoritative help output with:

```bash
python generateCombinations.py --help
```

Important options include:

- `--start N` and `--stop N`: inclusive/exclusive numeric boundaries.
- `--batch-size N`: number of CPU rows encoded per write batch.
- `--max-gpu-chunk N`: maximum rows allocated in a CUDA batch.
- `--progress-interval SECONDS`: reporting interval; `0` disables progress.
- `--resume`: continue from matching sidecar files.
- `--force`: intentionally replace existing completed/incomplete output.
- `--yes`: accept the large-job confirmation.
- `--output -`: emit data to stdout.

Expected errors return status `1`, invalid CLI usage returns `2`, and an interrupt
returns `130`.

## Development and verification

Install development tools and run the deterministic checks:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest
```

CI runs lint and tests on Linux, Windows, and macOS (Apple silicon) with Python
3.10 and 3.13.

The benchmark writes to the real temporary filesystem, verifies the resulting
file with SHA-256, and reports end-to-end throughput:

```bash
python benchmarks/benchmark.py --length 6 --count 1000000
```

Benchmark CPU and CUDA output on the same machine, destination drive, width, and
range before selecting a mode. Generation arithmetic is cheap; storage throughput
can be the limiting factor.

## License

MIT. See [LICENSE](LICENSE).
