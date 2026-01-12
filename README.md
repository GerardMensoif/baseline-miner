# Baseline Miner

Open source Stratum miner for Baseline pools, optimized for CPU and GPU throughput.

## Features
- Baseline Stratum client (subscribe/authorize/notify)
- Multi-process SHA256d CPU miner (one worker per process)
- **GPU mining support via OpenCL** (AMD, NVIDIA, Intel GPUs)
- C SHA256d backend (portable by default; optimized scan path)
- Vardiff `mining.set_difficulty` support
- Clean job handling and share validation

## Requirements
- Python 3.9+
- A C compiler (builds the portable hashing extension)

### GPU Mining (Optional)
- OpenCL runtime and drivers
- PyOpenCL and NumPy (`pip install pyopencl numpy`)

## Install

### CPU Mining Only
```
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

### With GPU Support
```
python -m venv .venv
. .venv/bin/activate
pip install -e ".[gpu]"
```

### Optional: CPU-specific build flags
By default the extension is built in a portable mode (safe to run on other machines).

To squeeze extra performance on the machine you build on, set `BASELINE_MINER_NATIVE=1` to enable CPU-specific compiler flags:

PowerShell (Windows):
```
$env:BASELINE_MINER_NATIVE="1"
pip install -e .
```

bash/zsh (Linux/macOS):
```
BASELINE_MINER_NATIVE=1 pip install -e .
```

Note: CPU-specific builds may crash with `Illegal instruction` if you copy the wheel to an older CPU.

## Usage

### CPU Mining
```
baseline-miner --host 127.0.0.1 --port 3333 --address <BLINE_ADDRESS> --worker rig1
```

### GPU Mining
```
# List available GPU devices
baseline-miner list-devices

# Mine with default GPU (platform 0, device 0)
baseline-miner --gpu --host 127.0.0.1 --port 3333 --address <BLINE_ADDRESS>

# Mine with specific GPU
baseline-miner --gpu --gpu-platform 0 --gpu-device 1 --host 127.0.0.1 --port 3333 --address <BLINE_ADDRESS>
```

### Common options
- `--threads` number of worker processes (CPU mining only, default: CPU count)
- `--gpu` enable GPU mining using OpenCL
- `--gpu-platform` OpenCL platform index (default: 0)
- `--gpu-device` GPU device index within platform (default: 0)
- `--password` stratum password (optional)
- `--stats-interval` seconds between hashrate reports
- `--log-level` debug|info|warning|error

## Benchmark

### CPU Benchmark
```
baseline-miner-bench --seconds 10 --threads 4
```

### GPU Benchmark
```
# List available devices
baseline-miner-bench --list-devices

# Benchmark GPU
baseline-miner-bench --gpu --seconds 10

# Benchmark specific GPU
baseline-miner-bench --gpu --gpu-platform 0 --gpu-device 0 --seconds 10
```

The default mode uses the native batch scan path (`--mode scan`). Use `--mode sha256d` to benchmark standalone hashing.

## Tests
```
python -m unittest discover -s tests
```

## GPU Notes
The OpenCL kernel exactly replicates the CPU SHA256d algorithm, ensuring:
- Identical hash output for the same inputs
- Correct endianness handling (critical for Baseline compatibility)
- Proper midstate optimization for mining efficiency

The GPU implementation is compatible with AMD, NVIDIA, and Intel GPUs that support OpenCL 1.2+.

## Baseline-specific notes
- Proof-of-work is SHA256d like Bitcoin, but the block interval target is 20 seconds.
- The pow limit bits default is `0x207fffff` (used for share target calculations).
- Coinbase payouts include a foundation output; the pool handles this in templates.
- Address version `0x35` is the Baseline prefix; pass a Baseline address for payouts.

## License
MIT
