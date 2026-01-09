# Baseline Miner

Open source Stratum miner for Baseline pools, optimized for CPU throughput with multi-process hashing.

## Features
- Baseline Stratum client (subscribe/authorize/notify)
- Multi-process SHA256d miner (one worker per process)
- Portable C SHA256d backend (always used; no CPU-specific intrinsics)
- Vardiff `mining.set_difficulty` support
- Clean job handling and share validation

## Requirements
- Python 3.9+
- A C compiler (builds the portable hashing extension)

## Install
```
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

## Usage
```
baseline-miner --host 127.0.0.1 --port 3333 --address <BLINE_ADDRESS> --worker rig1
```

### Common options
- `--threads` number of worker processes (default: CPU count)
- `--password` stratum password (optional)
- `--stats-interval` seconds between hashrate reports
- `--log-level` debug|info|warning|error

## Benchmark
```
baseline-miner-bench --seconds 10 --threads 4
```
The default mode uses the native batch scan path (`--mode scan`). Use `--mode sha256d` to benchmark standalone hashing.

## Tests
```
python -m unittest discover -s tests
```

## Baseline-specific notes
- Proof-of-work is SHA256d like Bitcoin, but the block interval target is 20 seconds.
- The pow limit bits default is `0x207fffff` (used for share target calculations).
- Coinbase payouts include a foundation output; the pool handles this in templates.
- Address version `0x35` is the Baseline prefix; pass a Baseline address for payouts.

## License
MIT
