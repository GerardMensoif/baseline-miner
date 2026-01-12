import argparse
import multiprocessing as mp
import os
import threading
import time

from .hashing import BACKEND, scan_hashes, sha256d

try:
    from .gpu import HAS_OPENCL, GPUHasher, list_devices as list_gpu_devices
except ImportError:
    HAS_OPENCL = False
    GPUHasher = None
    list_gpu_devices = lambda: []

BATCH_SIZE = 10000
GPU_BATCH_SIZE = 1 << 24  # 16M for GPU
DEFAULT_SECONDS = 10.0
DEFAULT_DATA_SIZE = 80


def _worker(stop: mp.Event, counter: mp.Value, data: bytes, mode: str) -> None:
    local = 0
    nonce = 0
    header_prefix = b""
    target = b""
    if mode == "scan":
        header_prefix = os.urandom(76)
        target = b"\x00" * 32
    while not stop.is_set():
        if mode == "scan":
            scan_hashes(header_prefix, nonce, BATCH_SIZE, target)
            nonce = (nonce + BATCH_SIZE) & 0xFFFFFFFF
            local += BATCH_SIZE
        else:
            sha256d(data)
            local += 1
        if local >= BATCH_SIZE:
            with counter.get_lock():
                counter.value += local
            local = 0
    if local:
        with counter.get_lock():
            counter.value += local


def _gpu_worker(
    stop: threading.Event,
    counter: list,
    counter_lock: threading.Lock,
    platform_idx: int,
    device_idx: int,
    batch_size: int,
) -> None:
    """GPU benchmark worker."""
    hasher = GPUHasher(
        platform_idx=platform_idx,
        device_idx=device_idx,
    )
    hasher.initialize()
    
    header_prefix = os.urandom(76)
    target = b"\x00" * 32  # Very easy target - accept all
    hasher.set_header_prefix(header_prefix)
    hasher.set_target(target)
    
    nonce = 0
    local = 0
    
    while not stop.is_set():
        remaining = 0x100000000 - nonce
        span = min(batch_size, remaining)
        
        hasher.scan_nonces(nonce, span)
        local += span
        nonce = (nonce + span) & 0xFFFFFFFF
        
        if nonce == 0:
            # Wrapped around, update header to get new midstate
            header_prefix = os.urandom(76)
            hasher.set_header_prefix(header_prefix)
        
        if local >= batch_size:
            with counter_lock:
                counter[0] += local
            local = 0
    
    if local:
        with counter_lock:
            counter[0] += local
    
    hasher.release()


def _run_gpu_benchmark(args: argparse.Namespace) -> None:
    """Run GPU benchmark."""
    if not HAS_OPENCL:
        print("OpenCL not available. Install pyopencl and numpy for GPU support.")
        return
    
    devices = list_gpu_devices()
    if not devices:
        print("No OpenCL GPU devices found.")
        return
    
    platform_idx = args.gpu_platform
    device_idx = args.gpu_device
    
    platform_devices = [d for d in devices if d.platform_idx == platform_idx]
    if not platform_devices:
        print(f"No GPU devices found on platform {platform_idx}")
        return
    if device_idx >= len(platform_devices):
        print(f"Invalid device index {device_idx}")
        return
    
    selected = platform_devices[device_idx]
    print(f"GPU: {selected.name}")
    print(f"Platform: {selected.platform_name}")
    
    batch_size = args.gpu_batch_size or GPU_BATCH_SIZE
    
    stop_event = threading.Event()
    counter = [0]
    counter_lock = threading.Lock()
    
    thread = threading.Thread(
        target=_gpu_worker,
        args=(stop_event, counter, counter_lock, platform_idx, device_idx, batch_size),
        daemon=True,
    )
    thread.start()
    
    start = time.monotonic()
    time.sleep(max(0.1, args.seconds))
    stop_event.set()
    thread.join(timeout=10)
    elapsed = max(0.001, time.monotonic() - start)
    
    with counter_lock:
        total = counter[0]
    rate = total / elapsed
    
    print(f"Backend: OpenCL GPU")
    print(f"Batch size: {batch_size}")
    print(f"Total hashes: {total}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Hashrate: {rate:.2f} H/s ({rate/1e6:.2f} MH/s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline miner hashing benchmark")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--mode", choices=["scan", "sha256d"], default="scan")
    parser.add_argument("--data-size", type=int, default=DEFAULT_DATA_SIZE)
    
    # GPU options
    parser.add_argument("--gpu", action="store_true", help="Benchmark GPU instead of CPU")
    parser.add_argument("--gpu-platform", type=int, default=0, help="OpenCL platform index")
    parser.add_argument("--gpu-device", type=int, default=0, help="GPU device index")
    parser.add_argument("--gpu-batch-size", type=int, default=None, 
                        help=f"GPU batch size (default: {GPU_BATCH_SIZE})")
    parser.add_argument("--list-devices", action="store_true", help="List available GPU devices")
    
    args = parser.parse_args()
    
    if args.list_devices:
        if not HAS_OPENCL:
            print("OpenCL not available. Install pyopencl and numpy for GPU support.")
            return
        devices = list_gpu_devices()
        if not devices:
            print("No OpenCL GPU devices found.")
            return
        print("Available OpenCL GPU devices:")
        for dev in devices:
            print(f"  {dev}")
        return
    
    if args.gpu:
        _run_gpu_benchmark(args)
        return

    data_size = max(1, args.data_size)
    payload = os.urandom(data_size)

    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    counters: list[mp.Value] = []
    processes: list[mp.Process] = []

    for _ in range(max(1, args.threads)):
        counter = ctx.Value("Q", 0)
        proc = ctx.Process(
            target=_worker,
            args=(stop_event, counter, payload, args.mode),
            daemon=True,
        )
        proc.start()
        counters.append(counter)
        processes.append(proc)

    start = time.monotonic()
    time.sleep(max(0.1, args.seconds))
    stop_event.set()
    for proc in processes:
        proc.join()
    elapsed = max(0.001, time.monotonic() - start)
    total = sum(counter.value for counter in counters)
    rate = total / elapsed

    print(f"Backend: {BACKEND}")
    print(f"Threads: {len(processes)}")
    print(f"Mode: {args.mode}")
    if args.mode != "scan":
        print(f"Data size: {data_size} bytes")
    print(f"Total hashes: {total}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Hashrate: {rate:.2f} H/s")


if __name__ == "__main__":
    main()
