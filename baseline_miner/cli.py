import argparse
import asyncio
import contextlib
from collections import deque
import logging
import os
import queue
import signal
import time

from .miner import Miner
from .stratum import StratumClient
from . import hashing

try:
    from .gpu import HAS_OPENCL, list_devices as list_gpu_devices
    from .gpu_miner import GPUMiner
except ImportError:
    HAS_OPENCL = False
    list_gpu_devices = lambda: []
    GPUMiner = None


def _build_worker_name(address: str, worker: str | None) -> str:
    if worker:
        return f"{address}.{worker}"
    return address


def _format_hashrate(hashrate: float) -> str:
    """Format hashrate with appropriate unit (H/s, KH/s, MH/s, GH/s)."""
    if hashrate >= 1e9:
        return f"{hashrate / 1e9:.2f} GH/s"
    elif hashrate >= 1e6:
        return f"{hashrate / 1e6:.2f} MH/s"
    elif hashrate >= 1e3:
        return f"{hashrate / 1e3:.2f} KH/s"
    else:
        return f"{hashrate:.2f} H/s"


async def _share_sender(
    client: StratumClient,
    miner: Miner,
    worker_name: str,
    stats: dict[str, int],
    stop_event: asyncio.Event,
) -> None:
    recent_keys: set[tuple[str, int, int, int]] = set()
    recent_order: deque[tuple[str, int, int, int]] = deque()
    max_recent = 50_000

    while client.connected and not stop_event.is_set():
        try:
            share = await asyncio.to_thread(miner.share_queue.get, True, 0.5)
        except queue.Empty:
            continue
        if stop_event.is_set() or not client.connected:
            break
        if share.hash_hex:
            # Server validates shares by recomputing the header hash and comparing
            # against the current share target; drop any shares that don't meet
            # our current difficulty (e.g. mined right before a vardiff increase).
            digest_be = bytes.fromhex(share.hash_hex)[::-1]
            if digest_be > miner.share_target:
                continue
        key = (share.job_id, share.extranonce2, share.ntime, share.nonce)
        if key in recent_keys:
            continue
        recent_keys.add(key)
        recent_order.append(key)
        if len(recent_order) > max_recent:
            old = recent_order.popleft()
            recent_keys.discard(old)
        ok = await client.submit_share(share, worker_name, miner.extranonce2_size)
        if ok:
            stats["accepted"] += 1
        else:
            stats["rejected"] += 1
        if share.is_block:
            stats["blocks"] += 1
            logging.getLogger("baseline_miner").info("Block candidate found: %s", share.hash_hex)


async def _stats_loop(miner: Miner, stats: dict[str, int], interval: float, stop_event: asyncio.Event) -> None:
    log = logging.getLogger("baseline_miner")
    last_hashes = miner.snapshot_hashes()
    last_time = time.monotonic()
    avg_rate: float | None = None
    while not stop_event.is_set():
        await asyncio.sleep(interval)
        total_hashes = miner.snapshot_hashes()
        now = time.monotonic()
        delta_hashes = total_hashes - last_hashes
        elapsed = max(0.001, now - last_time)
        rate = delta_hashes / elapsed
        if avg_rate is None:
            avg_rate = rate
        else:
            # Smooth with exponential moving average for more stable display.
            avg_rate = avg_rate * 0.7 + rate * 0.3
        log.info(
            "Hashrate inst=%s avg=%s | shares ok=%d rejected=%d blocks=%d",
            _format_hashrate(rate),
            _format_hashrate(avg_rate),
            stats["accepted"],
            stats["rejected"],
            stats["blocks"],
        )
        last_hashes = total_hashes
        last_time = now


async def run(args: argparse.Namespace) -> None:
    log = logging.getLogger("baseline_miner")
    stop_event = asyncio.Event()
    worker_name = _build_worker_name(args.address, args.worker)
    miner: Miner | GPUMiner | None = None
    stats = {"accepted": 0, "rejected": 0, "blocks": 0}
    latest_difficulty: float | None = None
    latest_job: object | None = None
    use_gpu = getattr(args, 'gpu', False)

    stats_task: asyncio.Task | None = None
    client: StratumClient | None = None

    try:
        while not stop_event.is_set():
            client = StratumClient(args.host, args.port)
            latest_difficulty = None
            latest_job = None
            def _on_difficulty(diff: float) -> None:
                nonlocal latest_difficulty
                latest_difficulty = diff
                if miner:
                    miner.set_difficulty(diff)

            def _on_job(job) -> None:
                nonlocal latest_job
                latest_job = job
                if miner:
                    miner.set_job(job)

            client.on_difficulty = _on_difficulty
            client.on_job = _on_job
            share_task: asyncio.Task | None = None
            try:
                await client.connect()
                log.info("Connected to %s:%s", args.host, args.port)
                await client.subscribe()
                await client.authorize(worker_name, args.password)
                if client.extranonce2_size is None:
                    raise RuntimeError("Missing extranonce2 size")
                if miner is None or miner.extranonce2_size != client.extranonce2_size:
                    if miner:
                        miner.stop()
                    if use_gpu:
                        if GPUMiner is None:
                            raise RuntimeError("GPU support not available. Install pyopencl and numpy.")
                        miner = GPUMiner(
                            extranonce2_size=client.extranonce2_size,
                            platform_idx=getattr(args, 'gpu_platform', 0),
                            device_idx=getattr(args, 'gpu_device', 0),
                        )
                        log.info("Using GPU mining")
                    else:
                        miner = Miner(threads=args.threads, extranonce2_size=client.extranonce2_size)
                        log.info("Using CPU mining with %d threads", args.threads)
                    miner.start()
                if latest_difficulty is not None:
                    miner.set_difficulty(latest_difficulty)
                if latest_job is not None:
                    miner.set_job(latest_job)
                if stats_task:
                    stats_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await stats_task
                stats_task = asyncio.create_task(
                        _stats_loop(miner, stats, args.stats_interval, stop_event),
                        name="miner-stats",
                    )
                share_task = asyncio.create_task(
                    _share_sender(client, miner, worker_name, stats, stop_event),
                    name="share-sender",
                )
                await client.wait_closed()
            except Exception as exc:
                log.warning("Disconnected: %s", exc)
            finally:
                if share_task:
                    share_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await share_task
                await client.close()
                if miner:
                    miner.clear_job()
            if not stop_event.is_set():
                await asyncio.sleep(args.reconnect_delay)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        if client:
            with contextlib.suppress(Exception):
                await client.close()
        if miner:
            miner.stop()
        if stats_task:
            stats_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stats_task


def _list_devices_cmd() -> None:
    """List available GPU devices."""
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
    print()
    print("Use --gpu --gpu-platform=P --gpu-device=D to select a device")


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline Stratum miner (CPU/GPU)")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command")
    
    # list-devices subcommand
    subparsers.add_parser("list-devices", help="List available GPU devices")
    
    # Main miner arguments
    parser.add_argument("--host", default=os.environ.get("BASELINE_STRATUM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("BASELINE_STRATUM_PORT", "3333")))
    parser.add_argument("--address", help="Baseline payout address")
    parser.add_argument("--worker", default=os.environ.get("BASELINE_WORKER"))
    parser.add_argument("--password", default=os.environ.get("BASELINE_PASSWORD", ""))
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1,
                        help="Number of CPU threads (CPU mining only)")
    parser.add_argument("--stats-interval", type=float, default=10.0)
    parser.add_argument("--reconnect-delay", type=float, default=5.0)
    parser.add_argument("--log-level", default=os.environ.get("BASELINE_LOG_LEVEL", "info"))
    
    # GPU arguments
    parser.add_argument("--gpu", action="store_true", 
                        help="Enable GPU mining using OpenCL")
    parser.add_argument("--gpu-platform", type=int, default=0,
                        help="OpenCL platform index (default: 0)")
    parser.add_argument("--gpu-device", type=int, default=0,
                        help="GPU device index within platform (default: 0)")
    
    args = parser.parse_args()
    
    # Handle subcommands
    if args.command == "list-devices":
        _list_devices_cmd()
        return
    
    # Regular mining mode - address is required
    if not args.address:
        parser.error("--address is required for mining")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("baseline_miner")
    
    if args.gpu:
        if not HAS_OPENCL:
            log.error("GPU mining requested but OpenCL not available.")
            log.error("Install with: pip install pyopencl numpy")
            return
        devices = list_gpu_devices()
        if not devices:
            log.error("No OpenCL GPU devices found.")
            return
        if args.gpu_platform >= len(set(d.platform_idx for d in devices)):
            log.error("Invalid GPU platform index %d", args.gpu_platform)
            return
        platform_devices = [d for d in devices if d.platform_idx == args.gpu_platform]
        if args.gpu_device >= len(platform_devices):
            log.error("Invalid GPU device index %d", args.gpu_device)
            return
        selected = platform_devices[args.gpu_device]
        log.info("Selected GPU: %s", selected)
    else:
        log.info("Native hashing backend: %s", getattr(hashing, "BACKEND", "unknown"))
    
    log.info(
        "If you see lots of rejected shares, consider upping the min_difficulty on the stratum server"
    )

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
