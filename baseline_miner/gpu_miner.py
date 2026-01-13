"""GPU-based miner implementation using OpenCL.

This module provides a GPUMiner class with a similar interface to the CPU Miner,
allowing easy switching between CPU and GPU mining.
"""

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .hashing import (
    compact_to_target,
    difficulty_to_target_bytes,
    sha256d,
    target_to_bytes,
)
from .job import MiningJob, Share

try:
    from .gpu import GPUHasher, HAS_OPENCL, list_devices
except ImportError:
    HAS_OPENCL = False
    GPUHasher = None
    list_devices = lambda: []


NONCE_LIMIT = 0x100000000
NTIME_UPDATE_INTERVAL = 1.0
SLEEP_NO_JOB = 0.1
MAX_SHARES_PER_BATCH = 64


def _merkle_root(coinbase_hash: bytes, branches: list[bytes]) -> bytes:
    merkle = coinbase_hash
    for sibling in branches:
        merkle = sha256d(merkle + sibling)
    return merkle


def _build_header_prefix(job: MiningJob, *, extranonce2: int, extranonce2_size: int, ntime: int) -> bytes:
    version_le = job.version.to_bytes(4, "little")
    bits_le = job.bits.to_bytes(4, "little")
    extranonce2_bytes = extranonce2.to_bytes(extranonce2_size, "big")
    coinbase = job.coinb1 + job.extranonce1 + extranonce2_bytes + job.coinb2
    coinbase_hash = sha256d(coinbase)
    merkle_root = _merkle_root(coinbase_hash, job.merkle_branches_le)
    return version_le + job.prev_hash_le + merkle_root + ntime.to_bytes(4, "little") + bits_le


def _gpu_worker_main(
    hasher: "GPUHasher",
    job_queue: queue.Queue,
    share_queue: queue.Queue,
    stop_event: threading.Event,
    hash_counter: list,  # [value] to allow modification
    counter_lock: threading.Lock,
    extranonce2_size: int,
    batch_size: int,
    worker_id: int,
    worker_count: int,
) -> None:
    """GPU worker thread main function."""
    share_target = difficulty_to_target_bytes(1.0)
    current_job: Optional[MiningJob] = None
    current_job_id: Optional[str] = None
    current_job_seq: int = 0
    ntime_base = 0
    block_target = target_to_bytes(1)
    max_extranonce2 = 1 << (extranonce2_size * 8)
    # Distribute nonce space across workers
    step = worker_count % max_extranonce2 or 1
    extranonce2 = worker_id % max_extranonce2
    nonce = 0
    last_ntime_update = 0.0
    rebuild_header = False
    header_prefix = b""
    current_ntime = 0

    while not stop_event.is_set():
        # Check for new jobs/settings
        while True:
            try:
                kind, payload = job_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "job":
                current_job_seq, current_job = payload
                if current_job is None:
                    continue
                current_job_id = current_job.job_id
                ntime_base = current_job.ntime
                block_target = target_to_bytes(compact_to_target(current_job.bits))
                extranonce2 = worker_id % max_extranonce2
                nonce = 0
                rebuild_header = True
                header_prefix = b""
                current_ntime = 0
            elif kind == "diff":
                share_target = difficulty_to_target_bytes(float(payload))
            elif kind == "clear":
                current_job = None
                current_job_id = None

        if current_job is None:
            time.sleep(SLEEP_NO_JOB)
            continue

        now = time.time()
        if rebuild_header or now - last_ntime_update >= NTIME_UPDATE_INTERVAL:
            ntime = max(ntime_base, int(now))
            header_prefix = _build_header_prefix(
                current_job,
                extranonce2=extranonce2,
                extranonce2_size=extranonce2_size,
                ntime=ntime,
            )
            current_ntime = ntime
            last_ntime_update = now
            rebuild_header = False
            
            # Update GPU hasher with new header
            try:
                hasher.set_header_prefix(header_prefix)
                hasher.set_target(share_target)
            except Exception:
                time.sleep(SLEEP_NO_JOB)
                continue
        else:
            ntime = current_ntime

        if not header_prefix:
            time.sleep(0)
            continue

        # Calculate batch size respecting nonce limit
        remaining = NONCE_LIMIT - nonce
        span = min(batch_size, remaining)
        
        try:
            matches = hasher.scan_nonces(nonce, span)
        except Exception:
            time.sleep(SLEEP_NO_JOB)
            continue
            
        emitted = 0
        for match_nonce, hash_bytes in matches:
            if stop_event.is_set() or emitted >= MAX_SHARES_PER_BATCH:
                break
            share = Share(
                job_id=current_job_id or "",
                extranonce2=extranonce2,
                ntime=ntime,
                nonce=match_nonce,
                is_block=hash_bytes <= block_target,
                hash_hex=hash_bytes[::-1].hex(),
                job_seq=current_job_seq,
            )
            try:
                share_queue.put_nowait(share)
                emitted += 1
            except queue.Full:
                continue
        
        with counter_lock:
            hash_counter[0] += span
        
        nonce += span
        if nonce >= NONCE_LIMIT:
            extranonce2 = (extranonce2 + step) % max_extranonce2
            nonce = 0
            rebuild_header = True


class GPUMiner:
    """GPU-based miner using OpenCL.

    This class provides a similar interface to the CPU Miner class,
    making it easy to switch between CPU and GPU mining.
    Supports multiple GPUs by creating a worker thread per GPU.
    """

    DEFAULT_BATCH_SIZE = 1 << 24  # 16M hashes per batch

    def __init__(
        self,
        extranonce2_size: int = 4,
        gpu_devices: Optional[list[tuple[int, int]]] = None,
        batch_size: Optional[int] = None,
        global_size: Optional[int] = None,
        local_size: Optional[int] = None,
    ):
        """Initialize the GPU miner.

        Args:
            extranonce2_size: Size of extranonce2 in bytes
            gpu_devices: List of (platform_idx, device_idx) tuples. If None, uses GPU 0:0
            batch_size: Number of nonces per mining batch
            global_size: OpenCL global work size
            local_size: OpenCL local work size
        """
        if not HAS_OPENCL:
            raise RuntimeError(
                "PyOpenCL is not installed. Install with: pip install pyopencl numpy"
            )

        self.extranonce2_size = extranonce2_size
        self.gpu_devices = gpu_devices or [(0, 0)]
        self.batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        self.global_size = global_size
        self.local_size = local_size

        self._hashers: list[Optional[GPUHasher]] = []
        self._worker_threads: list[Optional[threading.Thread]] = []
        self._job_queues: list[queue.Queue] = []
        self._share_queue: queue.Queue = queue.Queue(maxsize=10000)
        self._stop_event = threading.Event()
        self._hash_counters: list[list[int]] = []
        self._counter_locks: list[threading.Lock] = []

        self.current_job_id: Optional[str] = None
        self.job_seq: int = 0
        self.difficulty: float = 1.0
        self.share_target: bytes = difficulty_to_target_bytes(self.difficulty)
    
    @property
    def share_queue(self) -> queue.Queue:
        """Queue of found shares."""
        return self._share_queue

    def start(self) -> None:
        """Start GPU mining threads (one per GPU)."""
        if self._worker_threads and any(t and t.is_alive() for t in self._worker_threads):
            return

        # Reset state
        self._stop_event.clear()
        self._hashers.clear()
        self._worker_threads.clear()
        self._job_queues.clear()
        self._hash_counters.clear()
        self._counter_locks.clear()

        # Start one worker thread per GPU
        for gpu_idx, (platform_idx, device_idx) in enumerate(self.gpu_devices):
            # Initialize hasher for this GPU
            try:
                hasher = GPUHasher(
                    platform_idx=platform_idx,
                    device_idx=device_idx,
                    global_size=self.global_size,
                    local_size=self.local_size,
                )
                hasher.initialize()
            except Exception as e:
                print(f"Failed to initialize GPU {platform_idx}:{device_idx}: {e}")
                continue

            # Create job queue and counters for this worker
            job_queue = queue.Queue()
            hash_counter = [0]
            counter_lock = threading.Lock()

            # Start worker thread
            worker_thread = threading.Thread(
                target=_gpu_worker_main,
                args=(
                    hasher,
                    job_queue,
                    self._share_queue,
                    self._stop_event,
                    hash_counter,
                    counter_lock,
                    self.extranonce2_size,
                    self.batch_size,
                    gpu_idx,  # worker_id
                    len(self.gpu_devices),  # worker_count
                ),
                daemon=True,
                name=f"GPU-{platform_idx}:{device_idx}",
            )
            worker_thread.start()

            self._hashers.append(hasher)
            self._worker_threads.append(worker_thread)
            self._job_queues.append(job_queue)
            self._hash_counters.append(hash_counter)
            self._counter_locks.append(counter_lock)
    
    def stop(self) -> None:
        """Stop all GPU mining threads."""
        self._stop_event.set()

        # Signal all workers to stop
        for job_queue in self._job_queues:
            job_queue.put(("clear", None))

        # Wait for all threads to finish
        for worker_thread in self._worker_threads:
            if worker_thread is not None:
                worker_thread.join(timeout=5)

        # Release all hashers
        for hasher in self._hashers:
            if hasher is not None:
                hasher.release()

        self._worker_threads.clear()
        self._hashers.clear()

    def clear_job(self) -> None:
        """Clear the current mining job on all GPUs."""
        for job_queue in self._job_queues:
            job_queue.put(("clear", None))

    def _drain_shares(self) -> None:
        """Drain all pending shares from the queue."""
        while True:
            try:
                self._share_queue.get_nowait()
            except queue.Empty:
                break

    def set_difficulty(self, difficulty: float) -> None:
        """Set the mining difficulty on all GPUs."""
        previous = self.difficulty
        self.difficulty = float(difficulty)
        self.share_target = difficulty_to_target_bytes(self.difficulty)
        for job_queue in self._job_queues:
            job_queue.put(("diff", difficulty))
        if self.difficulty > previous:
            self._drain_shares()

    def snapshot_hashes(self) -> int:
        """Get the total number of hashes computed across all GPUs."""
        total = 0
        for counter, lock in zip(self._hash_counters, self._counter_locks):
            with lock:
                total += counter[0]
        return total

    def set_job(self, job: MiningJob) -> None:
        """Set a new mining job on all GPUs."""
        if job.clean:
            self.clear_job()
            self._drain_shares()
        self.job_seq += 1
        self.current_job_id = job.job_id
        for job_queue in self._job_queues:
            job_queue.put(("job", (self.job_seq, job)))


def get_miner(
    gpu: bool = False,
    threads: Optional[int] = None,
    extranonce2_size: int = 4,
    gpu_devices: Optional[list[tuple[int, int]]] = None,
    **kwargs,
):
    """Factory function to get either a CPU or GPU miner.

    Args:
        gpu: If True, return a GPUMiner; otherwise return CPU Miner
        threads: Number of CPU threads (CPU miner only)
        extranonce2_size: Size of extranonce2 in bytes
        gpu_devices: List of (platform_idx, device_idx) tuples (GPU miner only)
        **kwargs: Additional arguments passed to miner constructor

    Returns:
        Either a GPUMiner or CPU Miner instance
    """
    if gpu:
        return GPUMiner(
            extranonce2_size=extranonce2_size,
            gpu_devices=gpu_devices,
            **kwargs,
        )
    else:
        from .miner import Miner
        return Miner(threads=threads, extranonce2_size=extranonce2_size)
