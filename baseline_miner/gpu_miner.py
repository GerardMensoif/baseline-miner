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
) -> None:
    """GPU worker thread main function."""
    share_target = difficulty_to_target_bytes(1.0)
    current_job: Optional[MiningJob] = None
    current_job_id: Optional[str] = None
    current_job_seq: int = 0
    ntime_base = 0
    block_target = target_to_bytes(1)
    max_extranonce2 = 1 << (extranonce2_size * 8)
    extranonce2 = 0
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
                extranonce2 = 0
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
            extranonce2 = (extranonce2 + 1) % max_extranonce2
            nonce = 0
            rebuild_header = True


class GPUMiner:
    """GPU-based miner using OpenCL.
    
    This class provides a similar interface to the CPU Miner class,
    making it easy to switch between CPU and GPU mining.
    """
    
    DEFAULT_BATCH_SIZE = 1 << 24  # 16M hashes per batch
    
    def __init__(
        self,
        extranonce2_size: int = 4,
        platform_idx: int = 0,
        device_idx: int = 0,
        batch_size: Optional[int] = None,
        global_size: Optional[int] = None,
        local_size: Optional[int] = None,
    ):
        """Initialize the GPU miner.
        
        Args:
            extranonce2_size: Size of extranonce2 in bytes
            platform_idx: OpenCL platform index
            device_idx: Device index within platform
            batch_size: Number of nonces per mining batch
            global_size: OpenCL global work size
            local_size: OpenCL local work size
        """
        if not HAS_OPENCL:
            raise RuntimeError(
                "PyOpenCL is not installed. Install with: pip install pyopencl numpy"
            )
        
        self.extranonce2_size = extranonce2_size
        self.platform_idx = platform_idx
        self.device_idx = device_idx
        self.batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        self.global_size = global_size
        self.local_size = local_size
        
        self._hasher: Optional[GPUHasher] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._job_queue: queue.Queue = queue.Queue()
        self._share_queue: queue.Queue = queue.Queue(maxsize=10000)
        self._stop_event = threading.Event()
        self._hash_counter = [0]  # Use list for mutability
        self._counter_lock = threading.Lock()
        
        self.current_job_id: Optional[str] = None
        self.job_seq: int = 0
        self.difficulty: float = 1.0
        self.share_target: bytes = difficulty_to_target_bytes(self.difficulty)
    
    @property
    def share_queue(self) -> queue.Queue:
        """Queue of found shares."""
        return self._share_queue
    
    def start(self) -> None:
        """Start the GPU mining thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        
        # Initialize hasher
        self._hasher = GPUHasher(
            platform_idx=self.platform_idx,
            device_idx=self.device_idx,
            global_size=self.global_size,
            local_size=self.local_size,
        )
        self._hasher.initialize()
        
        # Reset state
        self._stop_event.clear()
        self._hash_counter[0] = 0
        
        # Start worker thread
        self._worker_thread = threading.Thread(
            target=_gpu_worker_main,
            args=(
                self._hasher,
                self._job_queue,
                self._share_queue,
                self._stop_event,
                self._hash_counter,
                self._counter_lock,
                self.extranonce2_size,
                self.batch_size,
            ),
            daemon=True,
        )
        self._worker_thread.start()
    
    def stop(self) -> None:
        """Stop the GPU mining thread."""
        self._stop_event.set()
        self._job_queue.put(("clear", None))
        
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5)
            self._worker_thread = None
        
        if self._hasher is not None:
            self._hasher.release()
            self._hasher = None
    
    def clear_job(self) -> None:
        """Clear the current mining job."""
        self._job_queue.put(("clear", None))
    
    def _drain_shares(self) -> None:
        """Drain all pending shares from the queue."""
        while True:
            try:
                self._share_queue.get_nowait()
            except queue.Empty:
                break
    
    def set_difficulty(self, difficulty: float) -> None:
        """Set the mining difficulty."""
        previous = self.difficulty
        self.difficulty = float(difficulty)
        self.share_target = difficulty_to_target_bytes(self.difficulty)
        self._job_queue.put(("diff", difficulty))
        if self.difficulty > previous:
            self._drain_shares()
    
    def snapshot_hashes(self) -> int:
        """Get the total number of hashes computed."""
        with self._counter_lock:
            return self._hash_counter[0]
    
    def set_job(self, job: MiningJob) -> None:
        """Set a new mining job."""
        if job.clean:
            self.clear_job()
            self._drain_shares()
        self.job_seq += 1
        self.current_job_id = job.job_id
        self._job_queue.put(("job", (self.job_seq, job)))


def get_miner(
    gpu: bool = False,
    threads: Optional[int] = None,
    extranonce2_size: int = 4,
    platform_idx: int = 0,
    device_idx: int = 0,
    **kwargs,
):
    """Factory function to get either a CPU or GPU miner.
    
    Args:
        gpu: If True, return a GPUMiner; otherwise return CPU Miner
        threads: Number of CPU threads (CPU miner only)
        extranonce2_size: Size of extranonce2 in bytes
        platform_idx: OpenCL platform index (GPU miner only)
        device_idx: Device index (GPU miner only)
        **kwargs: Additional arguments passed to miner constructor
    
    Returns:
        Either a GPUMiner or CPU Miner instance
    """
    if gpu:
        return GPUMiner(
            extranonce2_size=extranonce2_size,
            platform_idx=platform_idx,
            device_idx=device_idx,
            **kwargs,
        )
    else:
        from .miner import Miner
        return Miner(threads=threads, extranonce2_size=extranonce2_size)
