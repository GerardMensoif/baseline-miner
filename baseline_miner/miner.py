import multiprocessing as mp
import os
import queue
import signal
import struct
import time

from .hashing import (
    HAS_SCAN,
    compact_to_target,
    difficulty_to_target_bytes,
    scan_hashes,
    sha256d,
    target_to_bytes,
)
from .job import MiningJob, Share

NONCE_LIMIT = 0x100000000
CHUNK_SIZE = 20000
NTIME_UPDATE_INTERVAL = 1.0
SLEEP_NO_JOB = 0.1
# Cap per-chunk share emission to avoid Python overhead at very low difficulty.
# This reduces reported shares when targets are extremely easy but keeps hashing hot.
MAX_SHARES_PER_CHUNK = 64


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
    # Baseline node uses big-endian digest bytes for header merkle root / prev_hash fields.
    return version_le + job.prev_hash_le + merkle_root + ntime.to_bytes(4, "little") + bits_le


def _worker_main(
    worker_id: int,
    worker_count: int,
    job_queue: mp.Queue,
    share_queue: mp.Queue,
    stop_event: mp.Event,
    hash_counter: mp.Value,
    extranonce2_size: int,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    share_target = difficulty_to_target_bytes(1.0)
    current_job: MiningJob | None = None
    current_job_id: str | None = None
    current_job_seq: int = 0
    ntime_base = 0
    block_target = target_to_bytes(1)
    max_extranonce2 = 1 << (extranonce2_size * 8)
    step = worker_count % max_extranonce2 or 1
    extranonce2 = worker_id % max_extranonce2
    nonce = 0
    last_ntime_update = 0.0
    rebuild_header = False
    header: bytearray | None = None
    header_prefix = b""
    current_ntime = 0

    while not stop_event.is_set():
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
                header = None
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
            header = None
            current_ntime = ntime
            last_ntime_update = now
            rebuild_header = False
        else:
            ntime = current_ntime

        if not header_prefix:
            time.sleep(0)
            continue

        remaining = NONCE_LIMIT - nonce
        span = CHUNK_SIZE if remaining > CHUNK_SIZE else remaining
        if HAS_SCAN:
            matches = scan_hashes(header_prefix, nonce, span, share_target)
            emitted = 0
            for match_nonce, hash_bytes in matches:
                if stop_event.is_set() or emitted >= MAX_SHARES_PER_CHUNK:
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
                    # If queue is full, drop share to keep hashing responsive.
                    continue
            nonce += span
        else:
            if header is None:
                header = bytearray(header_prefix + b"\x00\x00\x00\x00")
            for offset in range(span):
                current_nonce = nonce + offset
                struct.pack_into("<I", header, 76, current_nonce)
                hash_bytes = sha256d(header)
                if hash_bytes <= share_target:
                    share = Share(
                        job_id=current_job_id or "",
                        extranonce2=extranonce2,
                        ntime=ntime,
                        nonce=current_nonce,
                        is_block=hash_bytes <= block_target,
                        hash_hex=hash_bytes[::-1].hex(),
                        job_seq=current_job_seq,
                    )
                    try:
                        share_queue.put_nowait(share)
                    except queue.Full:
                        if stop_event.is_set():
                            break
                        continue
            nonce += span
        with hash_counter.get_lock():
            hash_counter.value += span

        if nonce >= NONCE_LIMIT:
            extranonce2 = (extranonce2 + step) % max_extranonce2
            nonce = 0
            rebuild_header = True


class Miner:
    def __init__(self, threads: int | None = None, extranonce2_size: int = 4):
        self.threads = threads or os.cpu_count() or 1
        self.extranonce2_size = extranonce2_size
        self.ctx = mp.get_context("spawn")
        self.job_queues: list[mp.Queue] = []
        self.share_queue: mp.Queue = self.ctx.Queue()
        self.stop_event = self.ctx.Event()
        self.processes: list[mp.Process] = []
        self.hash_counters: list[mp.Value] = []
        self.current_job_id: str | None = None
        self.job_seq: int = 0
        self.difficulty: float = 1.0
        self.share_target: bytes = difficulty_to_target_bytes(self.difficulty)

    def start(self) -> None:
        if self.processes:
            return
        for worker_id in range(self.threads):
            job_queue = self.ctx.Queue()
            hash_counter = self.ctx.Value("Q", 0, lock=True)
            proc = self.ctx.Process(
                target=_worker_main,
                args=(
                    worker_id,
                    self.threads,
                    job_queue,
                    self.share_queue,
                    self.stop_event,
                    hash_counter,
                    self.extranonce2_size,
                ),
                daemon=True,
            )
            proc.start()
            self.job_queues.append(job_queue)
            self.hash_counters.append(hash_counter)
            self.processes.append(proc)

    def stop(self) -> None:
        self.stop_event.set()
        for queue_item in self.job_queues:
            queue_item.put(("clear", None))
        for proc in self.processes:
            proc.join(timeout=2)
            if proc.is_alive():
                proc.terminate()
        self.processes.clear()
        for queue_item in self.job_queues:
            queue_item.close()
            queue_item.join_thread()
        self.job_queues.clear()
        self.share_queue.close()
        self.share_queue.join_thread()

    def clear_job(self) -> None:
        for queue_item in self.job_queues:
            queue_item.put(("clear", None))

    def _drain_shares(self) -> None:
        while True:
            try:
                self.share_queue.get_nowait()
            except queue.Empty:
                break

    def set_difficulty(self, difficulty: float) -> None:
        previous = self.difficulty
        self.difficulty = float(difficulty)
        self.share_target = difficulty_to_target_bytes(self.difficulty)
        for queue_item in self.job_queues:
            queue_item.put(("diff", difficulty))
        if self.difficulty > previous:
            # Drop any queued shares mined at the old (easier) target.
            self._drain_shares()

    def snapshot_hashes(self) -> int:
        total = 0
        for counter in self.hash_counters:
            with counter.get_lock():
                total += counter.value
        return total

    def set_job(self, job: MiningJob) -> None:
        if job.clean:
            self.clear_job()
            # Drop any shares computed for stale work.
            self._drain_shares()
        self.job_seq += 1
        self.current_job_id = job.job_id
        for queue_item in self.job_queues:
            queue_item.put(("job", (self.job_seq, job)))
