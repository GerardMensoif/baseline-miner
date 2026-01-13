"""GPU mining support using OpenCL.

This module provides GPU-accelerated SHA256d mining that exactly replicates
the CPU algorithm for Baseline blockchain compatibility.
"""

from __future__ import annotations

import importlib.resources
import struct
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    import numpy as np
    import pyopencl as cl

try:
    import numpy as np
    import pyopencl as cl
    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False
    np = None  # type: ignore
    cl = None  # type: ignore


@dataclass
class GPUDevice:
    """Represents an OpenCL-capable GPU device."""
    platform_idx: int
    device_idx: int
    name: str
    platform_name: str
    max_work_group_size: int
    max_compute_units: int
    global_mem_size: int

    def __str__(self) -> str:
        mem_gb = self.global_mem_size / (1024**3)
        return f"[{self.platform_idx}:{self.device_idx}] {self.name} ({self.platform_name}) - {self.max_compute_units} CUs, {mem_gb:.1f} GB"


def list_devices() -> list[GPUDevice]:
    """List all available OpenCL GPU devices."""
    if not HAS_OPENCL:
        return []
    
    devices = []
    try:
        platforms = cl.get_platforms()
        for p_idx, platform in enumerate(platforms):
            try:
                gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
                for d_idx, device in enumerate(gpu_devices):
                    devices.append(GPUDevice(
                        platform_idx=p_idx,
                        device_idx=d_idx,
                        name=device.name.strip(),
                        platform_name=platform.name.strip(),
                        max_work_group_size=device.max_work_group_size,
                        max_compute_units=device.max_compute_units,
                        global_mem_size=device.global_mem_size,
                    ))
            except cl.RuntimeError:
                continue
    except Exception:
        pass
    return devices


def _load_kernel_source() -> str:
    """Load the OpenCL kernel source code."""
    try:
        files = importlib.resources.files("baseline_miner.opencl")
        return (files / "sha256d.cl").read_text()
    except Exception:
        import os
        kernel_path = os.path.join(os.path.dirname(__file__), "opencl", "sha256d.cl")
        with open(kernel_path) as f:
            return f.read()


def _load_be32(data: bytes, offset: int) -> int:
    """Load a 32-bit big-endian value from bytes."""
    return struct.unpack(">I", data[offset:offset+4])[0]


def _bswap32(x: int) -> int:
    """Byte-swap a 32-bit integer."""
    return ((x >> 24) & 0xFF) | ((x >> 8) & 0xFF00) | ((x << 8) & 0xFF0000) | ((x << 24) & 0xFF000000)


class GPUHasher:
    """OpenCL-based GPU hasher for SHA256d mining.
    
    This class manages the OpenCL context, kernel compilation, and provides
    methods for scanning nonces on the GPU.
    """
    
    # Default work sizes - optimized for modern GPUs
    DEFAULT_GLOBAL_SIZE = 1 << 24  # 16M work items per batch (matches batch_size)
    DEFAULT_LOCAL_SIZE = 256
    MAX_RESULTS = 1024
    
    def __init__(
        self,
        platform_idx: int = 0,
        device_idx: int = 0,
        global_size: Optional[int] = None,
        local_size: Optional[int] = None,
    ):
        """Initialize the GPU hasher.
        
        Args:
            platform_idx: OpenCL platform index
            device_idx: Device index within the platform
            global_size: Number of work items per batch (default: 1M)
            local_size: Work group size (default: 256)
        """
        if not HAS_OPENCL:
            raise RuntimeError("PyOpenCL is not installed. Install with: pip install pyopencl")
        
        self.platform_idx = platform_idx
        self.device_idx = device_idx
        self.global_size = global_size or self.DEFAULT_GLOBAL_SIZE
        self.local_size = local_size or self.DEFAULT_LOCAL_SIZE
        
        # Ensure global_size is a multiple of local_size
        if self.global_size % self.local_size != 0:
            self.global_size = ((self.global_size + self.local_size - 1) // self.local_size) * self.local_size
        
        self._ctx: Optional[cl.Context] = None
        self._queue: Optional[cl.CommandQueue] = None
        self._program: Optional[cl.Program] = None
        self._kernel: Optional[cl.Kernel] = None  # Cached kernel to avoid repeated retrieval
        self._initialized = False
        
        # Buffers
        self._midstate_buf: Optional[cl.Buffer] = None
        self._block2_buf: Optional[cl.Buffer] = None
        self._target_buf: Optional[cl.Buffer] = None
        self._results_buf: Optional[cl.Buffer] = None
        self._result_hashes_buf: Optional[cl.Buffer] = None
        self._result_count_buf: Optional[cl.Buffer] = None
        
        # Current job state (computed on CPU, uploaded to GPU)
        self._current_midstate: Optional[np.ndarray] = None
        self._current_block2: Optional[np.ndarray] = None
        
    def initialize(self) -> None:
        """Initialize OpenCL context and compile kernels."""
        if self._initialized:
            return
            
        platforms = cl.get_platforms()
        if self.platform_idx >= len(platforms):
            raise ValueError(f"Platform index {self.platform_idx} out of range (have {len(platforms)} platforms)")
        
        platform = platforms[self.platform_idx]
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        if self.device_idx >= len(devices):
            raise ValueError(f"Device index {self.device_idx} out of range (have {len(devices)} GPU devices)")
        
        device = devices[self.device_idx]
        
        # Adjust local size if needed
        max_wg_size = device.max_work_group_size
        if self.local_size > max_wg_size:
            self.local_size = max_wg_size
            if self.global_size % self.local_size != 0:
                self.global_size = ((self.global_size + self.local_size - 1) // self.local_size) * self.local_size
        
        self._ctx = cl.Context([device])
        self._queue = cl.CommandQueue(self._ctx)
        
        # Compile kernel (suppress pyopencl cache warnings)
        kernel_source = _load_kernel_source()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._program = cl.Program(self._ctx, kernel_source).build()
        
        # Cache kernel instance to avoid repeated retrieval warning
        self._kernel = cl.Kernel(self._program, "scan_nonces")
        
        # Allocate buffers
        mf = cl.mem_flags
        
        # Input buffers (read-only)
        self._midstate_buf = cl.Buffer(self._ctx, mf.READ_ONLY, 8 * 4)  # 8 uint32
        self._block2_buf = cl.Buffer(self._ctx, mf.READ_ONLY, 16 * 4)  # 16 uint32
        self._target_buf = cl.Buffer(self._ctx, mf.READ_ONLY, 8 * 4)  # 8 uint32
        
        # Output buffers
        self._results_buf = cl.Buffer(self._ctx, mf.WRITE_ONLY, self.MAX_RESULTS * 4)  # nonces
        self._result_hashes_buf = cl.Buffer(self._ctx, mf.WRITE_ONLY, self.MAX_RESULTS * 8 * 4)  # hashes
        self._result_count_buf = cl.Buffer(self._ctx, mf.READ_WRITE, 4)  # atomic counter
        
        self._initialized = True
    
    def release(self) -> None:
        """Release OpenCL resources."""
        self._midstate_buf = None
        self._block2_buf = None
        self._target_buf = None
        self._results_buf = None
        self._result_hashes_buf = None
        self._result_count_buf = None
        self._queue = None
        self._program = None
        self._ctx = None
        self._initialized = False
    
    def set_header_prefix(self, header_prefix: bytes) -> None:
        """Set the header prefix (first 76 bytes) and compute midstate.
        
        This should be called when a new job is received. The midstate
        (SHA256 state after processing first 64 bytes) is computed and
        cached for reuse across nonce scans.
        
        Args:
            header_prefix: 76-byte header prefix (everything except nonce)
        """
        if len(header_prefix) != 76:
            raise ValueError(f"header_prefix must be 76 bytes, got {len(header_prefix)}")
        
        if not self._initialized:
            self.initialize()
        
        # Compute midstate on CPU (matches C code exactly)
        # Load first 64 bytes as 16 big-endian words
        block1_words = np.array([
            _load_be32(header_prefix, i * 4) for i in range(16)
        ], dtype=np.uint32)
        
        # SHA256 initial state
        state = np.array([
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        ], dtype=np.uint32)
        
        # Compress first block
        self._sha256_compress_numpy(state, block1_words)
        self._current_midstate = state.copy()
        
        # Prepare block2 base
        # bytes 64-75 as big-endian words, then padding
        block2 = np.zeros(16, dtype=np.uint32)
        block2[0] = _load_be32(header_prefix, 64)
        block2[1] = _load_be32(header_prefix, 68)
        block2[2] = _load_be32(header_prefix, 72)
        block2[3] = 0  # nonce placeholder
        block2[4] = 0x80000000  # padding start
        # [5-14] are zeros
        block2[15] = 0x00000280  # length: 80 * 8 = 640 bits
        self._current_block2 = block2
        
        # Upload to GPU
        cl.enqueue_copy(self._queue, self._midstate_buf, self._current_midstate)
        cl.enqueue_copy(self._queue, self._block2_buf, self._current_block2)
    
    def set_target(self, target: bytes) -> None:
        """Set the target threshold.
        
        Args:
            target: 32-byte big-endian target
        """
        if len(target) != 32:
            raise ValueError(f"target must be 32 bytes, got {len(target)}")
        
        if not self._initialized:
            self.initialize()
        
        # Load as big-endian words
        target_words = np.array([
            _load_be32(target, i * 4) for i in range(8)
        ], dtype=np.uint32)
        
        cl.enqueue_copy(self._queue, self._target_buf, target_words)
    
    def scan_nonces(
        self,
        start_nonce: int,
        count: Optional[int] = None,
    ) -> list[tuple[int, bytes]]:
        """Scan a range of nonces for hashes meeting the target.
        
        Args:
            start_nonce: Starting nonce value
            count: Number of nonces to scan (default: global_size)
        
        Returns:
            List of (nonce, hash_bytes) tuples for matching nonces.
            Hash bytes are in big-endian format.
        """
        if not self._initialized:
            raise RuntimeError("GPUHasher not initialized")
        
        if self._current_midstate is None:
            raise RuntimeError("No header set - call set_header_prefix first")
        
        if count is None:
            count = self.global_size
        
        # Clamp count to not exceed nonce space
        if start_nonce >= 0x100000000:
            return []
        max_count = 0x100000000 - start_nonce
        if count > max_count:
            count = max_count
        
        # Process in batches
        results = []
        remaining = count
        current_nonce = start_nonce
        
        while remaining > 0:
            batch_size = min(remaining, self.global_size)
            # Round up to multiple of local_size
            actual_global = ((batch_size + self.local_size - 1) // self.local_size) * self.local_size
            
            # Reset result counter
            zero = np.array([0], dtype=np.uint32)
            cl.enqueue_copy(self._queue, self._result_count_buf, zero)
            
            # Run kernel using cached kernel instance
            self._kernel.set_args(
                self._midstate_buf,
                self._block2_buf,
                self._target_buf,
                np.uint32(current_nonce),
                self._results_buf,
                self._result_hashes_buf,
                self._result_count_buf,
                np.uint32(self.MAX_RESULTS),
            )
            cl.enqueue_nd_range_kernel(
                self._queue,
                self._kernel,
                (actual_global,),
                (self.local_size,),
            )
            
            # Read results
            result_count = np.zeros(1, dtype=np.uint32)
            cl.enqueue_copy(self._queue, result_count, self._result_count_buf)
            num_results = min(int(result_count[0]), self.MAX_RESULTS)
            
            if num_results > 0:
                nonces = np.zeros(num_results, dtype=np.uint32)
                hashes = np.zeros(num_results * 8, dtype=np.uint32)
                cl.enqueue_copy(self._queue, nonces, self._results_buf)
                cl.enqueue_copy(self._queue, hashes, self._result_hashes_buf)
                
                for i in range(num_results):
                    nonce = int(nonces[i])
                    # Filter out nonces beyond our actual batch
                    if nonce >= current_nonce + batch_size:
                        continue
                    # Convert hash words to bytes (big-endian)
                    hash_words = hashes[i*8:(i+1)*8]
                    hash_bytes = b"".join(struct.pack(">I", int(w)) for w in hash_words)
                    results.append((nonce, hash_bytes))
            
            current_nonce += batch_size
            remaining -= batch_size
        
        return results
    
    def _sha256_compress_numpy(self, state: np.ndarray, block: np.ndarray) -> None:
        """SHA256 compression function using numpy (for midstate computation).
        
        Modifies state in-place.
        """
        # Suppress overflow warnings - overflow is expected in SHA256 (modular arithmetic)
        with np.errstate(over='ignore'):
            self._sha256_compress_numpy_impl(state, block)
    
    def _sha256_compress_numpy_impl(self, state: np.ndarray, block: np.ndarray) -> None:
        """Internal SHA256 compression implementation."""
        K = np.array([
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
            0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
            0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
            0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
        ], dtype=np.uint32)
        
        def rotr(x, n):
            x = np.uint32(x)
            return np.uint32((x >> n) | (x << (32 - n)))
        
        def ch(x, y, z):
            return np.uint32((x & y) ^ (~x & z))
        
        def maj(x, y, z):
            return np.uint32((x & y) ^ (x & z) ^ (y & z))
        
        def bsig0(x):
            return np.uint32(rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22))
        
        def bsig1(x):
            return np.uint32(rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25))
        
        def ssig0(x):
            return np.uint32(rotr(x, 7) ^ rotr(x, 18) ^ (np.uint32(x) >> 3))
        
        def ssig1(x):
            return np.uint32(rotr(x, 17) ^ rotr(x, 19) ^ (np.uint32(x) >> 10))
        
        w = block.copy()
        
        a, b, c, d, e, f, g, h = [np.uint32(state[i]) for i in range(8)]
        
        for i in range(16):
            t1 = np.uint32(h + bsig1(e) + ch(e, f, g) + K[i] + w[i])
            t2 = np.uint32(bsig0(a) + maj(a, b, c))
            h = g
            g = f
            f = e
            e = np.uint32(d + t1)
            d = c
            c = b
            b = a
            a = np.uint32(t1 + t2)
        
        for i in range(16, 64):
            j = i & 15
            w[j] = np.uint32(w[j] + ssig0(w[(j + 1) & 15]) + w[(j + 9) & 15] + ssig1(w[(j + 14) & 15]))
            t1 = np.uint32(h + bsig1(e) + ch(e, f, g) + K[i] + w[j])
            t2 = np.uint32(bsig0(a) + maj(a, b, c))
            h = g
            g = f
            f = e
            e = np.uint32(d + t1)
            d = c
            c = b
            b = a
            a = np.uint32(t1 + t2)
        
        state[0] = np.uint32(state[0] + a)
        state[1] = np.uint32(state[1] + b)
        state[2] = np.uint32(state[2] + c)
        state[3] = np.uint32(state[3] + d)
        state[4] = np.uint32(state[4] + e)
        state[5] = np.uint32(state[5] + f)
        state[6] = np.uint32(state[6] + g)
        state[7] = np.uint32(state[7] + h)


def scan_hashes_gpu(
    header_prefix: bytes,
    start_nonce: int,
    count: int,
    target: bytes,
    hasher: Optional[GPUHasher] = None,
    platform_idx: int = 0,
    device_idx: int = 0,
) -> list[tuple[int, bytes]]:
    """GPU version of scan_hashes matching the CPU interface.
    
    Args:
        header_prefix: 76-byte header prefix
        start_nonce: Starting nonce value  
        count: Number of nonces to scan
        target: 32-byte big-endian target
        hasher: Optional pre-initialized GPUHasher instance
        platform_idx: OpenCL platform index (if hasher not provided)
        device_idx: Device index (if hasher not provided)
    
    Returns:
        List of (nonce, hash_bytes) tuples for matching nonces
    """
    if hasher is None:
        hasher = GPUHasher(platform_idx=platform_idx, device_idx=device_idx)
        hasher.initialize()
    
    hasher.set_header_prefix(header_prefix)
    hasher.set_target(target)
    return hasher.scan_nonces(start_nonce, count)
