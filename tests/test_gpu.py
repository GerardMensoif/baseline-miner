"""Tests for GPU mining correctness.

These tests verify that the GPU OpenCL implementation produces
identical results to the CPU implementation.
"""

import hashlib
import struct
import unittest

# Try to import GPU modules
try:
    from baseline_miner.gpu import HAS_OPENCL, GPUHasher, scan_hashes_gpu, list_devices
except ImportError:
    HAS_OPENCL = False
    GPUHasher = None
    scan_hashes_gpu = None
    list_devices = lambda: []

from baseline_miner.hashing import scan_hashes as cpu_scan_hashes, sha256d


def _cpu_sha256d_header(header_prefix: bytes, nonce: int) -> bytes:
    """Compute SHA256d of a header using CPU."""
    header = header_prefix + nonce.to_bytes(4, "little")
    return hashlib.sha256(hashlib.sha256(header).digest()).digest()


@unittest.skipUnless(HAS_OPENCL, "OpenCL not available")
class GPUHashingTests(unittest.TestCase):
    """Test GPU hashing correctness against CPU reference."""

    def setUp(self) -> None:
        """Set up GPU hasher."""
        devices = list_devices()
        if not devices:
            self.skipTest("No OpenCL GPU devices available")
        self.hasher = GPUHasher(platform_idx=0, device_idx=0, global_size=1024)
        self.hasher.initialize()

    def tearDown(self) -> None:
        """Clean up GPU resources."""
        if hasattr(self, 'hasher') and self.hasher:
            self.hasher.release()

    def test_scan_matches_cpu_zeros(self) -> None:
        """Test GPU scan matches CPU with all-zero header prefix."""
        header_prefix = b"\x00" * 76
        target = b"\xff" * 32  # Easy target - accept all
        
        self.hasher.set_header_prefix(header_prefix)
        self.hasher.set_target(target)
        
        # Scan first 100 nonces
        gpu_results = self.hasher.scan_nonces(0, 100)
        
        # All 100 should match since target is maximum
        self.assertEqual(len(gpu_results), 100)
        
        # Verify each hash matches CPU
        for nonce, gpu_hash in gpu_results:
            cpu_hash = _cpu_sha256d_header(header_prefix, nonce)
            self.assertEqual(
                gpu_hash, cpu_hash,
                f"Hash mismatch at nonce {nonce}: GPU={gpu_hash.hex()}, CPU={cpu_hash.hex()}"
            )

    def test_scan_matches_cpu_random(self) -> None:
        """Test GPU scan matches CPU with random header prefix."""
        import os
        header_prefix = os.urandom(76)
        target = b"\xff" * 32  # Easy target
        
        self.hasher.set_header_prefix(header_prefix)
        self.hasher.set_target(target)
        
        # Scan first 256 nonces
        gpu_results = self.hasher.scan_nonces(0, 256)
        
        self.assertEqual(len(gpu_results), 256)
        
        for nonce, gpu_hash in gpu_results:
            cpu_hash = _cpu_sha256d_header(header_prefix, nonce)
            self.assertEqual(
                gpu_hash, cpu_hash,
                f"Hash mismatch at nonce {nonce}"
            )

    def test_scan_matches_cpu_native_scan(self) -> None:
        """Test GPU scan matches native CPU scan_hashes function."""
        header_prefix = b"\x12\x34" * 38  # Pattern that's easily verifiable
        target = b"\xff" * 32
        
        self.hasher.set_header_prefix(header_prefix)
        self.hasher.set_target(target)
        
        # Compare against native C implementation
        cpu_results = cpu_scan_hashes(header_prefix, 0, 100, target)
        gpu_results = self.hasher.scan_nonces(0, 100)
        
        self.assertEqual(len(cpu_results), len(gpu_results))
        
        cpu_dict = {n: h for n, h in cpu_results}
        gpu_dict = {n: h for n, h in gpu_results}
        
        for nonce in cpu_dict:
            self.assertIn(nonce, gpu_dict, f"GPU missing nonce {nonce}")
            self.assertEqual(
                cpu_dict[nonce], gpu_dict[nonce],
                f"Hash mismatch at nonce {nonce}"
            )

    def test_target_filtering(self) -> None:
        """Test that GPU correctly filters by target."""
        header_prefix = b"\x00" * 76
        # Set a restrictive target - only hashes starting with 0x00 pass
        target = b"\x00" + b"\xff" * 31
        
        self.hasher.set_header_prefix(header_prefix)
        self.hasher.set_target(target)
        
        # Scan many nonces to find matches
        gpu_results = self.hasher.scan_nonces(0, 10000)
        
        # Verify all returned hashes meet target
        for nonce, gpu_hash in gpu_results:
            self.assertLessEqual(
                gpu_hash, target,
                f"GPU returned hash that doesn't meet target at nonce {nonce}"
            )
            # Also verify against CPU
            cpu_hash = _cpu_sha256d_header(header_prefix, nonce)
            self.assertEqual(gpu_hash, cpu_hash)

    def test_nonce_range(self) -> None:
        """Test GPU handles various nonce ranges correctly."""
        header_prefix = b"\xab\xcd" * 38
        target = b"\xff" * 32
        
        self.hasher.set_header_prefix(header_prefix)
        self.hasher.set_target(target)
        
        # Test at different starting points
        test_ranges = [
            (0, 50),
            (1000, 50),
            (0x10000, 50),
            (0xFFFFFF00, 50),  # Near end of nonce space
        ]
        
        for start, count in test_ranges:
            gpu_results = self.hasher.scan_nonces(start, count)
            
            # Clamp count if near end
            expected_count = min(count, 0x100000000 - start)
            self.assertEqual(
                len(gpu_results), expected_count,
                f"Wrong result count for range ({start}, {count})"
            )
            
            for nonce, gpu_hash in gpu_results:
                self.assertGreaterEqual(nonce, start)
                self.assertLess(nonce, start + expected_count)
                cpu_hash = _cpu_sha256d_header(header_prefix, nonce)
                self.assertEqual(gpu_hash, cpu_hash)

    def test_header_prefix_update(self) -> None:
        """Test GPU handles header prefix updates correctly."""
        header1 = b"\x11" * 76
        header2 = b"\x22" * 76
        target = b"\xff" * 32
        
        self.hasher.set_target(target)
        
        # First header
        self.hasher.set_header_prefix(header1)
        results1 = self.hasher.scan_nonces(0, 10)
        
        # Second header
        self.hasher.set_header_prefix(header2)
        results2 = self.hasher.scan_nonces(0, 10)
        
        # Results should be different
        hashes1 = {h for _, h in results1}
        hashes2 = {h for _, h in results2}
        self.assertNotEqual(hashes1, hashes2)
        
        # Verify both against CPU
        for nonce, gpu_hash in results1:
            cpu_hash = _cpu_sha256d_header(header1, nonce)
            self.assertEqual(gpu_hash, cpu_hash)
        
        for nonce, gpu_hash in results2:
            cpu_hash = _cpu_sha256d_header(header2, nonce)
            self.assertEqual(gpu_hash, cpu_hash)


@unittest.skipUnless(HAS_OPENCL, "OpenCL not available")
class GPUScanHashesFunctionTests(unittest.TestCase):
    """Test the scan_hashes_gpu convenience function."""

    def test_scan_hashes_gpu_matches_cpu(self) -> None:
        """Test scan_hashes_gpu produces same results as CPU."""
        devices = list_devices()
        if not devices:
            self.skipTest("No OpenCL GPU devices available")
        
        header_prefix = b"\x55" * 76
        target = b"\xff" * 32
        
        cpu_results = cpu_scan_hashes(header_prefix, 0, 100, target)
        gpu_results = scan_hashes_gpu(header_prefix, 0, 100, target)
        
        self.assertEqual(len(cpu_results), len(gpu_results))
        
        cpu_dict = dict(cpu_results)
        gpu_dict = dict(gpu_results)
        
        for nonce in cpu_dict:
            self.assertEqual(cpu_dict[nonce], gpu_dict[nonce])


@unittest.skipUnless(HAS_OPENCL, "OpenCL not available")  
class GPUMidstateTests(unittest.TestCase):
    """Test GPU midstate computation correctness."""

    def setUp(self) -> None:
        devices = list_devices()
        if not devices:
            self.skipTest("No OpenCL GPU devices available")
        self.hasher = GPUHasher(platform_idx=0, device_idx=0, global_size=256)
        self.hasher.initialize()

    def tearDown(self) -> None:
        if hasattr(self, 'hasher') and self.hasher:
            self.hasher.release()

    def test_midstate_correctness(self) -> None:
        """Test that GPU computes correct midstate."""
        # Use known test vector
        header_prefix = bytes(range(76))  # 0x00, 0x01, 0x02, ...
        target = b"\xff" * 32
        
        self.hasher.set_header_prefix(header_prefix)
        self.hasher.set_target(target)
        
        # Scan and verify results match CPU
        results = self.hasher.scan_nonces(0, 50)
        
        for nonce, gpu_hash in results:
            cpu_hash = _cpu_sha256d_header(header_prefix, nonce)
            self.assertEqual(
                gpu_hash, cpu_hash,
                f"Midstate computation error detected at nonce {nonce}"
            )


if __name__ == "__main__":
    unittest.main()
