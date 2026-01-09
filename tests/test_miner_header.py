import unittest

from baseline_miner.hashing import sha256d
from baseline_miner.job import MiningJob
from baseline_miner.miner import _build_header_prefix, _merkle_root


class MinerHeaderTests(unittest.TestCase):
    def test_header_prefix_serializes_merkle_root_big_endian(self) -> None:
        extranonce1 = bytes.fromhex("aabbccdd")
        extranonce2 = 0x1
        extranonce2_size = 4
        coinb1 = bytes.fromhex("010203")
        coinb2 = bytes.fromhex("040506")
        branch = bytes(range(32))
        ntime = 0x5F5E1000

        job = MiningJob(
            job_id="1",
            prev_hash_le=(bytes.fromhex("11" * 32)[::-1]),
            coinb1=coinb1,
            coinb2=coinb2,
            merkle_branches_le=[branch],
            version=1,
            bits=0x207FFFFF,
            ntime=ntime,
            extranonce1=extranonce1,
            clean=False,
        )

        header_prefix = _build_header_prefix(
            job,
            extranonce2=extranonce2,
            extranonce2_size=extranonce2_size,
            ntime=ntime,
        )

        extranonce2_bytes = extranonce2.to_bytes(extranonce2_size, "big")
        coinbase = coinb1 + extranonce1 + extranonce2_bytes + coinb2
        merkle_root_be = _merkle_root(sha256d(coinbase), [branch])

        self.assertEqual(len(header_prefix), 76)
        self.assertEqual(header_prefix[36:68], merkle_root_be)


if __name__ == "__main__":
    unittest.main()
