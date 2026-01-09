import unittest

from baseline_miner.job import Share
from baseline_miner.stratum import StratumClient


class StratumParseTests(unittest.TestCase):
    def test_parse_notify(self) -> None:
        client = StratumClient("127.0.0.1", 0)
        client.extranonce1 = bytes.fromhex("aabbccdd")
        branch = bytes(range(32))
        params = [
            "1",
            "00" * 32,
            "010203",
            "040506",
            [branch[::-1].hex()],
            "00000001",
            "207fffff",
            "5f5e1000",
            True,
        ]
        job = client._parse_notify(params)
        self.assertIsNotNone(job)
        assert job is not None
        self.assertEqual(job.job_id, "1")
        self.assertEqual(job.prev_hash_le, bytes.fromhex("00" * 32)[::-1])
        self.assertEqual(job.extranonce1, client.extranonce1)
        self.assertEqual(job.version, 1)
        self.assertEqual(job.bits, 0x207FFFFF)
        self.assertEqual(job.merkle_branches_le[0], branch)
        self.assertTrue(job.clean)


class StratumSubmitTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_share_formats_baseline_node_params(self) -> None:
        client = StratumClient("127.0.0.1", 0)
        captured: dict[str, object] = {}

        async def fake_request(method: str, params: list[object]):
            captured["method"] = method
            captured["params"] = params
            return True

        client.request = fake_request  # type: ignore[method-assign]

        share = Share(
            job_id="job123",
            extranonce2=0x1,
            ntime=0x5F5E1000,
            nonce=0x00ABCDEF,
            is_block=False,
            hash_hex="",
        )
        ok = await client.submit_share(share, worker_name="worker.rig1", extranonce2_size=4)

        self.assertTrue(ok)
        self.assertEqual(captured["method"], "mining.submit")
        self.assertEqual(
            captured["params"],
            ["worker.rig1", "job123", "00000001", "5f5e1000", "00abcdef"],
        )


if __name__ == "__main__":
    unittest.main()
