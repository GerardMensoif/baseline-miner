import asyncio
import contextlib
import json
import logging
from typing import Any, Callable

from .job import MiningJob, Share

MAX_MESSAGE_BYTES = 4096


class StratumRPCError(RuntimeError):
    def __init__(self, code: int, message: str, data: object = None):
        super().__init__(f"[{code}, {message!r}]")
        self.code = int(code)
        self.message = str(message)
        self.data = data


class StratumClient:
    def __init__(self, host: str, port: int, *, user_agent: str = "baseline-miner/0.1"):
        self.host = host
        self.port = port
        self.user_agent = user_agent
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 1
        self._reader_task: asyncio.Task | None = None
        self._closed = asyncio.Event()
        self.connected = False
        self.difficulty: float | None = None
        self.extranonce1: bytes | None = None
        self.extranonce2_size: int | None = None
        self.on_difficulty: Callable[[float], None] | None = None
        self.on_job: Callable[[MiningJob], None] | None = None
        self.log = logging.getLogger("baseline_miner.stratum")

    async def connect(self) -> None:
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        self.connected = True
        self._closed.clear()
        self._reader_task = asyncio.create_task(self._read_loop(), name="stratum-read")

    async def close(self) -> None:
        if not self.connected:
            return
        self.connected = False
        if self.writer:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except Exception:
                pass
        if self._reader_task:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
        self._closed.set()

    async def wait_closed(self) -> None:
        await self._closed.wait()

    async def subscribe(self) -> None:
        result = await self.request("mining.subscribe", [self.user_agent])
        if not isinstance(result, list) or len(result) < 3:
            raise RuntimeError("Unexpected subscribe response")
        self.extranonce1 = bytes.fromhex(str(result[1]))
        self.extranonce2_size = int(result[2])

    async def authorize(self, worker_name: str, password: str = "") -> None:
        result = await self.request("mining.authorize", [worker_name, password])
        if result is not True:
            raise RuntimeError("Authorization failed")

    async def submit_share(self, share: Share, worker_name: str, extranonce2_size: int) -> bool:
        extranonce2_hex = f"{share.extranonce2:0{extranonce2_size * 2}x}"
        ntime_hex = f"{share.ntime:08x}"
        nonce_hex = f"{share.nonce:08x}"
        try:
            result = await self.request(
                "mining.submit",
                [worker_name, share.job_id, extranonce2_hex, ntime_hex, nonce_hex],
            )
            return bool(result)
        except StratumRPCError as exc:
            # These are expected during job switches or when resubmitting a share.
            if exc.code in {29, 32}:
                self.log.debug(
                    "Share rejected for job %s (extranonce2=%s ntime=%s nonce=%s): %s",
                    share.job_id,
                    extranonce2_hex,
                    ntime_hex,
                    nonce_hex,
                    exc,
                )
                return False
            self.log.warning(
                "Share rejected for job %s (extranonce2=%s ntime=%s nonce=%s): %s",
                share.job_id,
                extranonce2_hex,
                ntime_hex,
                nonce_hex,
                exc,
            )
            return False
        except Exception as exc:
            self.log.warning(
                "Share rejected for job %s (extranonce2=%s ntime=%s nonce=%s): %s",
                share.job_id,
                extranonce2_hex,
                ntime_hex,
                nonce_hex,
                exc,
            )
            return False

    async def request(self, method: str, params: list[Any]) -> Any:
        msg_id = self._next_id
        self._next_id += 1
        payload = {"id": msg_id, "method": method, "params": params}
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[msg_id] = future
        try:
            await self._send(payload)
            return await future
        except Exception:
            self._pending.pop(msg_id, None)
            raise

    async def _send(self, payload: dict[str, Any]) -> None:
        if not self.writer:
            raise RuntimeError("Not connected")
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
        self.writer.write(data)
        await self.writer.drain()

    async def _read_loop(self) -> None:
        assert self.reader is not None
        try:
            while True:
                line = await self.reader.readline()
                if not line:
                    break
                if len(line) > MAX_MESSAGE_BYTES:
                    self.log.warning("Dropping oversized message")
                    continue
                try:
                    message = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    self.log.debug("Invalid JSON from server")
                    continue
                msg_id = message.get("id")
                if msg_id is not None and msg_id in self._pending:
                    future = self._pending.pop(msg_id)
                    error = message.get("error")
                    if error:
                        if isinstance(error, (list, tuple)) and len(error) >= 2:
                            code = error[0]
                            message_text = error[1]
                        else:
                            code = -1
                            message_text = error
                        future.set_exception(StratumRPCError(int(code), str(message_text), error))
                    else:
                        future.set_result(message.get("result"))
                    continue
                method = message.get("method")
                params = message.get("params") or []
                if method == "mining.set_difficulty" and params:
                    try:
                        difficulty = float(params[0])
                    except (TypeError, ValueError):
                        self.log.debug("Invalid difficulty from server")
                        continue
                    self.difficulty = difficulty
                    if self.on_difficulty:
                        self.on_difficulty(difficulty)
                    self.log.info("Set mining difficulty to %.4f", difficulty)
                elif method == "mining.notify":
                    job = self._parse_notify(params)
                    if job and self.on_job:
                        self.on_job(job)
        except Exception as exc:
            self.log.debug("Stratum read loop ended: %s", exc)
        finally:
            self.connected = False
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(RuntimeError("Stratum connection closed"))
            self._pending.clear()
            self._closed.set()

    def _parse_notify(self, params: list[Any]) -> MiningJob | None:
        if len(params) < 9:
            return None
        if self.extranonce1 is None:
            return None
        job_id = str(params[0])
        prev_hash = str(params[1])
        coinb1 = bytes.fromhex(str(params[2]))
        coinb2 = bytes.fromhex(str(params[3]))
        branches_hex = params[4] if isinstance(params[4], list) else []
        merkle_branches = [bytes.fromhex(str(item))[::-1] for item in branches_hex]
        version = int(str(params[5]), 16)
        bits = int(str(params[6]), 16)
        ntime = int(str(params[7]), 16)
        clean = bool(params[8])
        prev_hash_le = bytes.fromhex(prev_hash)[::-1]
        return MiningJob(
            job_id=job_id,
            prev_hash_le=prev_hash_le,
            coinb1=coinb1,
            coinb2=coinb2,
            merkle_branches_le=merkle_branches,
            version=version,
            bits=bits,
            ntime=ntime,
            extranonce1=self.extranonce1,
            clean=clean,
        )
