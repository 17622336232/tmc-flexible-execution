"""Immutable query bindings and a bounded background inference worker."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from typing import Callable, Dict, Optional, Tuple

from .evidence import Frame
from .task_ir import QuerySpec


@dataclass(frozen=True)
class QueryBinding:
    id: str
    object_id: str
    frame_id: str
    frame_version: int
    input_hash: str
    submitted_at: float
    outcomes: Tuple[str, ...]

    @classmethod
    def bind(cls, spec: QuerySpec, frame: Frame, prompt: str, at: float):
        if spec.object_id != frame.object_id or frame.captured_at > at:
            raise ValueError("Query and observation do not identify the same measured object")
        digest = sha256((frame.digest + "\n" + prompt).encode("utf8")).hexdigest()
        return cls(spec.id, frame.object_id, frame.id, frame.version, digest, at, spec.outcomes)


@dataclass(frozen=True)
class QueryReply:
    binding: QueryBinding
    outcome: str


class QueryBook:
    def __init__(self):
        self.pending: Dict[str, QueryBinding] = {}
        self.results: Dict[str, str] = {}

    def register(self, binding: QueryBinding) -> None:
        old = self.pending.get(binding.id)
        if binding.id in self.results:
            raise ValueError("A completed query ID cannot be reused")
        if old and (binding.object_id != old.object_id or binding.frame_version <= old.frame_version):
            raise ValueError("A replacement must use a newer frame of the same object")
        self.pending[binding.id] = binding

    def accept(self, reply: QueryReply, at: float) -> bool:
        current = self.pending.get(reply.binding.id)
        if (current is None or current != reply.binding or at < current.submitted_at
                or reply.outcome not in current.outcomes):
            return False
        del self.pending[current.id]
        self.results[current.id] = reply.outcome
        return True


class AsyncQueryWorker:
    """A blocking vision call uses one independent worker, outside motion dispatch."""

    def __init__(self, infer: Callable[[bytes, str], str]):
        self.infer = infer
        self.pool = ThreadPoolExecutor(max_workers=1)

    def submit(self, spec: QuerySpec, frame: Frame, prompt: str, at: float,
               book: QueryBook) -> asyncio.Task:
        binding = QueryBinding.bind(spec, frame, prompt, at)
        loop = asyncio.get_running_loop()
        book.register(binding)

        async def receive():
            response = await loop.run_in_executor(self.pool, self.infer, frame.payload, prompt)
            return QueryReply(binding, response)

        return loop.create_task(receive())

    def close(self):
        self.pool.shutdown(wait=True)
