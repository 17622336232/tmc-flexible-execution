"""Bind accepted material to the observation that actually produced it."""

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from typing import Optional

from .task_ir import Requirement


@dataclass(frozen=True)
class Frame:
    id: str
    object_id: str
    version: int
    captured_at: float
    payload: bytes
    provenance: str

    def __post_init__(self):
        if not self.id or not self.object_id or not self.provenance:
            raise ValueError("Frame identity and provenance are required")
        if self.version < 0 or not isfinite(self.captured_at) or self.captured_at < 0:
            raise ValueError("Invalid frame version or capture time")
        object.__setattr__(self, "payload", bytes(self.payload))
        if not self.payload:
            raise ValueError("The frozen observation is empty")

    @property
    def digest(self) -> str:
        return sha256(self.payload).hexdigest()


@dataclass(frozen=True)
class Observation:
    frame: Frame
    field: str
    value: str
    readable: bool
    valid_until: Optional[float] = None


@dataclass(frozen=True)
class Evidence:
    requirement_id: str
    object_id: str
    field: str
    frame_id: str
    frame_hash: str
    captured_at: float
    value: str
    provenance: str
    valid_until: Optional[float]

    def usable(self, requirement: Requirement, at: float) -> bool:
        return (
            self.requirement_id == requirement.id
            and self.object_id == requirement.object_id
            and self.field == requirement.field
            and self.captured_at <= at
            and (requirement.due is None or self.captured_at <= requirement.due)
            and (self.valid_until is None or at < self.valid_until)
        )


def accept(requirement: Requirement, observation: Observation, at: float) -> Optional[Evidence]:
    frame = observation.frame
    if not observation.readable or not observation.value:
        return None
    material = Evidence(
        requirement.id, frame.object_id, observation.field, frame.id, frame.digest,
        frame.captured_at, observation.value, frame.provenance, observation.valid_until,
    )
    return material if material.usable(requirement, at) else None
