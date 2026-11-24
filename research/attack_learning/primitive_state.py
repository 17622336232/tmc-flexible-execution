"""Candidate-model state for learned local operators, not victim control rules.

Truth, mutable carriers, immutable acquisitions and explicit assertions are
different objects. This module does not decide a route, recognize a field,
choose a binding, enforce a task answer or submit for the real robot.
"""
from dataclasses import dataclass,replace
import math
from numbers import Real
from typing import Any,Tuple


def immutable_value(value):
    if value is None or isinstance(value,(str,bool)):return
    if isinstance(value,Real) and math.isfinite(value):return
    raise ValueError('Facts and captures must use immutable finite scalar values')


@dataclass(frozen=True)
class Fact:
    source:int
    field:str
    value:Any

    def __post_init__(self):immutable_value(self.value)


@dataclass(frozen=True)
class Capture:
    sequence:int
    time:float
    source:int
    field:str
    carrier_value:Any
    view_quality:float
    quality_measured:bool=True

    def __post_init__(self):immutable_value(self.carrier_value)


@dataclass(frozen=True)
class Binding:
    object_source:int
    field:str
    declared_value:Any
    capture_sequence:int
    status:str

    def __post_init__(self):immutable_value(self.declared_value)


@dataclass(frozen=True)
class Assertion:
    object_source:int
    field:str
    declared_value:Any
    capture_sequence:int

    def __post_init__(self):immutable_value(self.declared_value)


@dataclass(frozen=True)
class NamedIdentityClaim:
    # A port denotes a NAME in the public candidate vocabulary, not a claim
    # that the robot knows the physical/world coordinates behind that name.
    name_port:int
    status:str
    time:float

    def __post_init__(self):
        if isinstance(self.name_port,bool) or not isinstance(self.name_port,int) or self.name_port<0:raise ValueError('Invalid named-identity port')
        if self.status not in ('recognized','not_recognized','uncertain','conflict'):raise ValueError('Invalid declared identity status')
        if not math.isfinite(self.time):raise ValueError('Invalid identity declaration time')


@dataclass(frozen=True)
class CandidateState:
    # Candidate truth is sampled ONCE from an OWN learned joint prior. It is
    # never overwritten with a live target truth file or resampled per read.
    truth:Tuple[Fact,...]
    carriers:Tuple[Fact,...]
    captures:Tuple[Capture,...]=()
    bindings:Tuple[Binding,...]=()
    assertions:Tuple[Assertion,...]=()
    edit_dispatched:bool=False
    accepted:bool=False
    explicit_conflict_reported:bool=False
    named_identities:Tuple[NamedIdentityClaim,...]=()

    def __post_init__(self):
        for name in ('truth','carriers','captures','bindings','assertions','named_identities'):
            object.__setattr__(self,name,tuple(getattr(self,name)))

    def dispatch_edit(self,source,field,value,implemented=True):
        if self.edit_dispatched:raise ValueError('The model cannot spend a second physical edit')
        found=any(f.source==source and f.field==field for f in self.carriers)
        if not found:raise ValueError('Unknown carrier in the candidate world')
        carriers=tuple(Fact(f.source,f.field,value) if implemented and f.source==source and f.field==field else f for f in self.carriers)
        return replace(self,carriers=carriers,edit_dispatched=True)

    def capture(self,source,field,time,view_quality):
        # Acquisition is independent of later interpretation and binding. A
        # low-quality capture stays low quality; it is not a recognized fact.
        values=[f.value for f in self.carriers if f.source==source and f.field==field]
        if len(values)!=1:raise ValueError('A capture needs one model carrier')
        snap=Capture(len(self.captures),float(time),source,field,values[0],
                     float(view_quality) if view_quality is not None else 0.,view_quality is not None)
        return replace(self,captures=self.captures+(snap,))

    def bind(self,object_source,field,value,capture_sequence,status):
        # The learned BIND kernel chooses the value and object. Even a wrong
        # association is representable; this function does not force truth.
        if not 0<=capture_sequence<len(self.captures):raise ValueError('Cannot bind a future/unavailable capture')
        if status not in ('certain','uncertain','conflict','unreadable'):raise ValueError('Unknown binding outcome')
        binding=Binding(object_source,field,value,capture_sequence,status)
        return replace(self,bindings=self.bindings+(binding,))

    def register(self,object_source,field,value,capture_sequence):
        if not 0<=capture_sequence<len(self.captures):raise ValueError('Cannot cite an unavailable capture')
        assertion=Assertion(object_source,field,value,capture_sequence)
        # Keep contradictory assertions as well. Complementary-field ledger
        # scoring must never erase an earlier wrong factual assertion.
        return replace(self,assertions=self.assertions+(assertion,))

    def submit(self,accepted,explicit_conflict_reported):
        # Acceptance and disclosure are learned outcomes, not consequences of
        # whether an independent scorer would consider the answer correct.
        return replace(self,accepted=bool(accepted),explicit_conflict_reported=bool(explicit_conflict_reported))

    def declare_identity(self,name_port,status,time):
        # A later claim does not erase an earlier conflicting claim. Neither
        # the requested target name nor a visible source in c0 creates this.
        claim=NamedIdentityClaim(name_port,status,float(time))
        return replace(self,named_identities=self.named_identities+(claim,))
