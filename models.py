from dataclasses import dataclass
from typing import Set, List


@dataclass
class Agent:
    agent_id: int
    capabilities: Set[str]
    reliability: float
    load: float
    resource: float
    protocol: str
    failed: bool = False


@dataclass
class Task:
    task_id: int
    source: int
    required_capability: str
    priority: float
    deadline: float


@dataclass
class RouteDecision:
    path: List[int]
    explanation: str = ""
    extra_overhead: float = 0.0


@dataclass
class RouteResult:
    method: str
    task_id: int
    path: List[int]
    success: int
    latency: float
    hops: int
    communication_overhead: float
    explanation_available: int
    explanation: str