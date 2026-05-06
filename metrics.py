import networkx as nx
from typing import Dict, List

from models import Agent, Task, RouteResult
from utils import path_latency


def evaluate_route(
    G: nx.Graph,
    agents: Dict[int, Agent],
    task: Task,
    method_name: str,
    path: List[int],
    extra_overhead: float = 0.0,
    explanation: str = "",
) -> RouteResult:

    if not path:
        path = [task.source]

    required = task.required_capability
    final_agent = path[-1]

    success = int(required in agents[final_agent].capabilities)
    latency = path_latency(G, agents, path)
    hops = max(0, len(path) - 1)

    # Penalty if the route did not reach a capable agent.
    if success == 0:
        latency += 50.0

    return RouteResult(
        method=method_name,
        task_id=task.task_id,
        path=path,
        success=success,
        latency=latency,
        hops=hops,
        communication_overhead=hops + extra_overhead,
        explanation_available=int(bool(explanation)),
        explanation=explanation,
    )