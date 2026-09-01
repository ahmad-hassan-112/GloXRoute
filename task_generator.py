"""
task_generator.py — generate routing tasks with a controllable
capability distribution so the routing problem is non-trivial.
"""
import random
from typing import List

from config import NUM_TASKS, NUM_AGENTS, CAPABILITIES
from models import Task


def generate_tasks(
    num_tasks: int = NUM_TASKS,
    seed: int = 42,
    skew: bool = True,
    num_agents: int = None,
) -> List[Task]:
    """
    Generate tasks.

    If skew=True (default) the required capability is sampled from a
    power-law distribution so that some capabilities are rare —
    this makes routing genuinely hard (100 % success not guaranteed).
    """
    rng = random.Random(seed)

    if skew:
        # Zipf-like weights: first cap ~4x more requested than last
        n = len(CAPABILITIES)
        weights = [1.0 / (i + 1) for i in range(n)]
    else:
        weights = [1.0] * len(CAPABILITIES)

    n_agents = num_agents if num_agents is not None else NUM_AGENTS
    tasks = []
    for task_id in range(num_tasks):
        cap = rng.choices(CAPABILITIES, weights=weights, k=1)[0]
        tasks.append(Task(
            task_id             = task_id,
            source              = rng.randint(0, n_agents - 1),
            required_capability = cap,
            priority            = rng.uniform(0.5, 1.0),
            deadline            = rng.uniform(15.0, 50.0),
        ))
    return tasks