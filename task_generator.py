"""
task_generator.py — generate routing tasks.

Two modes
---------
"exhaustive" (RECOMMENDED, default)
    Enumerate every (source_agent, required_capability) pair exactly once.
    For 100 agents x 8 capabilities this gives 800 tasks per seed, of which
    ~700 require actual routing (the rest are source-capable).

    Why this matters: the population of distinct routing problems is only
    N_agents x N_capabilities. Randomly sampling 2000 tasks from a population
    of 800 draws each problem ~2.5x, adds no information, and lets the
    HybridXAI route cache score free hits on repeats. Exhaustive enumeration
    removes within-seed sampling noise entirely, so all reported variance
    comes from the seed (topology + agent placement), which is where it
    belongs statistically.

"random" (legacy)
    Uniform random sampling, kept for backwards compatibility.

Deadlines
---------
Deadlines are now RELATIVE to the shortest achievable effective latency
for that task, scaled by priority:

    deadline_factor = 1.0 + (1 - priority) * 0.8

so a high-priority task (priority=1.0) has a tight factor of 1.0 while a
low-priority task (priority=0.5) gets 1.4. This makes both `priority` and
`deadline` meaningful — previously both fields were generated but never
used by any router or metric.
"""

import random
from typing import List

from config import NUM_TASKS, NUM_AGENTS, CAPABILITIES, TASK_MODE
from models import Task


def _make_task(task_id: int, source: int, cap: str, rng: random.Random) -> Task:
    priority        = rng.uniform(0.5, 1.0)
    deadline_factor = 1.0 + (1.0 - priority) * 0.8
    return Task(
        task_id             = task_id,
        source              = source,
        required_capability = cap,
        priority            = priority,
        deadline            = deadline_factor,   # now a FACTOR, not absolute
    )


def generate_tasks(
    num_tasks: int = NUM_TASKS,
    seed: int = 42,
    num_agents: int = None,
    mode: str = None,
) -> List[Task]:
    rng      = random.Random(seed)
    n_agents = num_agents if num_agents is not None else NUM_AGENTS
    mode     = mode or TASK_MODE

    if mode == "exhaustive":
        tasks, tid = [], 0
        for source in range(n_agents):
            for cap in CAPABILITIES:
                tasks.append(_make_task(tid, source, cap, rng))
                tid += 1
        rng.shuffle(tasks)   # shuffle so coordinator refresh isn't correlated
        return tasks

    # legacy random mode
    tasks = []
    for task_id in range(num_tasks):
        source = rng.randint(0, n_agents - 1)
        cap    = rng.choice(CAPABILITIES)
        tasks.append(_make_task(task_id, source, cap, rng))
    return tasks