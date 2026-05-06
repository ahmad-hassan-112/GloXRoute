import random
from typing import List

from config import NUM_TASKS, NUM_AGENTS, CAPABILITIES
from models import Task


def generate_tasks(num_tasks: int = NUM_TASKS) -> List[Task]:
    tasks = []

    for task_id in range(num_tasks):
        task = Task(
            task_id=task_id,
            source=random.randint(0, NUM_AGENTS - 1),
            required_capability=random.choice(CAPABILITIES),
            priority=random.uniform(0.5, 1.0),
            deadline=random.uniform(15.0, 50.0),
        )

        tasks.append(task)

    return tasks