import os
import random
import networkx as nx
from experiment import ExperimentRunner
from coordinator import LearnedGlobalCoordinator
from models import Agent
from config import CAPABILITIES, PROTOCOLS
from models import Task

def generate_benchmark_graphs(num_agents=50, seed=42):
    """
    Generate benchmark graphs with agents for IoA experiments.
    Returns:
        graphs: dict[str, nx.Graph] - benchmark graphs with edge attributes
        agents_dict: dict[str, dict[int, Agent]] - agents for each graph
    """
    graphs = {}
    agents_dict = {}

    print("\n=== Creating benchmark graphs ===", flush=True)
    random.seed(seed)

    # Helper: generate connected ER graph
    def connected_erdos_renyi(n, p, seed, max_tries=100):
        tries = 0
        G = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
        while not nx.is_connected(G) and tries < max_tries:
            seed += 1
            G = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
            tries += 1
        if not nx.is_connected(G):
            raise RuntimeError("Failed to generate connected ER graph")
        return G

    # 1️⃣ Erdos-Renyi
    print("Creating Erdos-Renyi graph...", flush=True)
    graphs["erdos_renyi"] = connected_erdos_renyi(num_agents, p=0.12, seed=seed)

    # 2️⃣ Watts-Strogatz
    print("Creating Watts-Strogatz graph...", flush=True)
    graphs["watts_strogatz"] = nx.connected_watts_strogatz_graph(n=num_agents, k=4, p=0.1, seed=seed)

    # 3️⃣ Barabasi-Albert
    print("Creating Barabasi-Albert graph...", flush=True)
    graphs["barabasi_albert"] = nx.barabasi_albert_graph(n=num_agents, m=2, seed=seed)

    # 4️⃣ Grid
    print("Creating Grid graph...", flush=True)
    side = int(num_agents ** 0.5)
    G_grid = nx.grid_2d_graph(side, side)
    graphs["grid"] = nx.convert_node_labels_to_integers(G_grid)

    # 5️⃣ Star
    print("Creating Star graph...", flush=True)
    graphs["star"] = nx.star_graph(num_agents - 1)

    # 6️⃣ Cycle
    print("Creating Cycle graph...", flush=True)
    graphs["cycle"] = nx.cycle_graph(num_agents)

    # Add edge attributes and agents
    for name, G in graphs.items():
        # Add edge attributes
        for u, v in G.edges():
            G[u][v]["latency"] = round(random.uniform(1.0, 10.0), 2)
            G[u][v]["bandwidth"] = round(random.uniform(0.5, 5.0), 2)

        # Create agents for each node
        agents = {}
        for node in G.nodes():
            agents[node] = Agent(
                agent_id=node,
                capabilities=set(random.sample(CAPABILITIES, k=1)),
                reliability=random.uniform(0.65, 0.99),
                load=random.uniform(0.0, 1.0),
                resource=random.uniform(0.4, 1.0),
                protocol=random.choice(PROTOCOLS),
            )
        agents_dict[name] = agents

    print("\n=== All benchmark graphs created ===", flush=True)
    return graphs, agents_dict

def generate_tasks_for_graph(G, num_tasks=100):
    nodes = list(G.nodes())
    tasks = []
    for i in range(num_tasks):
        src, tgt = random.sample(nodes, 2)
        tasks.append(Task(
            task_id=i,
            source=src,
            required_capability=random.choice(list(CAPABILITIES)),
            priority=random.uniform(0, 1),
            deadline=random.uniform(1, 10)
        ))
    return tasks


def run_all_benchmarks():
    graphs, agents_dict = generate_benchmark_graphs()

    for name, G in graphs.items():
        agents = agents_dict[name]
        print(f"\n=== Running benchmark for graph: {name} with {len(agents)} agents ===", flush=True)

        # Initialize ExperimentRunner in test mode
        runner = ExperimentRunner(mode="test")

        # Override network and agents with benchmark graph
        runner.G = G
        runner.agents = agents
        runner.test_tasks = generate_tasks_for_graph(G, num_tasks=2000)

        # Re-initialize learned coordinator and GNN context
        runner.learned_coordinator = LearnedGlobalCoordinator(G, agents, blend_alpha=0.7)
        runner.gnn_learner.set_graph_context(G, agents, runner.learned_coordinator)

        # Re-initialize routers
        runner.init_routers()

        # Run evaluation
        runner.run()
        print(f"=== Completed benchmark for graph: {name} ===\n", flush=True)


if __name__ == "__main__":
    run_all_benchmarks()