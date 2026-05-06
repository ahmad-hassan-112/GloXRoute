import numpy as np
import joblib
import networkx as nx

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from config import MAX_HOPS
from models import Agent, Task
from utils import capable_nodes, path_latency


class FeatureExtractor:
    """
    Converts one candidate next-hop decision into numerical features.
    These features replace manually selected routing weights.
    """

    feature_names = [
        "capability_match",
        "global_hint",
        "reliability",
        "resource",
        "load",
        "congestion",
        "latency_norm",
        "bandwidth_norm",
        "degree_norm",
    ]

    def extract(self, G, agents, coordinator, current_node, candidate_node, required_capability):
        capability_match = 1.0 if required_capability in agents[candidate_node].capabilities else 0.0

        global_hint = coordinator.guidance[candidate_node][required_capability]
        reliability = agents[candidate_node].reliability
        resource = agents[candidate_node].resource
        load = agents[candidate_node].load
        congestion = coordinator.congestion_penalty(candidate_node)

        latency = G[current_node][candidate_node]["latency"]
        bandwidth = G[current_node][candidate_node]["bandwidth"]

        latency_norm = latency / 10.0
        bandwidth_norm = bandwidth / 5.0
        degree_norm = G.degree[candidate_node] / max(1, len(G.nodes()))

        return np.array([
            capability_match,
            global_hint,
            reliability,
            resource,
            load,
            congestion,
            latency_norm,
            bandwidth_norm,
            degree_norm,
        ], dtype=float)


class CentralizedOracle:
    """
    Teacher model.
    It has full network information and provides the best path.
    We use it only to generate training labels.
    """

    def best_path(self, G, agents, task: Task):
        source = task.source
        required = task.required_capability

        candidates = capable_nodes(agents, required)

        best_path = [source]
        best_score = float("inf")

        for target in candidates:
            try:
                path = nx.shortest_path(G, source, target, weight="latency")

                latency = path_latency(G, agents, path)
                load_penalty = agents[target].load * 5.0
                reliability_bonus = agents[target].reliability * 3.0
                resource_bonus = agents[target].resource * 2.0

                score = latency + load_penalty - reliability_bonus - resource_bonus

                if score < best_score:
                    best_score = score
                    best_path = path

            except nx.NetworkXNoPath:
                continue

        return best_path


class TrainingDataBuilder:
    """
    Generates training data from simulation.

    X = candidate next-hop features
    y = 1 if candidate is the centralized oracle's selected next-hop, else 0
    """

    def __init__(self, G, agents, coordinator):
        self.G = G
        self.agents = agents
        self.coordinator = coordinator
        self.extractor = FeatureExtractor()
        self.oracle = CentralizedOracle()

    def build(self, tasks):
        X = []
        y = []

        for task in tasks:
            oracle_path = self.oracle.best_path(self.G, self.agents, task)

            if len(oracle_path) <= 1:
                continue

            current = task.source
            required = task.required_capability

            for step in range(min(MAX_HOPS, len(oracle_path) - 1)):
                oracle_next = oracle_path[step + 1]

                neighbors = list(self.G.neighbors(current))

                for candidate in neighbors:
                    features = self.extractor.extract(
                        self.G,
                        self.agents,
                        self.coordinator,
                        current,
                        candidate,
                        required,
                    )

                    label = 1 if candidate == oracle_next else 0

                    X.append(features)
                    y.append(label)

                current = oracle_next

                if required in self.agents[current].capabilities:
                    break

        return np.array(X), np.array(y)


class HybridParameterLearner:
    """
    Learns routing parameters automatically using logistic regression.
    """

    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ))
        ])

        self.extractor = FeatureExtractor()

    def fit(self, X, y):
        if len(X) == 0:
            raise ValueError("No training data generated. Increase number of tasks or agents.")

        self.model.fit(X, y)
        return self

    def predict_score(self, features):
        """
        Returns probability that this candidate is the best next-hop.
        """
        features = np.array(features).reshape(1, -1)
        return self.model.predict_proba(features)[0][1]

    def explain(self, features):
        """
        Simple XAI-style explanation using learned coefficient contributions.
        """
        scaler = self.model.named_steps["scaler"]
        clf = self.model.named_steps["clf"]

        features = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features)[0]
        coefficients = clf.coef_[0]

        contributions = scaled_features * coefficients

        ranked = sorted(
            zip(FeatureExtractor.feature_names, contributions),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        explanation_parts = [
            f"{name}={value:.3f}"
            for name, value in ranked[:4]
        ]

        return "top_factors: " + ", ".join(explanation_parts)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        return self