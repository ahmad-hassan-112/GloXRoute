import os
from datetime import datetime
from dataclasses import asdict, is_dataclass
import json
import shutil

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    OUTPUT_DIR,
    SEED,
    COORDINATOR_UPDATE_INTERVAL,
)

from learning import HybridParameterLearner
from network_builder import build_ioa_network
from coordinator import GlobalCoordinator, LearnedGlobalCoordinator
from models import Task

from routers import (
    ShortestPathRouter,
    CentralizedRouter,
    DecentralizedGreedyRouter,
    BiRouterStyleRouter,
    HybridXAIRouter,
    NeuralHybridXAIRouter,
    LearnedHybridXAIRouter,
    GNNHybridXAIRouter,
)

from metrics import evaluate_route
from nn_learning import NeuralHybridLearner
from gnn_learning import GNNHybridLearner
from logging_utils import get_logger


class ExperimentRunner:
    def __init__(self, mode="all", model_dir=None):
        # Ensure the main output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Directories Setup
        self.dataset_dir = os.path.join(OUTPUT_DIR, "datasets")

        self.model_dir = model_dir or os.path.join(OUTPUT_DIR, "models", "latest")

        run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_output_dir = os.path.join(OUTPUT_DIR, run_stamp)
        os.makedirs(self.run_output_dir, exist_ok=True)  # Ensure the run-specific directory exists

        # Path for logging experiment progress and info
        self.run_log_path = os.path.join(self.run_output_dir, "run.log")
        self.logger = get_logger("experiment", self.run_log_path)  # Initialize logger

        # Build the network topology and initialize agents with their capabilities and attributes
        self.G, self.agents = build_ioa_network(seed=SEED)

        # Load pre-generated tasks for training, validation, and testing
        self.train_tasks = self.load_tasks_csv("train_tasks.csv")
        self.val_tasks = self.load_tasks_csv("val_tasks.csv")
        self.test_tasks = self.load_tasks_csv("test_tasks.csv")

        # Load prebuilt datasets for neural network (NN) training and validation
        self.nn_train_df = self.load_dataset_csv("nn_train_dataset.csv")
        self.nn_val_df = self.load_dataset_csv("nn_val_dataset.csv")

        # Load prebuilt datasets for graph neural network (GNN) training and validation
        self.gnn_train_df = self.load_dataset_csv("gnn_train_dataset.csv")
        self.gnn_val_df = self.load_dataset_csv("gnn_val_dataset.csv")

        # Log the run directory and dataset location
        self.logger.info("Run output directory: %s", self.run_output_dir)
        self.logger.info("Using datasets from: %s", self.dataset_dir)

        # Initialize coordinators
        self.coordinator = GlobalCoordinator(self.G, self.agents)  # Standard global coordinator
        self.learned_coordinator = LearnedGlobalCoordinator(self.G, self.agents, blend_alpha=0.7)  # Learned coordinator with blending

        # Initialize learners for different routing strategies
        self.nn_learner = NeuralHybridLearner()    # NeuralHybridXAI learner
        self.gnn_learner = GNNHybridLearner()      # GNNHybridXAI learner
        self.hybrid_learner = HybridParameterLearner()  # Logistic regression / hybrid XAI learner

        # Provide graph context to GNN learner (graph structure, agents, coordinator)
        self.gnn_learner.set_graph_context(self.G, self.agents, self.learned_coordinator)

        # Depending on mode, either train models or load pre-trained models
        if mode in ("all", "train"):
            self.train_models()
        elif mode == "test":
            self.load_models()

        # Initialize router objects for evaluation and routing experiments
        self.init_routers()

    def train_models(self):
        # LearnedGlobalCoordinator
        self.logger.info("Training LearnedGlobalCoordinator...")
        self.learned_coordinator.train_guidance_model(self.train_tasks)
        self.logger.info("LearnedGlobalCoordinator training complete")

        # NeuralHybridXAI
        self.logger.info("Using prebuilt neural datasets")
        self.logger.info("NN datasets loaded: train=%d, val=%d", len(self.nn_train_df), len(self.nn_val_df))
        self.logger.info("Training NeuralHybridXAI model...")

        self.epoch_results = self.nn_learner.train(
            self.nn_train_df,
            val_df=self.nn_val_df,
            output_dir=self.run_output_dir,
            logger=get_logger("train.nn", self.run_log_path)
        )

        self.logger.info("Epoch results saved: %s/nn_epoch_results.csv", self.run_output_dir)
        self.logger.info("Model saved: %s/learned_hybrid_xai_nn.pt", self.run_output_dir)
        self.logger.info("NeuralHybridXAI training complete")

        # LearnedHybridXAIRouter
        self.logger.info("Training HybridParameterLearner for LearnedHybridXAIRouter...")
        results = self.hybrid_learner.train(
            G=self.G,
            agents=self.agents,
            coordinator=self.learned_coordinator,
            train_tasks=self.train_tasks,
            val_tasks=self.val_tasks,
            save_path=os.path.join(self.model_dir, "learned_hybrid_xai_hybrid.pkl")
        )
        self.logger.info("HybridParameterLearner training complete")
        self.logger.info("Validation results: %s", results)
        print("Validation results:", results)
        self.logger.info("HybridParameterLearner training complete")

        # GNNHybridXAI
        self.logger.info("Using prebuilt GNN datasets")
        self.logger.info("GNN datasets loaded: train=%d, val=%d", len(self.gnn_train_df), len(self.gnn_val_df))
        self.logger.info("Training GNNHybridXAI model...")

        self.gnn_epoch_results = self.gnn_learner.train(
            self.gnn_train_df,
            val_df=self.gnn_val_df,
            output_dir=self.run_output_dir,
            logger=get_logger("train.gnn", self.run_log_path)
        )

        self.logger.info("GNN epoch results saved: %s/gnn_epoch_results.csv", self.run_output_dir)
        self.logger.info("Model saved: %s/learned_hybrid_xai_gnn.pt", self.run_output_dir)
        self.logger.info("GNNHybridXAI training complete")

        os.makedirs(self.model_dir, exist_ok=True)
        shutil.copyfile(
            os.path.join(self.run_output_dir, "learned_hybrid_xai_nn.pt"),
            os.path.join(self.model_dir, "learned_hybrid_xai_nn.pt"),
        )
        shutil.copyfile(
            os.path.join(self.run_output_dir, "learned_hybrid_xai_gnn.pt"),
            os.path.join(self.model_dir, "learned_hybrid_xai_gnn.pt"),
        )
        self.learned_coordinator.save(
            os.path.join(self.model_dir, "learned_global_coordinator.npz")
        )
        self.logger.info("Latest models updated in: %s", self.model_dir)

    def load_models(self):
        nn_path = os.path.join(self.model_dir, "learned_hybrid_xai_nn.pt")
        gnn_path = os.path.join(self.model_dir, "learned_hybrid_xai_gnn.pt")
        hybrid_learner_path = os.path.join(self.model_dir, "learned_hybrid_xai_hybrid.pkl")
        coordinator_path = os.path.join(self.model_dir, "learned_global_coordinator.npz")
        if not os.path.exists(nn_path) or not os.path.exists(gnn_path) or not os.path.exists(coordinator_path):
            raise FileNotFoundError(
                f"Missing model files in {self.model_dir}. Run training first."
            )

        self.nn_learner.load(nn_path)
        self.gnn_learner.load(gnn_path)
        self.hybrid_learner.load(hybrid_learner_path)
        self.learned_coordinator.load(coordinator_path)
        self.logger.info("Loaded trained models from: %s", self.model_dir)

    def init_routers(self):
        self.routers = [
            ShortestPathRouter(),
            # CentralizedRouter(),
            # DecentralizedGreedyRouter(),
            BiRouterStyleRouter(),
            # HybridXAIRouter(self.coordinator, router_name="HybridXAI_HeuristicCoord"),
            HybridXAIRouter(self.learned_coordinator, router_name="HybridXAI_LearnedCoord"),
            NeuralHybridXAIRouter(self.learned_coordinator, self.nn_learner),
            GNNHybridXAIRouter(self.learned_coordinator, self.gnn_learner),
            LearnedHybridXAIRouter(self.learned_coordinator, self.hybrid_learner),
        ]

    def run(self):
        rows = []
        self.logger.info("Starting final comparison testing")

        for index, task in enumerate(tqdm(self.test_tasks, desc="Running final test comparison")):
            if index > 0 and index % COORDINATOR_UPDATE_INTERVAL == 0:
                self.coordinator.update_guidance()
                self.learned_coordinator.update_guidance()

            for router in self.routers:
                decision = router.route(self.G, self.agents, task)

                result = evaluate_route(
                    G=self.G,
                    agents=self.agents,
                    task=task,
                    method_name=router.name,
                    path=decision.path,
                    extra_overhead=decision.extra_overhead,
                    explanation=decision.explanation,
                )

                if is_dataclass(result):
                    rows.append(asdict(result))
                else:
                    rows.append(result)

        all_results_df = pd.DataFrame(rows)

        # 1. Full comparison file
        all_results_df.to_csv(
            f"{self.run_output_dir}/comparison_all_results.csv",
            index=False
        )

        comparison_summary = self.summarize(all_results_df)

        comparison_summary.to_csv(
            f"{self.run_output_dir}/comparison_summary_results.csv",
            index=False
        )
        comparison_summary.to_json(
            f"{self.run_output_dir}/comparison_summary_results.json",
            orient="records",
            indent=2
        )

        # 2. Separate neural model result file
        # nn_only_df = all_results_df[
        #     all_results_df["method"] == "NeuralHybridXAI"
        # ].copy()

        # nn_only_df.to_csv(
        #     f"{self.run_output_dir}/neural_hybrid_xai_final_results.csv",
        #     index=False
        # )

        # nn_only_summary = self.summarize(nn_only_df)

        # nn_only_summary.to_csv(
        #     f"{self.run_output_dir}/neural_hybrid_xai_summary_results.csv",
        #     index=False
        # )

        self.logger.info("Full comparison summary generated")
        self.logger.info("Summary table:\n%s", comparison_summary.to_string(index=False))

        # self.logger.info("NeuralHybridXAI-only summary generated")
        # self.logger.info("NeuralHybridXAI summary table:\n%s", nn_only_summary.to_string(index=False))

        best_row = comparison_summary.sort_values("final_tradeoff_score", ascending=False).iloc[0].to_dict()
        with open(f"{self.run_output_dir}/best_method.json", "w", encoding="utf-8") as f:
            json.dump(best_row, f, indent=2)
        self.logger.info(
            "Best method: %s (final_tradeoff_score=%.4f)",
            best_row["method"],
            best_row["final_tradeoff_score"],
        )

        self.plot_comparison(comparison_summary)
        nn_epoch_path = os.path.join(self.run_output_dir, "nn_epoch_results.csv")
        if os.path.exists(nn_epoch_path):
            self.plot_epoch_results()
        self.logger.info("Run completed")

    def load_tasks_csv(self, filename):
        path = os.path.join(self.dataset_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing dataset file: {path}. Run prepare_datasets.py first."
            )
        df = pd.read_csv(path)
        return [
            Task(
                task_id=int(row.task_id),
                source=int(row.source),
                required_capability=str(row.required_capability),
                priority=float(row.priority),
                deadline=float(row.deadline),
            )
            for row in df.itertuples(index=False)
        ]

    def load_dataset_csv(self, filename):
        path = os.path.join(self.dataset_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing dataset file: {path}. Run prepare_datasets.py first."
            )
        return pd.read_csv(path)

    def summarize(self, df):
        summary = df.groupby("method").agg(
            success_rate=("success", "mean"),
            avg_latency=("latency", "mean"),
            avg_hops=("hops", "mean"),
            avg_overhead=("communication_overhead", "mean"),
            explanation_rate=("explanation_available", "mean"),
        ).reset_index()

        summary["success_rate"] *= 100
        summary["explanation_rate"] *= 100

        summary["latency_score"] = 1 / (1 + summary["avg_latency"])
        summary["overhead_score"] = 1 / (1 + summary["avg_overhead"])

        summary["final_tradeoff_score"] = (
            0.35 * (summary["success_rate"] / 100)
            + 0.25 * summary["latency_score"]
            + 0.20 * summary["overhead_score"]
            + 0.20 * (summary["explanation_rate"] / 100)
        )

        return summary

    def plot_comparison(self, summary):
        metrics = [
            "success_rate",
            "avg_latency",
            "avg_hops",
            "avg_overhead",
            "explanation_rate",
            "final_tradeoff_score",
        ]

        for metric in metrics:
            plt.figure(figsize=(10, 5))
            plt.bar(summary["method"], summary[metric])
            plt.xticks(rotation=30, ha="right")
            plt.ylabel(metric)
            plt.title(f"Comparison of Routing Methods: {metric}")
            plt.tight_layout()
            plt.savefig(f"{self.run_output_dir}/comparison_{metric}.png", dpi=300)
            plt.close()

    def plot_epoch_results(self):
        epoch_df = pd.read_csv(f"{self.run_output_dir}/nn_epoch_results.csv")

        plt.figure(figsize=(8, 5))
        plt.plot(epoch_df["epoch"], epoch_df["train_loss"], label="Train Loss")
        plt.plot(epoch_df["epoch"], epoch_df["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("NeuralHybridXAI Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.run_output_dir}/nn_training_loss.png", dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epoch_df["epoch"], epoch_df["val_accuracy"], label="Validation Accuracy")
        plt.plot(epoch_df["epoch"], epoch_df["val_f1"], label="Validation F1")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("NeuralHybridXAI Validation Performance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.run_output_dir}/nn_validation_scores.png", dpi=300)
        plt.close()