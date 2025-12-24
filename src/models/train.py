"""
Model training module with XGBoost, Optuna hyperparameter optimization, and MLFlow tracking.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import mlflow
import mlflow.xgboost
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict, Tuple
import pickle

from src.data.preprocessing import DataPreprocessor, get_X_y


class FraudDetectionTrainer:
    """Trainer for credit card fraud detection model."""

    def __init__(
        self,
        experiment_name: str = "fraud-detection",
        n_trials: int = 50,
        n_folds: int = 5,
        random_state: int = 42,
    ):
        """
        Initialize the trainer.

        Args:
            experiment_name: MLFlow experiment name
            n_trials: Number of Optuna trials
            n_folds: Number of cross-validation folds
            random_state: Random seed
        """
        self.experiment_name = experiment_name
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.random_state = random_state

        # Set MLFlow tracking URI from environment or default
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
        mlflow.set_experiment(experiment_name)

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for fraud detection.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold

        Returns:
            Dictionary of metrics
        """
        y_pred = (y_pred_proba >= threshold).astype(int)

        metrics = {
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
            "pr_auc": average_precision_score(y_true, y_pred_proba),
            "f1_score": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred),
        }

        return metrics

    def find_optimal_threshold(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray
    ) -> Tuple[float, float]:
        """
        Find optimal classification threshold using F1 score.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities

        Returns:
            Tuple of (optimal_threshold, best_f1)
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = (
            thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        )
        best_f1 = f1_scores[optimal_idx]

        return optimal_threshold, best_f1

    def objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Optuna objective function for hyperparameter optimization.

        Args:
            trial: Optuna trial object
            X: Features
            y: Target

        Returns:
            Mean PR-AUC score across folds
        """
        # Hyperparameter search space
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "random_state": self.random_state,
            # Hyperparameters to optimize
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 2),
        }

        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = (y == 0).sum() / (y == 1).sum()
        params["scale_pos_weight"] = scale_pos_weight

        # Cross-validation
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )
        pr_auc_scores = []
        roc_auc_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # Train model
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train_fold,
                y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False,
            )

            # Predict and evaluate
            y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
            pr_auc = average_precision_score(y_val_fold, y_pred_proba)
            roc_auc = roc_auc_score(y_val_fold, y_pred_proba)

            pr_auc_scores.append(pr_auc)
            roc_auc_scores.append(roc_auc)

        mean_pr_auc = np.mean(pr_auc_scores)
        mean_roc_auc = np.mean(roc_auc_scores)

        # Log to MLFlow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_pr_auc_mean", mean_pr_auc)
            mlflow.log_metric("cv_pr_auc_std", np.std(pr_auc_scores))
            mlflow.log_metric("cv_roc_auc_mean", mean_roc_auc)
            mlflow.log_metric("cv_roc_auc_std", np.std(roc_auc_scores))

        return mean_pr_auc

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = "confusion_matrix.png",
    ):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def plot_feature_importance(
        self, model: xgb.XGBClassifier, save_path: str = "feature_importance.png"
    ):
        """Plot and save feature importance."""
        fig, ax = plt.subplots(figsize=(10, 8))
        xgb.plot_importance(model, ax=ax, max_num_features=20)
        plt.title("Top 20 Feature Importances")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def plot_shap_summary(
        self,
        model: xgb.XGBClassifier,
        X_sample: pd.DataFrame,
        save_path: str = "shap_summary.png",
    ):
        """Generate and save SHAP summary plot."""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[xgb.XGBClassifier, Dict, float, CalibratedClassifierCV]:
        """
        Train the model with Optuna optimization and MLFlow tracking.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Tuple of (best_model, best_params, optimal_threshold)
        """
        print(f"Starting hyperparameter optimization with {self.n_trials} trials...")

        with mlflow.start_run(run_name="optuna_optimization"):
            # Optuna optimization
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: self.objective(trial, X_train, y_train),
                n_trials=self.n_trials,
                show_progress_bar=True,
            )

            best_params = study.best_params
            best_params["objective"] = "binary:logistic"
            best_params["eval_metric"] = "auc"
            best_params["tree_method"] = "hist"
            best_params["random_state"] = self.random_state
            best_params["scale_pos_weight"] = (y_train == 0).sum() / (
                y_train == 1
            ).sum()

            print(f"\nBest parameters: {best_params}")
            print(f"Best CV PR-AUC: {study.best_value:.4f}")

            # Train final model with best parameters
            print("\nTraining final model...")
            best_model = xgb.XGBClassifier(**best_params)
            best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            # Calibrate model
            print("Calibrating model probabilities...")
            calibrated_model = CalibratedClassifierCV(best_model, cv="prefit", method="sigmoid")
            calibrated_model.fit(X_val, y_val)
            
            # Use calibrated model for predictions
            y_val_pred_proba = calibrated_model.predict_proba(X_val)[:, 1]

            # Find optimal threshold
            optimal_threshold, best_f1 = self.find_optimal_threshold(
                y_val, y_val_pred_proba
            )
            print(f"Optimal threshold: {optimal_threshold:.4f} (F1: {best_f1:.4f})")

            # Calculate metrics
            val_metrics = self.calculate_metrics(
                y_val, y_val_pred_proba, optimal_threshold
            )

            # Log to MLFlow
            mlflow.log_params(best_params)
            mlflow.log_metric("val_roc_auc", val_metrics["roc_auc"])
            mlflow.log_metric("val_pr_auc", val_metrics["pr_auc"])
            mlflow.log_metric("val_f1_score", val_metrics["f1_score"])
            mlflow.log_metric("val_precision", val_metrics["precision"])
            mlflow.log_metric("val_recall", val_metrics["recall"])
            mlflow.log_metric("optimal_threshold", optimal_threshold)

            # Generate and log artifacts
            os.makedirs("artifacts", exist_ok=True)

            # Confusion matrix
            y_val_pred = (y_val_pred_proba >= optimal_threshold).astype(int)
            self.plot_confusion_matrix(
                y_val, y_val_pred, "artifacts/confusion_matrix.png"
            )
            mlflow.log_artifact("artifacts/confusion_matrix.png")

            # Feature importance
            self.plot_feature_importance(best_model, "artifacts/feature_importance.png")
            mlflow.log_artifact("artifacts/feature_importance.png")

            # SHAP summary (sample 1000 points for speed)
            X_sample = X_val.sample(
                min(1000, len(X_val)), random_state=self.random_state
            )
            self.plot_shap_summary(best_model, X_sample, "artifacts/shap_summary.png")
            mlflow.log_artifact("artifacts/shap_summary.png")

            # Log model
            mlflow.xgboost.log_model(best_model, "model")

            # Save threshold
            with open("artifacts/optimal_threshold.txt", "w") as f:
                f.write(str(optimal_threshold))
            mlflow.log_artifact("artifacts/optimal_threshold.txt")

            # Tag as best model
            mlflow.set_tag("best_model", "true")

            print("\n✓ Model training completed and logged to MLFlow")

        return best_model, best_params, optimal_threshold, calibrated_model


def main():
    """Main training pipeline."""
    # Load data
    print("Loading and preprocessing data...")
    preprocessor = DataPreprocessor()

    # Check if processed data exists
    if os.path.exists("data/processed/train.csv"):
        train_df = pd.read_csv("data/processed/train.csv")
        val_df = pd.read_csv("data/processed/val.csv")
        test_df = pd.read_csv("data/processed/test.csv")
    else:
        train_df, val_df, test_df = preprocessor.preprocess_pipeline()

    X_train, y_train = get_X_y(train_df)
    X_val, y_val = get_X_y(val_df)
    X_test, y_test = get_X_y(test_df)

    # Train model
    trainer = FraudDetectionTrainer(n_trials=5)
    best_model, best_params, optimal_threshold, calibrated_model = trainer.train(
        X_train, y_train, X_val, y_val
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
    test_metrics = trainer.calculate_metrics(
        y_test, y_test_pred_proba, optimal_threshold
    )

    print("\nTest Set Performance:")
    print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")

    # Save model locally
    os.makedirs("models", exist_ok=True)
    # Save the calibrated model wrapper
    with open("models/fraud_detector_calibrated.pkl", "wb") as f:
        pickle.dump(calibrated_model, f)
    
    # Also save the base XGBoost model for reference
    best_model.save_model("models/fraud_detector.json")

    # Save threshold and scaler
    with open("models/optimal_threshold.pkl", "wb") as f:
        pickle.dump(optimal_threshold, f)

    print("\n✓ Model saved to models/fraud_detector.json")


if __name__ == "__main__":
    main()
