"""
TabNet training module with Optuna hyperparameter optimization and MLFlow tracking.
"""

import os
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import optuna
import mlflow
import mlflow.pytorch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from typing import Dict, Tuple
import pickle

from src.data.preprocessing import DataPreprocessor, get_X_y


class TabNetTrainer:
    """Trainer for TabNet credit card fraud detection model."""

    def __init__(
        self,
        experiment_name: str = "fraud-detection-tabnet",
        n_trials: int = 20,
        n_folds: int = 3,
        random_state: int = 42,
    ):
        self.experiment_name = experiment_name
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.random_state = random_state

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
        mlflow.set_experiment(experiment_name)

    def objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        # Hyperparameters for TabNet
        params = {
            "n_d": trial.suggest_int("n_d", 8, 64),
            "n_a": trial.suggest_int("n_a", 8, 64),
            "n_steps": trial.suggest_int("n_steps", 3, 10),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-4, 1e-2, log=True),
            "optimizer_params": dict(lr=trial.suggest_float("lr", 1e-3, 1e-1, log=True)),
            "mask_type": trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
            "verbose": 0,
            "seed": self.random_state,
        }

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        pr_auc_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_idx].values, X.iloc[val_idx].values
            y_train_fold, y_val_fold = y.iloc[train_idx].values, y.iloc[val_idx].values

            model = TabNetClassifier(**params)
            model.fit(
                X_train=X_train_fold,
                y_train=y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                eval_metric=["auc"],
                max_epochs=20,
                patience=5,
                batch_size=1024,
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False,
            )

            y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
            pr_auc = average_precision_score(y_val_fold, y_pred_proba)
            pr_auc_scores.append(pr_auc)

        mean_pr_auc = np.mean(pr_auc_scores)
        
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_pr_auc_mean", mean_pr_auc)

        return mean_pr_auc

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        print(f"Starting TabNet optimization with {self.n_trials} trials...")
        
        with mlflow.start_run(run_name="tabnet_optimization"):
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=self.n_trials)

            best_params = study.best_params
            print(f"Best parameters: {best_params}")

            # Reconstruct optimizer_params from best_params
            optimizer_params = {"lr": best_params.pop("lr")}
            
            # Train final model
            model = TabNetClassifier(
                **best_params, 
                optimizer_params=optimizer_params,
                seed=self.random_state
            )
            model.fit(
                X_train=X_train.values,
                y_train=y_train.values,
                eval_set=[(X_val.values, y_val.values)],
                eval_metric=["auc"],
                max_epochs=50,
                patience=10,
                batch_size=1024,
                virtual_batch_size=128,
            )

            # Log model
            mlflow.pytorch.log_model(model.network, "tabnet_model")
            
            return model


def main():
    preprocessor = DataPreprocessor()
    if os.path.exists("data/processed/train.csv"):
        train_df = pd.read_csv("data/processed/train.csv")
        val_df = pd.read_csv("data/processed/val.csv")
    else:
        train_df, val_df, _ = preprocessor.preprocess_pipeline()

    X_train, y_train = get_X_y(train_df)
    X_val, y_val = get_X_y(val_df)

    trainer = TabNetTrainer(n_trials=2)
    trainer.train(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    main()
