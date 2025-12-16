"""
Exploratory Data Analysis (EDA) script for Credit Card Fraud Detection.
Generates visualizations and statistics to understand the dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def load_data(path="data/raw/creditcard.csv"):
    """Load dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def plot_class_distribution(df, save_path="reports/figures/class_distribution.png"):
    """Plot class distribution."""
    plt.figure(figsize=(8, 6))
    sns.countplot(x="Class", data=df)
    plt.title("Class Distribution \n (0: No Fraud || 1: Fraud)", fontsize=14)
    plt.yscale("log")  # Log scale due to high imbalance
    plt.ylabel("Count (Log Scale)")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


def plot_time_amount_distribution(df, save_dir="reports/figures"):
    """Plot distribution of Time and Amount features."""
    # Time
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df["Time"], bins=50, kde=True)
    plt.title("Distribution of Time Feature")

    plt.subplot(1, 2, 2)
    sns.histplot(
        df[df["Class"] == 1]["Time"], bins=50, kde=True, color="red", label="Fraud"
    )
    sns.histplot(
        df[df["Class"] == 0]["Time"],
        bins=50,
        kde=True,
        color="blue",
        label="Normal",
        alpha=0.3,
    )
    plt.title("Time Distribution by Class")
    plt.legend()
    plt.savefig(f"{save_dir}/time_distribution.png")
    plt.close()

    # Amount
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df["Amount"], bins=50, kde=True)
    plt.title("Distribution of Amount Feature")
    plt.yscale("log")

    plt.subplot(1, 2, 2)
    sns.boxplot(x="Class", y="Amount", data=df)
    plt.title("Amount Distribution by Class")
    plt.yscale("log")
    plt.savefig(f"{save_dir}/amount_distribution.png")
    plt.close()
    print(f"Saved time and amount distributions to {save_dir}")


def plot_correlation_matrix(df, save_path="reports/figures/correlation_matrix.png"):
    """Plot correlation matrix."""
    plt.figure(figsize=(24, 20))
    corr = df.corr()
    sns.heatmap(corr, cmap="coolwarm_r", annot_kws={"size": 20}, annot=False)
    plt.title("Correlation Matrix", fontsize=14)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


def plot_feature_distributions(
    df, save_path="reports/figures/feature_distributions.png"
):
    """Plot distributions of V features."""
    v_features = [f"V{i}" for i in range(1, 29)]

    plt.figure(figsize=(20, 40))
    for i, col in enumerate(v_features):
        plt.subplot(8, 4, i + 1)
        sns.kdeplot(
            df[df["Class"] == 0][col], label="Legitimate", shade=True, color="blue"
        )
        sns.kdeplot(df[df["Class"] == 1][col], label="Fraud", shade=True, color="red")
        plt.title(f"{col} Distribution")
        if i == 0:
            plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


def main():
    print("Starting EDA...")
    os.makedirs("reports/figures", exist_ok=True)

    df = load_data()
    print(f"Data loaded: {df.shape}")

    # Basic Stats
    print("\nClass Balance:")
    print(df["Class"].value_counts(normalize=True))

    print("\nGenerating visualizations...")
    plot_class_distribution(df)
    plot_time_amount_distribution(df)
    plot_correlation_matrix(df)
    plot_feature_distributions(df)

    print("\nEDA Completed. Check reports/figures/ for visualizations.")


if __name__ == "__main__":
    main()
