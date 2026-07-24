"""
EDA Module for Playground Series S6E7: Predicting Student Health Risk
Performs exploratory data analysis and generates visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

TARGET = "health_condition"
ID_COL = "id"


def load_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample_sub = pd.read_csv("sample_submission.csv")
    return train, test, sample_sub


def eda_overview(train, test, save_dir="."):
    print("=" * 60)
    print("TRAINING DATA OVERVIEW")
    print("=" * 60)
    print(f"Shape: {train.shape}")
    print(f"\nColumn dtypes:\n{train.dtypes}")
    print(f"\nMissing values:\n{train.isnull().sum()[train.isnull().sum() > 0]}")
    print(f"\nBasic statistics:\n{train.describe()}")

    print("\n" + "=" * 60)
    print("TEST DATA OVERVIEW")
    print("=" * 60)
    print(f"Shape: {test.shape}")
    print(f"\nMissing values:\n{test.isnull().sum()[test.isnull().sum() > 0]}")

    print("\n" + "=" * 60)
    print("TARGET DISTRIBUTION")
    print("=" * 60)
    print(train[TARGET].value_counts())
    print(f"\nClass proportions:\n{train[TARGET].value_counts(normalize=True)}")

    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in [ID_COL, TARGET]]

    cat_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in [TARGET]]

    print(f"\nNumerical features ({len(num_cols)}): {num_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")

    return num_cols, cat_cols


def plot_target_distribution(train, save_dir="."):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    train[TARGET].value_counts().plot.bar(ax=axes[0], color=["#2ecc71", "#e74c3c", "#f39c12"])
    axes[0].set_title("Target Class Counts")
    axes[0].set_ylabel("Count")
    train[TARGET].value_counts(normalize=True).plot.pie(
        ax=axes[1], autopct="%1.1f%%", colors=["#2ecc71", "#e74c3c", "#f39c12"]
    )
    axes[1].set_title("Target Class Proportions")
    axes[1].set_ylabel("")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/target_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: target_distribution.png")


def plot_correlation_heatmap(train, num_cols, save_dir="."):
    cols = num_cols[:20]
    corr = train[cols + [TARGET]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Correlation Heatmap (Top Numeric Features)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/correlation_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: correlation_heatmap.png")


def plot_feature_distributions(train, num_cols, save_dir="."):
    n = min(len(num_cols), 12)
    cols_plot = num_cols[:n]
    n_cols_grid = 3
    n_rows = (n + n_cols_grid - 1) // n_cols_grid
    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(5 * n_cols_grid, 4 * n_rows))
    axes = np.atleast_2d(axes)
    for i, col in enumerate(cols_plot):
        r, c = divmod(i, n_cols_grid)
        for label in train[TARGET].unique():
            subset = train[train[TARGET] == label][col].dropna()
            axes[r, c].hist(subset, bins=30, alpha=0.5, label=str(label), density=True)
        axes[r, c].set_title(col)
        axes[r, c].legend(fontsize=7)
    for i in range(n, n_rows * n_cols_grid):
        r, c = divmod(i, n_cols_grid)
        axes[r, c].set_visible(False)
    plt.suptitle("Feature Distributions by Target", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_distributions.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: feature_distributions.png")


def plot_top_features_vs_target(train, num_cols, save_dir="."):
    target_map = {v: i for i, v in enumerate(train[TARGET].unique())}
    y = train[TARGET].map(target_map)
    corrs = train[num_cols].corrwith(y).abs().sort_values(ascending=False)
    top6 = corrs.head(6).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(top6):
        for label in train[TARGET].unique():
            subset = train[train[TARGET] == label][col].dropna()
            axes[i].hist(subset, bins=30, alpha=0.4, label=str(label), density=True)
        axes[i].set_title(f"{col} (corr={corrs[col]:.3f})")
        axes[i].legend(fontsize=7)
    plt.suptitle("Top 6 Features vs Target", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/top_features_vs_target.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: top_features_vs_target.png")


def plot_categorical_features(train, cat_cols, save_dir="."):
    if not cat_cols:
        print("No categorical features to plot.")
        return
    n = min(len(cat_cols), 6)
    cols_plot = cat_cols[:n]
    n_cols_grid = 3
    n_rows = (n + n_cols_grid - 1) // n_cols_grid
    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(5 * n_cols_grid, 4 * n_rows))
    axes = np.atleast_2d(axes)
    for i, col in enumerate(cols_plot):
        r, c = divmod(i, n_cols_grid)
        ct = pd.crosstab(train[col], train[TARGET], normalize="index")
        ct.plot(kind="bar", stacked=True, ax=axes[r, c], colormap="Set2")
        axes[r, c].set_title(col)
        axes[r, c].tick_params(axis="x", rotation=45)
        axes[r, c].legend(fontsize=7)
    for i in range(n, n_rows * n_cols_grid):
        r, c = divmod(i, n_cols_grid)
        axes[r, c].set_visible(False)
    plt.suptitle("Categorical Features vs Target (Proportions)", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/categorical_vs_target.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: categorical_vs_target.png")


if __name__ == "__main__":
    train, test, sample_sub = load_data()
    num_cols, cat_cols = eda_overview(train, test)
    plot_target_distribution(train)
    plot_correlation_heatmap(train, num_cols)
    plot_feature_distributions(train, num_cols)
    plot_top_features_vs_target(train, num_cols)
    plot_categorical_features(train, cat_cols)
    print("\nEDA complete.")
