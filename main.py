"""
Main Pipeline for Playground Series S6E7: Predicting Student Health Risk
========================================================================
Multi-class classification: at-risk / unhealthy / fit
Metric: Balanced Accuracy

Usage:
  python main.py              # Full pipeline
  python main.py --skip-eda   # Skip EDA, go straight to modeling
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import warnings
warnings.filterwarnings("ignore")

from eda import load_data
from preprocessing import (
    identify_features, build_preprocessor, prepare_model_data
)
from models import (
    train_lgb_oof, train_xgb_oof, train_cat_oof,
    evaluate_ensemble, HAS_LGB, HAS_XGB, HAS_CAT
)

TARGET = "health_condition"
ID_COL = "id"
N_FOLDS = 5
SEED = 42
OUTPUT_FILE = "submission.csv"


def run_eda(train, test):
    from eda import (
        eda_overview, plot_target_distribution, plot_correlation_heatmap,
        plot_feature_distributions, plot_top_features_vs_target,
        plot_categorical_features
    )
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    num_cols, cat_cols = eda_overview(train, test)
    plot_target_distribution(train)
    plot_correlation_heatmap(train, num_cols)
    plot_feature_distributions(train, num_cols)
    plot_top_features_vs_target(train, num_cols)
    plot_categorical_features(train, cat_cols)


def train_single_model_on_folds(X_train, y_train, X_test, cat_cols, cat_indices,
                                 model_name, n_folds=N_FOLDS, seed=SEED):
    """Train a single model type with CV and return OOF + test predictions."""
    print(f"\n{'=' * 60}")
    print(f"Training {model_name}")
    print(f"{'=' * 60}")

    if model_name == "lightgbm":
        oof, test_p, info = train_lgb_oof(X_train, y_train, X_test, cat_cols, n_folds, seed)
    elif model_name == "xgboost":
        oof, test_p, info = train_xgb_oof(X_train, y_train, X_test, n_folds, seed)
    elif model_name == "catboost":
        oof, test_p, info = train_cat_oof(X_train, y_train, X_test, cat_indices, n_folds, seed)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return oof, test_p, info


def optimize_ensemble_weights(oof_list, y_true, model_names):
    """
    Optimize ensemble weights using scipy.optimize.
    """
    from scipy.optimize import minimize

    def neg_score(w):
        w = np.abs(w)
        w = w / w.sum()
        ensemble = np.zeros_like(oof_list[0])
        for pred, weight in zip(oof_list, w):
            ensemble += pred * weight
        y_pred = np.argmax(ensemble, axis=1)
        return -balanced_accuracy_score(y_true, y_pred)

    n = len(oof_list)
    x0 = np.ones(n) / n
    bounds = [(0, 1)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    result = minimize(neg_score, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                      options={"maxiter": 200})
    best_weights = np.abs(result.x)
    best_weights = best_weights / best_weights.sum()
    return list(best_weights), -result.fun


def create_submission(test_ids, test_preds, le_target, output=OUTPUT_FILE):
    """Create submission file from test predictions."""
    y_pred = np.argmax(test_preds, axis=1)
    labels = le_target.inverse_transform(y_pred)
    submission = pd.DataFrame({ID_COL: test_ids, TARGET: labels})
    submission.to_csv(output, index=False)
    print(f"\nSubmission saved to {output}")
    print(f"Shape: {submission.shape}")
    print(f"Predictions:\n{submission[TARGET].value_counts()}")
    return submission


def main():
    parser = argparse.ArgumentParser(description="TPS S6E7 Student Health Risk")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA step")
    parser.add_argument("--quick", action="store_true", help="Quick mode: LGB only, 3 folds")
    parser.add_argument("--all-models", action="store_true", help="Include CatBoost (slow on large data)")
    parser.add_argument("--folds", type=int, default=N_FOLDS, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Output filename")
    args = parser.parse_args()

    start = time.time()

    print("=" * 60)
    print("TPS S6E7: Predicting Student Health Risk")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading data...")
    train, test, sample_sub = load_data()
    print(f"  Train: {train.shape}, Test: {test.shape}")
    print(f"  Target classes: {train[TARGET].unique()}")

    # 2. EDA
    if not args.skip_eda:
        print("\n[2/5] Running EDA...")
        run_eda(train, test)
    else:
        print("\n[2/5] Skipping EDA...")

    # 3. Preprocessing & Feature Engineering
    print("\n[3/5] Preprocessing & Feature Engineering...")
    train_processed, test_processed, num_cols, cat_cols, le_dict, te_maps = \
        build_preprocessor(train, test, TARGET)

    X_train, X_test, y_train, feature_cols, cat_indices, le_target = \
        prepare_model_data(train_processed, test_processed, num_cols, cat_cols, TARGET)

    print(f"  Feature matrix: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"  Target distribution: {np.bincount(y_train)}")

    # 4. Model Training
    n_folds = 3 if args.quick else args.folds
    print("\n[4/5] Training models with {}-fold CV...".format(n_folds))

    results = {}
    oof_list = []
    test_list = []
    model_names = []

    if args.quick:
        available_models = ["lightgbm"] if HAS_LGB else []
    else:
        available_models = []
        if HAS_LGB:
            available_models.append("lightgbm")
        if HAS_XGB:
            available_models.append("xgboost")
        if args.all_models and HAS_CAT:
            available_models.append("catboost")

    if not available_models:
        print("ERROR: No boosting libraries available. Install lightgbm, xgboost, catboost.")
        sys.exit(1)

    print(f"  Available models: {available_models}")

    for model_name in available_models:
        oof, test_p, info = train_single_model_on_folds(
            X_train, y_train, X_test, cat_cols, cat_indices,
            model_name, n_folds, args.seed
        )
        if oof is not None:
            oof_list.append(oof)
            test_list.append(test_p)
            model_names.append(model_name)
            results[model_name] = info

    if not oof_list:
        print("ERROR: No models trained successfully.")
        sys.exit(1)

    # 5. Ensemble & Evaluation
    print("\n[5/5] Building ensemble...")

    if len(oof_list) == 1:
        best_weights = [1.0]
        final_oof = oof_list[0]
        final_test = test_list[0]
    else:
        print("  Optimizing ensemble weights...")
        best_weights, opt_score = optimize_ensemble_weights(oof_list, y_train, model_names)
        print(f"  Optimal weights: {dict(zip(model_names, [f'{w:.2f}' for w in best_weights]))}")

        final_oof = np.zeros_like(oof_list[0])
        final_test = np.zeros_like(test_list[0])
        for pred, w in zip(oof_list, best_weights):
            final_oof += pred * w
        for pred, w in zip(test_list, best_weights):
            final_test += pred * w

    # Report
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    for name, info in results.items():
        print(f"  {name}: CV={info['mean']:.5f} +/- {info['std']:.5f}")

    final_y = np.argmax(final_oof, axis=1)
    final_score = balanced_accuracy_score(y_train, final_y)
    print(f"\n  Ensemble CV Balanced Accuracy: {final_score:.5f}")

    # Feature importance summary
    print("\n" + "=" * 60)
    print("TOP 15 FEATURES (averaged across models)")
    print("=" * 60)
    all_imp = pd.DataFrame()
    for name, info in results.items():
        imp = info["importance"]
        imp_norm = imp / imp.sum()
        all_imp[name] = imp_norm
    avg_imp = all_imp.mean(axis=1).sort_values(ascending=False)
    print(avg_imp.head(15).to_string())

    # Save submission
    test_ids = sample_sub[ID_COL].values
    submission = create_submission(test_ids, final_test, le_target, args.output)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
    print("Done!")

    return submission


if __name__ == "__main__":
    main()
