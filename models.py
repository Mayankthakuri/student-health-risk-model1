"""
Modeling Module for Playground Series S6E7
LightGBM + XGBoost + CatBoost ensemble with CV and balanced accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import warnings
warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Warning: lightgbm not installed")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: xgboost not installed")

try:
    from catboost import CatBoostClassifier, Pool
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("Warning: catboost not installed")


def make_lgb_params(num_class=3, n_estimators=1500):
    return {
        "objective": "multiclass",
        "num_class": num_class,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": n_estimators,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
        "class_weight": "balanced",
    }


def make_xgb_params(num_class=3, n_estimators=1000):
    return {
        "objective": "multi:softprob",
        "num_class": num_class,
        "eval_metric": "mlogloss",
        "learning_rate": 0.03,
        "max_depth": 7,
        "min_child_weight": 20,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "gamma": 1.0,
        "n_estimators": n_estimators,
        "tree_method": "hist",
        "grow_policy": "lossguide",
        "max_leaves": 63,
        "random_state": 42,
        "n_jobs": -1,
    }


def make_catboost_params(num_class=3, iterations=1000):
    return {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 3,
        "iterations": iterations,
        "random_seed": 42,
        "verbose": 0,
        "auto_class_weights": "Balanced",
        "early_stopping_rounds": 100,
    }


def train_lgb_oof(X_train, y_train, X_test, cat_cols, n_folds=5, seed=42):
    if not HAS_LGB:
        return None, None, None

    n_classes = len(np.unique(y_train))
    params = make_lgb_params(n_classes)
    n_est = params.pop("n_estimators", 2000)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_preds = np.zeros((len(X_train), n_classes))
    test_preds = np.zeros((len(X_test), n_classes))
    scores = []
    importances = pd.DataFrame(index=X_train.columns)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"  [LGB] Fold {fold + 1}/{n_folds}")
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
        dval = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols, free_raw_data=False)

        callbacks = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
        model = lgb.train(
            params, dtrain,
            num_boost_round=n_est,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        oof_preds[va_idx] = model.predict(X_va)
        test_preds += model.predict(X_test) / n_folds

        y_pred = np.argmax(oof_preds[va_idx], axis=1)
        fold_score = balanced_accuracy_score(y_va, y_pred)
        scores.append(fold_score)
        print(f"         Balanced Acc: {fold_score:.5f}")

        importances[f"fold_{fold}"] = model.feature_importance(importance_type="gain")

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"  [LGB] CV Mean: {mean_score:.5f} +/- {std_score:.5f}")

    imp = importances.mean(axis=1).sort_values(ascending=False)
    return oof_preds, test_preds, {"mean": mean_score, "std": std_score, "importance": imp}


def train_xgb_oof(X_train, y_train, X_test, n_folds=5, seed=42):
    if not HAS_XGB:
        return None, None, None

    n_classes = len(np.unique(y_train))
    params = make_xgb_params(n_classes)
    n_est = params.pop("n_estimators", 1000)

    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (n_classes * class_counts)
    sample_weights = class_weights[y_train]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_preds = np.zeros((len(X_train), n_classes))
    test_preds = np.zeros((len(X_test), n_classes))
    scores = []
    importances = pd.DataFrame(index=X_train.columns)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"  [XGB] Fold {fold + 1}/{n_folds}")
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        w_tr = sample_weights[tr_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
        dval = xgb.DMatrix(X_va, label=y_va)
        dtest = xgb.DMatrix(X_test)

        model = xgb.train(
            params, dtrain,
            num_boost_round=n_est,
            evals=[(dval, "val")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )

        oof_preds[va_idx] = model.predict(dval)
        test_preds += model.predict(dtest) / n_folds

        y_pred = np.argmax(oof_preds[va_idx], axis=1)
        fold_score = balanced_accuracy_score(y_va, y_pred)
        scores.append(fold_score)
        print(f"         Balanced Acc: {fold_score:.5f}")

        imp = model.get_score(importance_type="gain")
        importances[f"fold_{fold}"] = pd.Series(imp).reindex(X_train.columns).fillna(0)

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"  [XGB] CV Mean: {mean_score:.5f} +/- {std_score:.5f}")

    imp = importances.mean(axis=1).sort_values(ascending=False)
    return oof_preds, test_preds, {"mean": mean_score, "std": std_score, "importance": imp}


def train_cat_oof(X_train, y_train, X_test, cat_indices, n_folds=5, seed=42):
    if not HAS_CAT:
        return None, None, None

    n_classes = len(np.unique(y_train))
    params = make_catboost_params(n_classes)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_preds = np.zeros((len(X_train), n_classes))
    test_preds = np.zeros((len(X_test), n_classes))
    scores = []
    importances = pd.DataFrame(index=X_train.columns)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"  [CAT] Fold {fold + 1}/{n_folds}")
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        train_pool = Pool(X_tr, label=y_tr, cat_features=cat_indices)
        val_pool = Pool(X_va, label=y_va, cat_features=cat_indices)
        test_pool = Pool(X_test, cat_features=cat_indices)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, verbose=0)

        oof_preds[va_idx] = model.predict_proba(X_va)
        test_preds += model.predict_proba(X_test) / n_folds

        y_pred = np.argmax(oof_preds[va_idx], axis=1)
        fold_score = balanced_accuracy_score(y_va, y_pred)
        scores.append(fold_score)
        print(f"         Balanced Acc: {fold_score:.5f}")

        imp = model.get_feature_importance()
        importances[f"fold_{fold}"] = imp

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"  [CAT] CV Mean: {mean_score:.5f} +/- {std_score:.5f}")

    imp = importances.mean(axis=1).sort_values(ascending=False)
    return oof_preds, test_preds, {"mean": mean_score, "std": std_score, "importance": imp}


def ensemble_predictions(oof_list, test_list, weights=None):
    """
    Weighted average ensemble of OOF and test predictions.
    """
    valid_oof = [p for p in oof_list if p is not None]
    valid_test = [p for p in test_list if p is not None]

    if not valid_oof:
        raise ValueError("No valid model predictions to ensemble")

    if weights is None:
        weights = [1.0 / len(valid_oof)] * len(valid_oof)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    oof_ensemble = np.zeros_like(valid_oof[0])
    test_ensemble = np.zeros_like(valid_test[0])

    for oof, w in zip(valid_oof, weights):
        oof_ensemble += oof * w
    for test_p, w in zip(valid_test, weights):
        test_ensemble += test_p * w

    oof_y = np.argmax(oof_ensemble, axis=1)
    score = balanced_accuracy_score(None, None) if False else 0

    return oof_ensemble, test_ensemble


def evaluate_ensemble(oof_preds_list, y_true, weights=None, model_names=None):
    """
    Evaluate individual models and the ensemble.
    """
    valid = [(p, n) for p, n in zip(oof_preds_list, model_names) if p is not None]

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    individual_scores = []
    for preds, name in valid:
        y_pred = np.argmax(preds, axis=1)
        score = balanced_accuracy_score(y_true, y_pred)
        individual_scores.append(score)
        print(f"  {name}: {score:.5f}")

    if weights is None:
        weights = [1.0 / len(valid)] * len(valid)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    ensemble_oof = np.zeros_like(valid[0][0])
    for (preds, _), w in zip(valid, weights):
        ensemble_oof += preds * w

    y_ens = np.argmax(ensemble_oof, axis=1)
    ens_score = balanced_accuracy_score(y_true, y_ens)
    print(f"\n  Ensemble (weighted avg): {ens_score:.5f}")

    return ens_score, ensemble_oof
