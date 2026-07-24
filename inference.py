"""
Inference Module — loads trained models and predicts on new data.
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from preprocessing import create_health_features, identify_features

TARGET = "health_condition"
ID_COL = "id"


def save_models(models_dict, path="models.pkl"):
    with open(path, "wb") as f:
        pickle.dump(models_dict, f)


def load_models(path="models.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def train_and_save_models(train_path="train.csv", model_dir="models"):
    """
    Train all models on full training data and save for inference.
    """
    os.makedirs(model_dir, exist_ok=True)

    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder

    train = pd.read_csv(train_path)

    num_cols, cat_cols = identify_features(train)
    train = create_health_features(train)

    all_num = [c for c in train.select_dtypes(include=[np.number]).columns if c not in [ID_COL, TARGET]]
    all_cat = [c for c in train.select_dtypes(include=["object", "category"]).columns if c not in [TARGET]]

    for col in all_num:
        train[col] = train[col].fillna(train[col].median())
    for col in all_cat:
        train[col] = train[col].fillna("missing").astype(str)

    le_dict = {}
    for col in all_cat:
        le = LabelEncoder()
        le.fit(train[col])
        train[col] = le.transform(train[col])
        le_dict[col] = le

    from preprocessing import target_encode_cv
    te_maps = {}
    for col in all_cat:
        _, mapping = target_encode_cv(train, col, TARGET)
        train[f"{col}_te"] = train[col].map(mapping).fillna(0)
        te_maps[col] = mapping

    feature_cols = all_num + all_cat + [f"{c}_te" for c in all_cat]
    X = train[feature_cols]
    le_target = LabelEncoder()
    y = le_target.fit_transform(train[TARGET])

    cat_indices = list(range(len(all_num), len(all_num) + len(all_cat)))

    print("Training LightGBM...")
    lgb_params = {
        "objective": "multiclass",
        "num_class": 3,
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
        "n_estimators": 1500,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
        "class_weight": "balanced",
    }
    n_est = lgb_params.pop("n_estimators")
    dtrain = lgb.Dataset(X, label=y, categorical_feature=all_cat, free_raw_data=False)
    lgb_model = lgb.train(lgb_params, dtrain, num_boost_round=n_est)

    print("Training XGBoost...")
    xgb_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "learning_rate": 0.03,
        "max_depth": 7,
        "min_child_weight": 20,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "gamma": 1.0,
        "n_estimators": 1000,
        "tree_method": "hist",
        "grow_policy": "lossguide",
        "max_leaves": 63,
        "random_state": 42,
        "n_jobs": -1,
    }
    n_est_xgb = xgb_params.pop("n_estimators")
    class_counts = np.bincount(y)
    class_weights = len(y) / (3 * class_counts)
    weights = class_weights[y]
    dtrain_xgb = xgb.DMatrix(X, label=y, weight=weights)
    xgb_model = xgb.train(xgb_params, dtrain_xgb, num_boost_round=n_est_xgb)

    model_data = {
        "lgb_model": lgb_model,
        "xgb_model": xgb_model,
        "le_dict": le_dict,
        "te_maps": te_maps,
        "le_target": le_target,
        "feature_cols": feature_cols,
        "all_num": all_num,
        "all_cat": all_cat,
        "cat_indices": cat_indices,
    }
    save_models(model_data, os.path.join(model_dir, "trained_models.pkl"))
    print(f"Models saved to {model_dir}/trained_models.pkl")
    return model_data


def preprocess_input(input_df, model_data):
    """
    Apply same preprocessing to user input as training data.
    """
    df = input_df.copy()
    all_num = model_data["all_num"]
    all_cat = model_data["all_cat"]
    le_dict = model_data["le_dict"]
    te_maps = model_data["te_maps"]

    df = create_health_features(df)

    for col in all_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isnull().any():
                df[col] = df[col].fillna(0)

    for col in all_cat:
        if col in df.columns:
            df[col] = df[col].fillna("missing").astype(str)
            if col in le_dict:
                le = le_dict[col]
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])

    for col in all_cat:
        te_col = f"{col}_te"
        if col in te_maps and col in df.columns:
            mapping = te_maps[col]
            inv_mapping = {v: k for k, v in mapping.items()} if isinstance(list(mapping.keys())[0], (int, float)) else mapping
            df[te_col] = df[col].map(lambda x: mapping.get(x, 0))
        else:
            df[te_col] = 0

    for col in model_data["feature_cols"]:
        if col not in df.columns:
            df[col] = 0

    return df[model_data["feature_cols"]]


def predict(input_df, model_data):
    """
    Predict health condition from preprocessed input.
    Returns: list of predicted labels, array of probabilities.
    """
    import xgboost as xgb

    X = preprocess_input(input_df, model_data)

    lgb_probs = model_data["lgb_model"].predict(X)
    dmatrix = xgb.DMatrix(X)
    xgb_probs = model_data["xgb_model"].predict(dmatrix)

    ensemble_probs = 0.5 * lgb_probs + 0.5 * xgb_probs
    y_pred = np.argmax(ensemble_probs, axis=1)
    labels = model_data["le_target"].inverse_transform(y_pred)

    return labels, ensemble_probs, model_data["le_target"].classes_
