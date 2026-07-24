"""
Preprocessing & Feature Engineering for Playground Series S6E7
Handles missing values, encoding, and feature creation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

TARGET = "health_condition"
ID_COL = "id"


def identify_features(train):
    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in [ID_COL, TARGET]]
    cat_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in [TARGET]]
    return num_cols, cat_cols


def create_health_features(df):
    """
    Domain-specific feature engineering for student health data.
    Creates abnormal flags, interaction features, and aggregates.
    """
    df = df.copy()

    if "sleep_duration" in df.columns:
        df["sleep_abnormal"] = (df["sleep_duration"] <= 6.5).astype(int)
        df["sleep_low"] = (df["sleep_duration"] < 5.5).astype(int)
        df["sleep_high"] = (df["sleep_duration"] > 9).astype(int)

    if "bmi" in df.columns:
        df["bmi_abnormal"] = (df["bmi"] >= 27).astype(int)
        df["bmi_obese"] = (df["bmi"] >= 30).astype(int)
        df["bmi_underweight"] = (df["bmi"] < 18.5).astype(int)

    if "step_count" in df.columns:
        df["steps_low"] = (df["step_count"] <= 4000).astype(int)
        df["steps_very_low"] = (df["step_count"] <= 2000).astype(int)
        df["steps_high"] = (df["step_count"] >= 10000).astype(int)

    if "exercise_duration" in df.columns:
        df["exercise_low"] = (df["exercise_duration"] <= 25).astype(int)
        df["exercise_very_low"] = (df["exercise_duration"] <= 10).astype(int)
        df["exercise_high"] = (df["exercise_duration"] >= 60).astype(int)

    abnormal_cols = [c for c in df.columns if c.endswith("_abnormal") or c.endswith("_low") or c.endswith("_high") or c.endswith("_obese") or c.endswith("_underweight") or c.endswith("_very_low")]
    if abnormal_cols:
        df["abnormal_count"] = df[abnormal_cols].sum(axis=1)
        df["healthy_count"] = len(abnormal_cols) - df["abnormal_count"]

    numeric_for_agg = []
    for c in ["sleep_duration", "bmi", "step_count", "exercise_duration", "heart_rate",
              "blood_pressure_systolic", "blood_pressure_diastolic", "cholesterol",
              "blood_glucose", "mental_health_score"]:
        if c in df.columns:
            numeric_for_agg.append(c)

    if len(numeric_for_agg) >= 2:
        df["num_mean"] = df[numeric_for_agg].mean(axis=1)
        df["num_std"] = df[numeric_for_agg].std(axis=1)
        df["num_min"] = df[numeric_for_agg].min(axis=1)
        df["num_max"] = df[numeric_for_agg].max(axis=1)
        df["num_range"] = df["num_max"] - df["num_min"]
        df["num_median"] = df[numeric_for_agg].median(axis=1)
        df["num_skew"] = df[numeric_for_agg].skew(axis=1)

    if "sleep_duration" in df.columns and "exercise_duration" in df.columns:
        df["sleep_exercise_ratio"] = df["sleep_duration"] / (df["exercise_duration"] + 1)
        df["sleep_exercise_diff"] = df["sleep_duration"] - df["exercise_duration"]

    if "bmi" in df.columns and "step_count" in df.columns:
        df["bmi_steps_interaction"] = df["bmi"] * df["step_count"]
        df["bmi_steps_ratio"] = df["bmi"] / (df["step_count"] + 1)

    if "heart_rate" in df.columns and "exercise_duration" in df.columns:
        df["hr_exercise_interaction"] = df["heart_rate"] * df["exercise_duration"]

    if "blood_pressure_systolic" in df.columns and "blood_pressure_diastolic" in df.columns:
        df["bp_ratio"] = df["blood_pressure_systolic"] / (df["blood_pressure_diastolic"] + 1)
        df["pulse_pressure"] = df["blood_pressure_systolic"] - df["blood_pressure_diastolic"]

    if "cholesterol" in df.columns and "blood_glucose" in df.columns:
        df["chol_gluc_interaction"] = df["cholesterol"] * df["blood_glucose"]

    if "mental_health_score" in df.columns and "sleep_duration" in df.columns:
        df["mental_sleep_interaction"] = df["mental_health_score"] * df["sleep_duration"]

    return df


def target_encode_cv(train, col, target, n_folds=5, seed=42):
    """
    Target encode a categorical column using CV to prevent leakage.
    Returns encoded column for train and a mapping dict for test.
    """
    from sklearn.model_selection import StratifiedKFold

    le = LabelEncoder()
    y = le.fit_transform(train[target])
    encoded = pd.Series(index=train.index, dtype=float)
    global_mean = y.mean()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    global_means = {}

    for train_idx, val_idx in skf.split(train, y):
        tr = train.iloc[train_idx]
        va = train.iloc[val_idx]
        tr_y = y[train_idx]

        means = {}
        for val in tr[col].unique():
            mask = tr[col] == val
            if mask.sum() > 0:
                means[val] = tr_y[mask].mean()
            else:
                means[val] = global_mean

        encoded.iloc[val_idx] = va[col].map(means).fillna(global_mean).values

        for val, m in means.items():
            if val not in global_means:
                global_means[val] = []
            global_means[val].append(m)

    final_means = {k: np.mean(v) for k, v in global_means.items()}
    return encoded, final_means


def build_preprocessor(train, test, target=TARGET):
    """
    Full preprocessing pipeline:
    1. Feature engineering
    2. Missing value imputation
    3. Categorical encoding (label encoding for tree models)
    4. Target encoding with CV
    """
    num_cols, cat_cols = identify_features(train)

    train = create_health_features(train)
    test = create_health_features(test)

    all_num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    all_num_cols = [c for c in all_num_cols if c not in [ID_COL, TARGET]]
    all_cat_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
    all_cat_cols = [c for c in all_cat_cols if c not in [TARGET]]

    for col in all_num_cols:
        median_val = train[col].median()
        train[col] = train[col].fillna(median_val)
        test[col] = test[col].fillna(median_val)

    for col in all_cat_cols:
        train[col] = train[col].fillna("missing").astype(str)
        test[col] = test[col].fillna("missing").astype(str)

    le_dict = {}
    for col in all_cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        le_dict[col] = le

    target_encodings = {}
    for col in all_cat_cols:
        encoded_col, mapping = target_encode_cv(train, col, target)
        train[f"{col}_te"] = encoded_col
        test[f"{col}_te"] = test[col].map(mapping)
        test[f"{col}_te"] = test[f"{col}_te"].fillna(train[target].map(
            {v: i for i, v in enumerate(train[target].unique())}
        ).mean() if False else 0)
        target_encodings[col] = mapping

    test_global_mean = 0.5
    for col in all_cat_cols:
        te_col = f"{col}_te"
        if test[te_col].isnull().any():
            test[te_col] = test[te_col].fillna(test_global_mean)

    final_num = [c for c in train.columns if c not in [ID_COL, TARGET] and c not in all_cat_cols]
    final_cat = all_cat_cols

    print(f"\nFinal feature set: {len(final_num)} numerical + {len(final_cat)} categorical = {len(final_num) + len(final_cat)} total")

    return train, test, final_num, final_cat, le_dict, target_encodings


def prepare_model_data(train, test, num_cols, cat_cols, target=TARGET):
    """
    Prepare separate feature matrices for different model types:
    - LightGBM/XGBoost: all features as-is (label-encoded categoricals)
    - CatBoost: separate cat feature indices
    """
    feature_cols = num_cols + cat_cols
    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    le_target = LabelEncoder()
    y_train = le_target.fit_transform(train[target])

    cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

    return X_train, X_test, y_train, feature_cols, cat_indices, le_target
