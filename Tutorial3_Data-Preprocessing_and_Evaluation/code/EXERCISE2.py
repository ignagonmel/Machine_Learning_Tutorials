import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 1. Load datasets and create Train/Validation/Test split

DATA_PATH = "DATA_T3"

full_datasets = {
    "original": pd.read_csv(os.path.join(DATA_PATH, "dataset_original.csv")),
    "additional": pd.read_csv(os.path.join(DATA_PATH, "dataset_additional.csv")),
    "mi": pd.read_csv(os.path.join(DATA_PATH, "dataset_mi.csv")),
    "rfe": pd.read_csv(os.path.join(DATA_PATH, "dataset_rfe.csv"))
}

datasets_train_val = {}
datasets_test = {}

for name, df in full_datasets.items():

    X_full = df.drop(columns=["action"])
    y_full = df["action"]

    # 80% Train/Val, 20% Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full # 'stratify' ayuda a mantener la proporción de clases
    )

    datasets_train_val[name] = pd.concat([X_train_val, y_train_val], axis=1)
    datasets_test[name] = pd.concat([X_test, y_test], axis=1)

print(f"✅ Loaded {len(full_datasets)} datasets from {DATA_PATH} and split them into Train/Validation (80%) and Test (20%).\n")


# 2. QUESTION 1: Base model + Cross-validation

print("================= QUESTION 1 =================")

base_params = {
    "criterion": "entropy",
    "max_depth": 8,
    "min_samples_split": 10,
    "random_state": 42
}

q1_results = []

for name, df in datasets_train_val.items():
    print(f"\n>>> Cross-validating base model on {name.upper()} TRAIN/VAL dataset (80% of data)")

    X = df.drop(columns=["action"])
    y = df["action"]
    n_samples = len(y)

    clf = DecisionTreeClassifier(**base_params)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

    acc = scores.mean()
    correct = int(acc * n_samples)
    incorrect = n_samples - correct

    clf.fit(X, y)
    y_pred = clf.predict(X)
    prec = precision_score(y, y_pred, average="macro", zero_division=0)
    rec = recall_score(y, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y, y_pred, average="macro", zero_division=0)

    q1_results.append({
        "Dataset": name,
        "Correct": correct,
        "Incorrect": incorrect,
        "Accuracy (CV)": acc, 
        "Precision (Train)": prec,
        "Recall (Train)": rec,
        "F1 (Train)": f1
    })

q1_df = pd.DataFrame(q1_results)
print("\n===== Q1: CROSS-VALIDATION RESULTS (on 80% TRAIN/VAL) =====")
print(q1_df.round(4))


# 3. QUESTION 2: GridSearchCV + top 3 per dataset

print("\n================= QUESTION 2 =================")

param_grid = {
    "criterion": ["gini", "entropy"],
    "splitter": ["best", "random"],
    "max_depth": [3, 5, 8, 10, None],
    "min_samples_split": [2, 5, 10],
    "class_weight": [None, "balanced"]
}

all_top3 = []

for name, df in datasets_train_val.items():
    print(f"\n>>> Performing GridSearchCV on {name.upper()} TRAIN/VAL dataset (80% of data)")

    X = df.drop(columns=["action"])
    y = df["action"]

    grid = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X, y)

    best_params = grid.best_params_
    best_acc = grid.best_score_
    print(f"Best CV Accuracy: {best_acc:.4f}")
    print(f"Best Parameters: {best_params}")

    results_df = pd.DataFrame(grid.cv_results_)
    top3 = results_df.sort_values(by="mean_test_score", ascending=False).head(3)
    top3["Dataset"] = name
    all_top3.append(top3[["Dataset", "params", "mean_test_score"]])

q2_all_top3 = pd.concat(all_top3, ignore_index=True)

top3_global = q2_all_top3.sort_values(by="mean_test_score", ascending=False).head(3)

print("\n===== Q2: TOP 3 MODELS PER DATASET (by CV Accuracy) =====")
print(q2_all_top3)
print("\n===== Q2: TOP 3 MODELS ACROSS ALL DATASETS (by CV Accuracy) =====")
print(top3_global)


# 4. QUESTION 3: Retrain top 3 models and evaluate on dedicated TEST data (20%)

print("\n================= QUESTION 3 =================")

q3_results = []

for idx, row in top3_global.iterrows():
    name = row["Dataset"]
    params = row["params"]


    df_train_val = datasets_train_val[name]
    X_train_final = df_train_val.drop(columns=["action"])
    y_train_final = df_train_val["action"]


    df_test = datasets_test[name]
    X_test_final = df_test.drop(columns=["action"])
    y_test_final = df_test["action"]


    clf = DecisionTreeClassifier(**params, random_state=42)
    print(f"\nTraining Model #{idx+1} on {name.upper()} (80% Train/Val data) and testing on 20% dedicated Test data...")
    clf.fit(X_train_final, y_train_final)
    

    y_pred = clf.predict(X_test_final)
    n_test_samples = len(y_test_final)


    acc = accuracy_score(y_test_final, y_pred)
    prec = precision_score(y_test_final, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test_final, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test_final, y_pred, average="macro", zero_division=0)

    correct = int(acc * n_test_samples)
    incorrect = n_test_samples - correct

    q3_results.append({
        "Model Rank": idx+1,
        "Dataset": name,
        "Parameters": params,
        "Correct": correct,
        "Incorrect": incorrect,
        "Accuracy (TEST)": acc,
        "Precision (TEST)": prec,
        "Recall (TEST)": rec,
        "F1 (TEST)": f1
    })

q3_df = pd.DataFrame(q3_results)
print("\n===== Q3: FINAL TEST RESULTS (on 20% DEDICATED TEST SET) =====")
print(q3_df.round(4))


# 5. Save all results

os.makedirs("RESULTS_T3", exist_ok=True)
q1_df.to_csv("RESULTS_T3/q1_crossval_results.csv", index=False)
q2_all_top3.to_csv("RESULTS_T3/q2_top_models.csv", index=False)
top3_global.to_csv("RESULTS_T3/q2_top3_global.csv", index=False)
q3_df.to_csv("RESULTS_T3/q3_final_results.csv", index=False)

print("\n All results saved in 'RESULTS_T3' folder.")