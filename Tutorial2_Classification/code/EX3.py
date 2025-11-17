import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


df = pd.read_csv("experiences.csv")


X = df[["state"]]     # input: game state
y = df["action"]      # output: action to predict

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42,stratify = y)

#WE ARE GONNA USE GridSearchCV

param_grid = {      #Define params
    "criterion" : ["gini", "entropy"],
    "splitter": ["best", "random"],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "class_weight": [None, "balanced"]
}

grid_search = GridSearchCV(
    estimator = DecisionTreeClassifier(random_state = 42),
    param_grid = param_grid,
    scoring = "accuracy",
    cv = 5, #DEFAULT
    n_jobs = -1 #WE USE ALL CPU CORES
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Best params were found")
print(grid_search.best_params_)
print(f"Accuracy: {acc:.3f}")
print("Classification report")
print(classification_report(y_test, y_pred))


plt.figure(figsize=(20,10))
plot_tree(
    best_model,
    filled=False,
    feature_names=["state"],
    class_names=[str(i) for i in sorted(y.unique())]
)
plt.title("Best Decision Tree - GridSearchCV")
plt.savefig("best_decision_tree_ex3_grid.png", dpi=300, bbox_inches="tight")
plt.close()

print(" Best tree saved as best_decision_tree_ex3_grid.png")