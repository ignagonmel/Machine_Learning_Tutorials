

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


df = pd.read_csv("experiences.csv")


X = df[["state"]]     # input: game state
y = df["action"]      # output: action to predict


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train) # Train tree


y_pred = clf.predict(X_test)



acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.3f}\n")


print("Classification Report:")
print(classification_report(y_test, y_pred))



plt.figure(figsize=(20, 10))
plot_tree(
    clf,
    filled=True,
    feature_names=["state"],
    class_names=[str(i) for i in sorted(y.unique())]
)
plt.title("Decision Tree - Behavioral Cloning (Exercise 2)")
plt.savefig("decision_tree_ex2.png", dpi=300, bbox_inches="tight")
plt.close()

print("Tree image saved as decision_tree_ex2.png")
