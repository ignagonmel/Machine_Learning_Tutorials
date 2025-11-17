import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Load Data
train_data = pd.read_csv("Datasets/training_data.csv")
test_data = pd.read_csv("Datasets/test_data.csv")

X_train = train_data.drop(columns=["Action"])
y_train = train_data["Action"]
X_test = test_data.drop(columns=["Action"])
y_test = test_data["Action"]


lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.3f}")


fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

p = ax.scatter(X_train['x'], X_train['y'], X_train['Angular_velocity'], c=y_train, cmap='viridis', s=50)
fig.colorbar(p, ax=ax, label='Action')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Angular_velocity')
plt.title('State Space:')
plt.show()