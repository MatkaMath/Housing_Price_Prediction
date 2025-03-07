import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv", header=None, names=column_names)

print(data.head())
print(data.isnull().sum())

# Exploratory Data Analysis
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Exploratory Data Analysis of Boston Housing Dataset', fontsize=16)

sns.histplot(data['MEDV'], bins=30, kde=True, ax=axs[0, 0])
axs[0, 0].set_title('Distribution of Housing Prices (MEDV)')
axs[0, 0].set_xlabel('MEDV')
axs[0, 0].set_ylabel('Frequency')

sns.scatterplot(x='RM', y='MEDV', data=data, ax=axs[0, 1])
axs[0, 1].set_title('Relationship between Number of Rooms and Housing Prices')
axs[0, 1].set_xlabel('Number of Rooms (RM)')
axs[0, 1].set_ylabel('Housing Prices (MEDV)')

correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=axs[1, 0])
axs[1, 0].set_title('Correlation Matrix')

sns.violinplot(x='RAD', y='MEDV', data=data)
axs[1, 1].set_title('Violin Plot of Housing Prices by Accessibility to Radial Highways')
axs[1, 1].set_xlabel('Accessibility to Radial Highways (RAD)')
axs[1, 1].set_ylabel('Housing Prices (MEDV)')

plt.tight_layout()
plt.show()

# Capping outliers
lower_cap = data.quantile(0.05)
upper_cap = data.quantile(0.95)

data_capped = data.clip(lower=lower_cap, upper=upper_cap, axis=1)

# Prepare data for modeling
X_capped = data_capped.drop('MEDV', axis=1)
y_capped = data_capped['MEDV']

X_train_capped, X_test_capped, y_train_capped, y_test_capped = train_test_split(X_capped, y_capped, test_size=0.2, random_state=42)

# Initialize a DataFrame to store metrics
metrics_df = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'R-squared'])

# Linear regression model
model_capped = LinearRegression()
model_capped.fit(X_train_capped, y_train_capped)
y_pred_capped = model_capped.predict(X_test_capped)

mae_capped = mean_absolute_error(y_test_capped, y_pred_capped)
mse_capped = mean_squared_error(y_test_capped, y_pred_capped)
r2_capped = r2_score(y_test_capped, y_pred_capped)

linear_metrics = pd.DataFrame({'Model': ['Linear Regression'], 'MAE': [mae_capped], 'MSE': [mse_capped], 'R-squared': [r2_capped]})

metrics_df = pd.concat([metrics_df, linear_metrics], ignore_index=True)

# Regularization: Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_capped, y_train_capped)
y_pred_lasso = lasso.predict(X_test_capped)

mae_lasso = mean_absolute_error(y_test_capped, y_pred_lasso)
mse_lasso = mean_squared_error(y_test_capped, y_pred_lasso)
r2_lasso = r2_score(y_test_capped, y_pred_lasso)

lasso_metrics = pd.DataFrame({'Model': ['Lasso Regression'], 'MAE': [mae_lasso], 'MSE': [mse_lasso], 'R-squared': [r2_lasso]})

metrics_df = pd.concat([metrics_df, lasso_metrics], ignore_index=True)

# Regularization: Ridge Regression
ridge = Ridge(alpha=0.1)
ridge.fit(X_train_capped, y_train_capped)
y_pred_ridge = ridge.predict(X_test_capped)

mae_ridge = mean_absolute_error(y_test_capped, y_pred_ridge)
mse_ridge = mean_squared_error(y_test_capped, y_pred_ridge)
r2_ridge = r2_score(y_test_capped, y_pred_ridge)

ridge_metrics = pd.DataFrame({'Model': ['Ridge Regression'], 'MAE': [mae_ridge], 'MSE': [mse_ridge], 'R-squared': [r2_ridge]})

metrics_df = pd.concat([metrics_df, ridge_metrics], ignore_index=True)

print(metrics_df)

# Vizualize results
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Actual vs Predicted Housing Prices
axs[0].scatter(y_test_capped, y_pred_capped, label='Linear Regression', color='blue', alpha=0.6)
#axs[0].scatter(y_test_capped, y_pred_lasso, label='Lasso Regression', color='orange', alpha=0.6)
#axs[0].scatter(y_test_capped, y_pred_ridge, label='Ridge Regression', color='green', alpha=0.6)
axs[0].plot([min(y_test_capped), max(y_test_capped)], [min(y_test_capped), max(y_test_capped)], color='red', linestyle='--')
axs[0].set_xlabel('Actual Prices')
axs[0].set_ylabel('Predicted Prices')
axs[0].set_title('Actual vs Predicted Housing Prices (Capped Data)')
axs[0].legend()

# Combine coefficients for comparison
coefficients = pd.DataFrame(model_capped.coef_, X_capped.columns, columns=['Linear Coefficients'])
coefficients_lasso = pd.DataFrame(lasso.coef_, X_capped.columns, columns=['Lasso Coefficient'])
coefficients_ridge = pd.DataFrame(ridge.coef_, X_capped.columns, columns=['Ridge Coefficient'])

coefficients_combined = pd.concat([coefficients, coefficients_lasso, coefficients_ridge], axis=1)
print(coefficients_combined)

# Plot 2: Coefficients of Linear, Lasso, and Ridge Regression
coefficients_combined.plot(kind='bar', ax=axs[1], figsize=(12, 6))
axs[1].axhline(0, color='black', lw=1, ls='--')  # Draw horizontal line at y=0
axs[1].set_title('Coefficients of Linear, Lasso, and Ridge Regression')
axs[1].set_ylabel('Coefficient Value')
axs[1].set_xlabel('Features')

plt.tight_layout()
plt.show()