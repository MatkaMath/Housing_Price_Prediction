import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

os.makedirs("results", exist_ok=True)

def evaluate_models(models, X_test, y_test):
    metrics_list = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics_list.append({
            'Model': name, 
            'MAE': mean_absolute_error(y_test, y_pred), 
            'MSE': mean_squared_error(y_test, y_pred), 
            'R-squared': r2_score(y_test, y_pred)
        })
    
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv("results/metrics.csv", index=False)
    print(metrics_df)

def generate_plots(models, X_test, y_test, feature_names):
    plt.figure(figsize=(12, 8))
    sns.heatmap(pd.DataFrame(X_test).corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.savefig("results/correlation_heatmap.png")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, model in models.items():
        ax.scatter(y_test, model.predict(X_test), label=name, alpha=0.6)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    ax.set_xlabel('Actual Prices')
    ax.set_ylabel('Predicted Prices')
    ax.set_title('Actual vs Predicted Housing Prices')
    ax.legend()
    plt.savefig("results/actual_vs_predicted.png")
    plt.close()
    
    # Feature Importance Plot (Random Forest)
    if "Random Forest" in models:
        feature_importances = pd.Series(models["Random Forest"].feature_importances_, index=feature_names)
        plt.figure(figsize=(10, 6))
        feature_importances.nlargest(10).plot(kind='barh')
        plt.title('Top 10 Important Features (Random Forest)')
        plt.xlabel('Feature Importance')
        plt.savefig("results/feature_importance.png")
        plt.close()
