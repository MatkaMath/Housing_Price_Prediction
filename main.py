import os
from preprocessing import load_data, preprocess_data, split_and_scale_data
from train import get_models, tune_model, train_models, cross_validate_model
from evaluation import evaluate_models, generate_plots

os.makedirs("results", exist_ok=True)

data = load_data()
X, y = preprocess_data(data)
X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)

models = get_models()
rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
gb_param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.05, 0.1, 0.2]}

models["Random Forest"] = tune_model(models["Random Forest"], rf_param_grid, X_train, y_train)
models["Gradient Boosting"] = tune_model(models["Gradient Boosting"], gb_param_grid, X_train, y_train)

trained_models = train_models(models, X_train, y_train)

# Cross-validation scores
rf_cv_score = cross_validate_model(trained_models["Random Forest"], X_train, y_train)
gb_cv_score = cross_validate_model(trained_models["Gradient Boosting"], X_train, y_train)
print(f"Random Forest Cross-validated R²: {rf_cv_score:.4f}")
print(f"Gradient Boosting Cross-validated R²: {gb_cv_score:.4f}")

evaluate_models(trained_models, X_test, y_test)
generate_plots(trained_models, X_test, y_test, X.columns)
