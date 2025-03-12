from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import os

os.makedirs("models", exist_ok=True)

def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

def tune_model(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def cross_validate_model(model, X_train, y_train):
    return cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()

def train_models(models, X_train, y_train):
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f"models/{name.replace(' ', '_')}_model.pkl")
    return models