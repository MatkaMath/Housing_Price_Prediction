# 🏡 Boston Housing Price Prediction

## 📌 Overview
This project builds a **housing price prediction model** using the **Boston Housing Dataset**.  
It employs machine learning techniques to **predict housing prices** based on key features such as crime rate, number of rooms, and property tax.

### 🔹 Key Steps:
- **Data Preprocessing**: Handling outliers using **Winsorization**, scaling numerical features.
- **Feature Engineering**: Correlation analysis and feature selection.
- **Machine Learning Models**: Training **Linear Regression, Random Forest, and Gradient Boosting** models.
- **Hyperparameter Tuning**: Using **GridSearchCV** to optimize model performance.
- **Model Evaluation**: Measuring **MAE, MSE, and R² scores**.
- **Data Visualization**: Generating plots for insights into model predictions and feature importance.

---

## 📊 Dataset
The dataset is sourced from **[Boston Housing Data](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv)**.

- **Samples**: 506  
- **Features**: 13 numerical features  
- **Target Variable (`MEDV`)**: Median house price in $1000s  

---

## 📈 Visualizations
The project generates several key visualizations:

1. **Feature Correlation Heatmap** (`correlation_heatmap.png`)  
   - Displays relationships between different features in the dataset.

2. **Actual vs Predicted Prices Scatter Plot** (`actual_vs_predicted.png`)  
   - Shows how well the models' predictions align with actual housing prices.

3. **Top 10 Important Features (Random Forest)** (`feature_importance.png`)  
   - Highlights the most influential features for predicting housing prices.

---

## 🚀 Running the Project
### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt

### 2️⃣ Run the Pipeline
```bash
python main.py
```

### 3️⃣ Check Outputs
- Trained models will be saved in the `models/` directory.
- Evaluation results and visualizations will be stored in the `results/` directory.

---

## 📂 Project Structure
```
📦 Boston_Housing_Prediction
├── 📄 main.py             # Runs preprocessing, training, and evaluation
├── 📄 preprocessing.py    # Data loading and preprocessing
├── 📄 train.py            # Model training and hyperparameter tuning
├── 📄 evaluation.py       # Model evaluation and visualization
├── 📄 requirements.txt    # Dependencies
├── 📂 models/             # Saved trained models
└── 📂 results/            # Evaluation metrics & visualizations
```

---

## 📊 Results & Evaluation
After training, the model's performance is evaluated using:
- **Mean Absolute Error (MAE):** Measures prediction error in absolute terms.
- **Mean Squared Error (MSE):** Penalizes larger errors more heavily.
- **R² Score:** Indicates how well the model explains variance in house prices.
- **Cross-Validation Scores:** Ensures robustness across different data splits.

---

## 🔗 References
- [Boston Housing Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)

---

🚀 Ready to predict housing prices? Run the pipeline and explore the results!
