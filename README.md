# 🌾 Fertilizer Recommendation System using Machine Learning

This project implements a supervised machine learning model to recommend the **top 3 most suitable fertilizers** based on soil and crop characteristics. The model is trained on real-world agricultural data and predicts fertilizers using a multiclass classification approach.

---

## 📌 Problem Statement

In agriculture, choosing the right fertilizer is critical for maximizing crop yield and soil health. This project addresses the challenge of recommending fertilizers by analyzing:

- Soil Type
- Crop Type
- Nutrient Levels (Nitrogen, Phosphorus, Potassium)

The goal is to predict the **top 3 recommended fertilizers** for a given set of conditions.

---

## 🔍 Dataset

- `train.csv`: Contains labeled data with features like soil type, crop type, and fertilizer name.
- `test.csv`: Contains similar features without the target label, used for prediction.

Each record includes:
- `Soil Type`: Categorical
- `Crop Type`: Categorical
- `N`, `P`, `K`: Nutrient levels
- `Fertilizer Name`: Target variable (for training)

---

## 🧠 Approach

### 1. **Preprocessing**
- Dropped `id` column (not useful for training)
- Applied `LabelEncoder` to encode categorical variables: `Soil Type`, `Crop Type`, and `Fertilizer Name`
- Split features (`X`) and target (`y`)

### 2. **Model Training**
- Used `LightGBMClassifier` for its speed and accuracy in multiclass problems
- Set parameters: `n_estimators=200`, `learning_rate=0.1`, `random_state=42`

### 3. **Prediction**
- Predicted **class probabilities** using `predict_proba`
- Extracted **top 3 most probable fertilizers** for each test sample
- Decoded predictions back to fertilizer names

### 4. **Output**
- Final submission contains `id` and top-3 predicted fertilizers in space-delimited format

---

## 📊 Exploratory Data Analysis

- Visualized class distribution of the target variable (`Fertilizer Name`)
- Observed some imbalance in class frequencies, which could affect performance

---

## 🚀 Results

- Successfully trained and applied model to generate top-3 recommendations
- Output format is compatible with common Kaggle-style submission formats

> ✅ Bonus: Used probability-based ranking for top-3 predictions instead of just predicting a single class — increasing the real-world usability.

---

## 💻 Tools & Technologies

- Python (Jupyter Notebook)
- `pandas`, `numpy` – Data manipulation
- `scikit-learn` – Preprocessing, encoding
- `lightgbm` – Model training
- `matplotlib` – Visualization




