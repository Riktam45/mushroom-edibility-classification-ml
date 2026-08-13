#  Mushroom Edibility Classification

## Machine Learning Classification Project

###  Project Overview

Mushroom foraging can be dangerous — some species are edible while others are poisonous. There is **no simple visual rule** that works for every mushroom.

This project builds and compares multiple Machine Learning models that classify mushrooms as **Edible** or **Poisonous** based on their physical characteristics (cap shape, odor, gill size, spore print color, habitat, etc.).

The goal is to create a clear, educational, and high-performing classification pipeline using classical Machine Learning algorithms.

**Target Labels:**
- `0` → Edible
- `1` → Poisonous

---

##  Objectives

1. Load and understand the UCI Mushroom dataset
2. Perform data validation and handle missing values
3. Conduct Exploratory Data Analysis (EDA)
4. Analyze class distribution (Edible vs Poisonous)
5. Explore important categorical features (odor, spore-print-color, gill-size, etc.)
6. Measure feature–target association using Cramér’s V
7. Encode categorical features with One-Hot Encoding
8. Split data into training and testing sets (stratified)
9. Build preprocessing + model pipelines
10. Train and compare six Machine Learning models
11. Evaluate models using Accuracy, Precision, Recall, and F1 Score
12. Perform 5-fold Stratified Cross-Validation
13. Visualize Confusion Matrix and Decision Tree
14. Analyze feature importance (Random Forest)
15. Save all trained models for future use

---

##  Dataset

**Source:** UCI Machine Learning Repository – Mushroom Data Set  
**Original Source:** The Audubon Society Field Guide to North American Mushrooms (1981)

### Dataset Files

```text
data/
├── agaricus-lepiota.csv
└── agaricus-lepiota.names
```

### Dataset Summary

| Property              | Value                          |
|-----------------------|--------------------------------|
| Total Instances       | 8,124                          |
| Number of Features    | 22 (all categorical)           |
| Target Variable       | class (edible / poisonous)     |
| Edible Mushrooms      | 4,208 (51.8%)                  |
| Poisonous Mushrooms   | 3,916 (48.2%)                  |
| Missing Values        | 2,480 (only in `stalk-root`)   |

### Target Labels

| Original | Mapped Label | Numeric |
|----------|--------------|---------|
| `e`      | Edible       | 0       |
| `p`      | Poisonous    | 1       |

### Key Features

| Feature                  | Description                          |
|--------------------------|--------------------------------------|
| cap-shape                | bell, conical, convex, flat, etc.    |
| cap-surface              | fibrous, grooves, scaly, smooth      |
| cap-color                | brown, yellow, white, red, etc.      |
| bruises                  | bruises / no bruises                 |
| odor                     | almond, anise, foul, none, etc.      |
| gill-size                | broad / narrow                       |
| gill-color               | black, brown, buff, etc.             |
| stalk-root               | bulbous, club, equal, missing (`?`)  |
| spore-print-color        | black, brown, green, white, etc.     |
| population               | abundant, clustered, solitary, etc.  |
| habitat                  | grasses, leaves, woods, urban, etc.  |

> **Note:** The classic rule “odor = almond/anise/none → likely edible” is very strong, but Machine Learning models learn combinations of many features automatically.

---

##  Problem Statement

Given the physical characteristics of a mushroom, predict whether it is **Edible** or **Poisonous**.

- **Input (X):** 22 categorical features describing the mushroom
- **Output (y):** Binary class (Edible / Poisonous)

All features are nominal, so the pipeline uses:
1. Missing value imputation (`most_frequent`)
2. One-Hot Encoding
3. Classification models

---

##  Project Workflow

```text
┌─────────────────────────────────┐
│     Mushroom Dataset            │
│  agaricus-lepiota.csv           │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│     Data Loading & Cleaning     │
│  • Assign column names          │
│  • Handle "?" missing values    │
│  • Remove duplicates            │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   Exploratory Data Analysis     │
│  • Class distribution           │
│  • Feature vs Class plots       │
│  • Cramér’s V association       │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Feature & Target Preparation   │
│  X → 22 categorical features    │
│  y → Edible (0) / Poisonous (1) │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│      Train / Test Split         │
│     80% Train  |  20% Test      │
│         (stratified)            │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   Preprocessing Pipeline        │
│  • SimpleImputer                │
│  • OneHotEncoder                │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│     Train 6 ML Models           │
│  Logistic Regression            │
│  Linear Regression              │
│  SVM                            │
│  KNN                            │
│  Decision Tree                  │
│  Random Forest                  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│     Model Evaluation            │
│  Accuracy • Precision • Recall  │
│  F1 Score • Confusion Matrix    │
│  5-Fold Cross-Validation        │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│     Save Trained Models         │
│  *.pkl files in /models         │
└─────────────────────────────────┘
```

---

##  Methodology

### 1. Data Loading & Cleaning
- Loaded the CSV (no header) and assigned the 23 official column names
- Replaced `?` with `NaN`
- Imputed missing values in `stalk-root` with the most frequent category
- Checked and removed any duplicate rows

### 2. Exploratory Data Analysis
- Visualized class balance (almost perfectly balanced)
- Created count plots for important features vs class:
  - Odor
  - Cap color
  - Gill size
  - Spore print color
  - Population
- Calculated **Cramér’s V** to rank feature association with the target

### 3. Preprocessing Pipeline
```python
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
```

### 4. Models Trained
| Model                  | Type                    |
|------------------------|-------------------------|
| Logistic Regression    | Linear Classifier       |
| Linear Regression      | Used with thresholding  |
| Support Vector Machine | Kernel SVM              |
| K-Nearest Neighbors    | Instance-based          |
| Decision Tree          | Tree-based              |
| Random Forest          | Ensemble of trees       |

All models are wrapped in a full `Pipeline` (preprocessor + classifier).

### 5. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- 5-Fold Stratified Cross-Validation

---

##  Model Performance

### Test Set Results

| Model                | Accuracy  | Precision | Recall   | F1 Score  |
|----------------------|-----------|-----------|----------|-----------|
| Linear Regression    | 1.0000    | 1.0000    | 1.0000   | 1.0000    |
| SVM                  | 1.0000    | 1.0000    | 1.0000   | 1.0000    |
| KNN                  | 1.0000    | 1.0000    | 1.0000   | 1.0000    |
| Decision Tree        | 1.0000    | 1.0000    | 1.0000   | 1.0000    |
| Random Forest        | 1.0000    | 1.0000    | 1.0000   | 1.0000    |
| Logistic Regression  | 0.9988    | 1.0000    | 0.9974   | 0.9987    |

### 5-Fold Cross-Validation

| Model                | CV Accuracy | CV Precision | CV Recall | CV F1    |
|----------------------|-------------|--------------|-----------|----------|
| SVM                  | 1.0000      | 1.0000       | 1.0000    | 1.0000   |
| KNN                  | 1.0000      | 1.0000       | 1.0000    | 1.0000   |
| Decision Tree        | 1.0000      | 1.0000       | 1.0000    | 1.0000   |
| Random Forest        | 1.0000      | 1.0000       | 1.0000    | 1.0000   |
| Logistic Regression  | 0.9998      | 1.0000       | 0.9995    | 0.9997   |

> Most models achieve perfect or near-perfect scores on this dataset because many features (especially **odor** and **spore-print-color**) are extremely predictive.

---

##  Visualizations Included

```text
figures/
├── class_distribution.png
├── odor_vs_class.png
├── cap_color_vs_class.png
├── gill_size_vs_class.png
├── feature_association.png          # Cramér’s V ranking
├── confusion_matrix_random_forest.png
└── decision_tree_visualization.png
```

---

##  Saved Models

```text
models/
├── logistic_regression.pkl
├── linear_regression.pkl
├── svm.pkl
├── knn.pkl
├── decision_tree.pkl
└── random_forest.pkl
```

All models can be loaded later with `joblib.load()` without retraining.

---

##  Project Structure

```text
mushroom-edibility-classification/
│
├── data/
│   ├── agaricus-lepiota.csv
│   └── agaricus-lepiota.names
│
├── notebooks/
│   └── mushroom_classification.ipynb
│
├── figures/
│   ├── class_distribution.png
│   ├── odor_vs_class.png
│   ├── cap_color_vs_class.png
│   ├── gill_size_vs_class.png
│   ├── feature_association.png
│   ├── confusion_matrix_random_forest.png
│   └── decision_tree_visualization.png
│
├── models/
│   ├── logistic_regression.pkl
│   ├── linear_regression.pkl
│   ├── svm.pkl
│   ├── knn.pkl
│   ├── decision_tree.pkl
│   └── random_forest.pkl
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

##  Technologies Used

| Category               | Tools / Libraries                          |
|------------------------|--------------------------------------------|
| Programming Language   | Python                                     |
| Data Processing        | Pandas, NumPy                              |
| Statistical Analysis   | SciPy (Chi-Square / Cramér’s V)            |
| Machine Learning       | Scikit-learn                               |
| Visualization          | Matplotlib, Seaborn                        |
| Model Persistence      | Joblib                                     |
| Environment            | Jupyter Notebook                           |

---

##  Installation

```bash
# Clone the repository
git clone https://github.com/Riktam45/mushroom-edibility-classification.git

# Navigate into the project
cd mushroom-edibility-classification

# Create virtual environment
python -m venv venv

# Activate
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

##  Requirements

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
joblib
jupyter
```

---

##  How to Run

1. Clone the repository
2. Install the requirements
3. Make sure the dataset is present at `data/agaricus-lepiota.csv`
4. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
5. Open `notebooks/mushroom_classification.ipynb`
6. Run all cells from top to bottom

The notebook will:
- Load and clean the data
- Perform EDA and association analysis
- Train six different models
- Evaluate them thoroughly
- Save the trained models to the `models/` folder

---

##  Important Notes & Limitations

- This is a **highly separable** dataset. Many models reach 100% accuracy.
- Real-world mushroom identification is far more complex and can be life-threatening.
- **Never** rely solely on a Machine Learning model for deciding whether a wild mushroom is safe to eat.
- Always consult expert mycologists and field guides.

---

##  Future Improvements

- Hyperparameter tuning with GridSearchCV / RandomizedSearchCV
- Add more advanced models (XGBoost, LightGBM, CatBoost)
- Build a simple Streamlit / Gradio web interface for prediction
- Deploy the best model as a web API
- Collect more real-world images + tabular data for multi-modal classification
- Explainability with SHAP values

---

##  Author

**Riktam Sarkar**

GitHub: [https://github.com/Riktam45](https://github.com/Riktam45)

This project was created for educational and learning purposes.
```
