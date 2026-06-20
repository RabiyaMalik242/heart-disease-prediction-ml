# ❤️ Heart Disease Prediction

A machine learning project that predicts whether a patient is at risk of heart disease based on clinical health data. Built as part of an AI/ML Internship — Task 3.

---

## 📌 Objective

Build and evaluate binary classification models to predict heart disease risk using the Heart Disease UCI Dataset, and identify the most important clinical features driving predictions.

---

## 📂 Dataset

- **Source:** [Heart Disease UCI Dataset — Kaggle](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)
- **File:** `heart.csv`
- **Size:** 303 patient records, 14 features
- **Target:** `condition` — 0 = No Disease, 1 = Heart Disease

**Key Features:**

| Feature | Description |
|---|---|
| `age` | Age in years |
| `cp` | Chest pain type (0–3) |
| `thalach` | Maximum heart rate achieved |
| `ca` | Number of major vessels (0–3) |
| `oldpeak` | ST depression induced by exercise |
| `thal` | Thalassemia type |

---

## 🛠️ Tools & Libraries

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

---

## ⚙️ Project Pipeline

1. **Data Loading & Inspection** — shape, dtypes, missing values
2. **Exploratory Data Analysis (EDA)**
   - Target distribution
   - Feature histograms by class
   - Correlation heatmap
   - Box plots for outlier detection
   - Categorical feature analysis
3. **Preprocessing**
   - Train/test split (80/20, stratified)
   - StandardScaler (before vs after comparison)
4. **Model Training**
   - Logistic Regression
   - Decision Tree Classifier
5. **Evaluation**
   - Accuracy, ROC-AUC, Confusion Matrix
   - ROC Curves
6. **Feature Importance Analysis**
   - LR coefficients
   - DT Gini importance
   - Decision Tree visualization
7. **Predictions**
   - Test set predictions with error analysis
   - Single patient prediction with risk gauge
   - Batch predictions for multiple patients
8. **Model Saving** — joblib (.pkl)

---

## 📊 Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 91% | 0.95 ✅ |
| Decision Tree | 80% | 0.78 |

**Logistic Regression is the recommended model** — 95% ROC-AUC is considered excellent for medical classification tasks.

---

## 🔑 Key Findings

- `cp` (chest pain type), `thalach` (max heart rate), `ca` (major vessels), and `oldpeak` (ST depression) are the strongest predictors of heart disease
- Logistic Regression significantly outperformed Decision Tree due to the small dataset size (303 samples) where linear models generalize better
- The performance gap (ROC-AUC: 0.95 vs 0.78) confirms LR as the model of choice for deployment

---

## 💾 Models

Run the notebook to train and save the models locally. The following files will be generated:

```
heart_disease_lr_model_BEST.pkl   ← recommended
heart_disease_dt_model.pkl
heart_disease_scaler.pkl
```
> ⚠️ The scaler must be loaded alongside Logistic Regression — new inputs must be scaled before prediction.

---

## 📁 Repository Structure

```
heart-disease-prediction-ml/
│
├── heart_disease_prediction.ipynb   # Main notebook
├── heart.csv                        # Dataset
├── images/                          # Saved plot images
└── README.md
```
---

## 🚀 How to Run

1. Clone the repository
```bash
   git clone https://github.com/RabiyaMalik242/heart-disease-prediction-ml.git
```
2. Install dependencies
```bash
   pip install numpy pandas matplotlib seaborn scikit-learn joblib
```
3. Download `heart.csv` from [Kaggle](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci) and place it in the project folder
4. Open and run `heart_disease_prediction.ipynb` top to bottom

---

## ⚕️ Disclaimer

This project is for educational purposes only. The model is not a substitute for professional medical diagnosis. Always consult a qualified healthcare professional for medical concerns.

---

## 👩‍💻 Author

Rabiya Malik BS Software Engineering — AI/ML Internship Project

