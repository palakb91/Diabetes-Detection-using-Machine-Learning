# Diabetes-Detection-using-Machine-Learning
# 🩺 Diabetes Prediction using Machine Learning  

## 📘 Project Overview  
This project aims to predict whether a person has diabetes or not using machine learning algorithms.  
We use the **Pima Indians Diabetes Dataset**, perform exploratory data analysis (EDA), clean and preprocess the data, train **two ML models (Logistic Regression and Random Forest)**, and compare their performance based on several evaluation metrics.

---

## 📊 Dataset Description  

**Dataset name:** `diabetes.csv`  
**Rows:** 768  **Columns:** 9  

| Column | Description |
|:-------|:-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (after 2 hours in an oral glucose tolerance test) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index (weight in kg/(height in m)^2) |
| DiabetesPedigreeFunction | Diabetes pedigree function (family history impact) |
| Age | Age in years |
| Outcome | 1 = Diabetes, 0 = No Diabetes |

---

## ⚙️ Steps Implemented

### 1️⃣ Data Loading  
- Loaded dataset using `pandas.read_csv()`  
- Verified shape and column names  

### 2️⃣ Exploratory Data Analysis (EDA)  
Performed basic visualization and statistical analysis to understand data distribution and relationships.

**EDA steps included:**
- Viewing data info and summary statistics  
- Checking target class balance (`Outcome` column)  
- Plotting histograms of numeric features  
- Correlation heatmap to visualize relationships between variables  
- Countplot for outcome distribution  

📊 *Example Visuals:*  
- Feature histograms  
- Correlation matrix  
- Class balance plot  

---

### 3️⃣ Data Cleaning and Preprocessing  
- Identified invalid zero values in features like:
  `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`  
- Replaced zeros with median values for each column  
- Split dataset into **Train (80%)** and **Test (20%)** sets using `train_test_split()`  
- Applied **StandardScaler** for normalization  

---

### 4️⃣ Model Building  

Two models were trained using Scikit-learn pipelines for consistency and reproducibility:

#### 🔹 Model 1: Logistic Regression  
A simple and interpretable linear model used for binary classification.  
- Implemented with `LogisticRegression(max_iter=1000)`  
- Performance baseline model  

#### 🔹 Model 2: Random Forest Classifier  
An ensemble model that combines multiple decision trees to improve predictive performance.  
- Implemented with `RandomForestClassifier(n_estimators=200, random_state=42)`  

---

### 5️⃣ Model Evaluation  

Both models were evaluated on the test dataset using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC Score**

and visualized using **ROC Curves**.

---

## 📈 Results & Comparison  

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|:------|:----------|:-----------|:---------|:----------|:---------|
| Logistic Regression | 0.7078 | 0.6000 | 0.5000 | 0.5455 | 0.8130 |
| Random Forest | **0.7403** | **0.6522** | **0.5556** | **0.6000** | **0.8161** |

### 🧩 Interpretation:
- Both models achieved decent performance.  
- **Random Forest** slightly outperformed Logistic Regression in almost every metric.  
- ROC-AUC values show both models are good at distinguishing between diabetic and non-diabetic cases.  
- The confusion matrix shows Random Forest predicts positives (class 1) more accurately.

**✅ Best Model:** Random Forest Classifier  

---

### 🔍 Key Insights from EDA  
- Higher glucose levels and BMI are strongly associated with diabetes.  
- Older individuals have a higher probability of diabetes.  
- The dataset is slightly imbalanced (more non-diabetic than diabetic cases).  

---

## 💾 Model Saving  
The best model (Random Forest) is saved using `joblib` for future use.

```python
import joblib
joblib.dump({'model_name': 'RandomForest', 'model': best_model}, 'best_diabetes_model.pkl')
```

To load and use the saved model later:
```python
import joblib
m = joblib.load('best_diabetes_model.pkl')
model = m['model']
pred = model.predict(X_new)
```

---

## 📊 Sample Prediction Output

| True Label | Predicted Label | Probability (Class=1) |
|:------------|:----------------|:----------------------|
| 0 | 0 | 0.0021 |

---

## 🚀 Technologies Used  
- Python 🐍  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Joblib  

---

## 📚 Files in Repository  

| File | Description |
|:-----|:-------------|
| `diabetes.csv` | Dataset used for the project |
| `diabetes_prediction.ipynb` | Jupyter/Colab notebook with code |
| `best_diabetes_model.pkl` | Trained and saved Random Forest model |
| `README.md` | Project documentation (this file) |

---

## 🧠 Conclusion  

- Two machine learning models were implemented and compared.  
- Random Forest achieved **better performance (74% accuracy)** compared to Logistic Regression.  
- The analysis demonstrated that **Glucose**, **BMI**, and **Age** are the most influential predictors.  
- The final model can effectively be used for predicting diabetes risk in new patients.  

---

## 🔮 Future Enhancements  
- Implement additional models such as XGBoost or SVM.  
- Use hyperparameter tuning (GridSearchCV) for optimization.  
- Apply oversampling (SMOTE) for better handling of class imbalance.  
- Deploy model using Flask/Streamlit web app.  

---

## 👩‍💻 Author  
**Palak Bedi**  
B.Tech in Computer Science 
