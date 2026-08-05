# 📊 Study-Insight: Student Performance Analytics Platform

* 🎈 **Live Web App Link:** [study-insight-student-performance-prediction.streamlit.app](https://study-insight-student-performance-prediction-73zly6fczprhcq4fi.streamlit.app/)
* 📈 **Dataset:** [View on Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
* 🔄 **Experiment Tracking:** [View on MLflow/DagsHub](https://dagshub.com/anacondademon/MLproject.mlflow)

## 📌 Overview 
What actually drives a student's academic success? Is it strictly study time, or do hidden factors like nutrition and demographics play a bigger role? 

**Study-Insight** is a machine learning pipeline built to answer that question. It analyzes socioeconomic and educational variables to predict a student's math score. More importantly, it translates those predictions into actionable advice—helping educators and students understand exactly which external factors (like test prep or lunch programs) move the needle most.

---

## 🚀 Live Cloud Dashboard
The predictive model is deployed as a custom, highly interactive web app on Streamlit Community Cloud. 

### 1. Prediction & Insights Panel
Drop in a student's profile parameters to get a real-time predicted math score, global percentile, and specific advice based on historical data.

![Prediction Dashboard](images/Screenshot%202026-08-05%20170502.png)

### 2. Interactive Model Analytics
Machine learning shouldn't be a black box. The diagnostics tab visually maps the model's actual vs. predicted performance across 1,000 test data points, complete with a dynamic error color scale so you can see exactly where the model excels.

![Model Analytics](images/Screenshot%202026-08-05%20170520.png)

---

## 🧠 Approach & Key Findings

### Phase 1: Exploratory Data Analysis (EDA)
Before training any models, we dug into the dataset to uncover the actual stories hidden in the numbers. 

**The Insights:**
* **The Nutrition Factor:** This was one of the strongest predictors of success. Students on standard lunch programs scored, on average, **11 points higher** in math compared to students on free or reduced lunch programs. 

![EDA - Lunch Impact](images/Screenshot%202026-08-05%20170716.png)

* **Gender Disparities:** While male students averaged slightly higher in math (approx. 68 vs 63), female students significantly outperformed their male counterparts in reading and writing averages.

![EDA - Gender Distribution 1](images/Screenshot%202026-08-05%20170644.png)

![EDA - Gender Distribution 2](images/Screenshot%202026-08-05%20170659.png)

### Phase 2: Model Training & Evaluation
We pitted multiple regression algorithms against each other to find the best fit, tracking $R^2$ Scores and Mean Absolute Error (MAE). 

**Model Leaderboard:**
1. **Linear Regression: 88.11% 🏆**
2. Ridge Regression: 88.05%
3. CatBoost Regressor: 85.16%
4. Random Forest: 85.00%
5. XGBoost: 82.77%

*The takeaway? A well-processed dataset with clear linear relationships doesn't need overly complex ensemble methods. Linear Regression took the crown, offering both the highest accuracy and the best interpretability.*

![Model Leaderboard](images/Screenshot%202026-08-05%20170756.png)

![Regression Plots](images/Screenshot%202026-08-05%20170808.png)

### Phase 3: MLOps & Experiment Tracking
To keep the project production-ready, we integrated **MLflow** via DagsHub. This ensures every parameter is tracked, metrics are logged, and all registered models are strictly version-controlled.

![MLflow Tracking](images/Screenshot%202026-08-04%20200640.png)

---

## 🛠️ Tech Stack
* **Core:** Python, Pandas, NumPy
* **Machine Learning:** Scikit-Learn, CatBoost, XGBoost
* **Visuals:** Matplotlib, Seaborn, Plotly Express
* **MLOps:** MLflow, DagsHub
* **Deployment:** Streamlit Community Cloud

---

## 📞 Let's Connect
* **LinkedIn:** [Rahul Kumar Srivastava](https://www.linkedin.com/in/rahulkumarsrivastava/)
* **Email:** rahulkumarsrivastava90@gmail.com

*If you found these insights or the code structure helpful, feel free to drop a ⭐ on the repository!*