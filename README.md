# heart_ML

❤️ Heart Disease Prediction using Machine Learning

This project builds and deploys a machine learning model to predict the presence of heart disease using patient health data. It leverages data preprocessing, multiple ML models, and an interactive Gradio web interface.

📊 Dataset

Dataset Source:
Heart Disease Dataset from GitHub

Features include:

age, sex, resting blood pressure, cholesterol, fasting blood sugar, max heart rate, exercise-induced angina
Categorical features such as chest pain type, resting ECG, ST slope
target: 1 = heart disease, 0 = no heart disease
🔍 Project Workflow

✅ Data Preprocessing
One-Hot Encoding applied to nominal categorical variables:
chest pain type
resting ecg
Binary categorical features (sex, fbs, exang, slope) are kept as-is.
Feature scaling with StandardScaler.
🧠 Models Used
Logistic Regression
Random Forest
With both GridSearchCV and RandomizedSearchCV hyperparameter tuning
Decision Tree
With Grid Search
🎯 Performance Evaluation
Each model is trained and evaluated on an 80/20 train-test split. Random forest models are optimized through cross-validation and hyperparameter tuning.

🧪 Example: Random Forest Results
RandomForestClassifier(max_depth=15, max_features='sqrt', min_samples_leaf=1, min_samples_split=2)
Accuracy (Train): ~1.0
Accuracy (Test): ~0.86
🖥️ Gradio Web App

An interactive interface built with Gradio allows users to input patient data and receive predictions in real time.

✅ Features
Dropdowns and sliders for user-friendly input
Model inference based on trained RandomForestClassifier
Binary classification output: "Patient has heart disease." or "Patient does not have heart disease."

🧪 Example
Input:
  age = 52
  sex = male
  chest pain type = typical angina
  fasting blood sugar = >120 mg/dl
  max heart rate = 150
  ...
Output:
  "Patient has heart disease."
🛠️ Model Saving & Loading
The final trained model is saved and loaded with pickle:

with open('saved_model', 'wb') as f:
    pickle.dump(rf2, f)

with open('saved_model', 'rb') as f:
    mod = pickle.load(f)

🐍 Requirements

Python 3.x
pandas
scikit-learn
matplotlib
numpy
gradio
Install requirements:

pip install pandas scikit-learn matplotlib numpy gradio

📦 Deployment

Launch the app with:

demo.launch()
Or run locally in Jupyter/Colab.

📁 File Structure (if converted into a repo)

.
├── heart_disease_prediction.ipynb  # Jupyter/Colab notebook
├── saved_model                     # Trained model (pickled)
├── README.md

🙋 Author

Jehyun Sung
