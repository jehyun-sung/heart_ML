# -*- coding: utf-8 -*-
"""인공 하츄전문핑 ML

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1a1nlR1TYWzbPlA9d6NTZzOWv7bRkQDLf
"""

import pandas as pd
from matplotlib import pyplot as plt
from google.colab import drive
from matplotlib.pyplot import figure

#Importing csv file
url = "https://raw.githubusercontent.com/ynoh253/Heart/ee89febfa51796882b06f9d995f02d3e8c0a2bd4/heart_disease_prediction.csv"
df = pd.read_csv(url)
df.info()

df.head()
#sex             1 = male, 0= female
#Chest Pain Type -- Value 1: typical angina
#                -- Value 2: atypical angina
#                -- Value 3: non-anginal pain
#                -- Value 4: asymptomatic
#Fasting Blood sugar(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
#higher than 120 indicates diabetes
#Resting electrocardiogram results -- Value 0: normal
#(testing heart beat rate)         -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
#                                  -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
#Exercise induced angina.          1 = yes; 0 = no
#the slope of the peak exercise ST segment.  -- Value 1: upsloping
#                                            -- Value 2: flat
#                                            -- Value 3: downsloping
#class              1 = heart disease, 0 = Normal

#Plotting
fig = figure(num=None, figsize=(15, 10))
#fig.tight_layout(h_pad=5)

xlabel = ['in years','0,1','1,2,3,4','mm Hg','mg/dl','1,0 > 120mg/dl','0,1,2','bpm','0,1','depression','0,1,2','0,1']
for i in range(1, 13):
  plt.subplot(3,4,i)
  x = df.iloc[0:-1, i-1]
  plt.title(df.columns[i-1])
  plt.xlabel(xlabel[i-1])
  plt.ylabel('Number of Patients')
  values, bins, bars = plt.hist(x)
  plt.bar_label(bars, fontsize=8)
plt.subplots_adjust(wspace=0.35, hspace=0.35)

df.dtypes

"""Categorical variables >> from integer to object & one-hot encoding (nominal)

sex, chest pain type, fasting blood sugar, resting ecg, exercise angina, ST slope, target

Need One-Hot Encoding: cp, restecg

Don't Need One-Hot Encoding: sex, fbs, exang, slope, ca --> binary
"""

from sklearn.preprocessing import OneHotEncoder

categorical_nominal = ["chest pain type", "resting ecg"]

encoder = OneHotEncoder(sparse_output=False)

one_hot_encoded = encoder.fit_transform(df[categorical_nominal])

#one_hot_df = pd.DataFrame(one_hot_encoded)
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_nominal))


df_encoded = pd.concat([df, one_hot_df], axis=1)

df_encoded = df_encoded.drop(categorical_nominal, axis=1)

df_encoded.head()

from sklearn.model_selection import train_test_split
no_target = df_encoded.drop('target', axis=1)
data = no_target.to_numpy()
target = df_encoded['target'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

print(data.shape, target.shape)
print(train_input.shape, test_input.shape, train_target.shape, test_target.shape)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

#Logistic Regression

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000, max_iter= 10)

lr.fit(train_scaled, train_target)

print(lr.coef_, lr.intercept_)

print(lr.predict_proba(train_scaled[:5]))

from scipy.special import expit
z_scores = lr.decision_function(train_scaled)
print(expit(z_scores[:5]))

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

#Random Forest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

rf = RandomForestClassifier(n_jobs=-1, random_state = 42)
rf.fit(train_input, train_target)
print(rf.score(train_input, train_target))
print(rf.score(test_input, test_target))
print(rf.feature_importances_)

#with cross-validation
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
rf.fit(train_input, train_target)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
print(rf.score(test_input, test_target))
print('')

# from sklearn.model_selection import GridSearchCV
# params = {'min_impurity_decrease' : np.arange(0.0001, 0.001, 0.0001),'max_depth': range(5, 20, 1), 'min_samples_split': range(2, 100, 10)}
# #0.0001부터 0.0005까지 0.0001씩 증가하는 5개의 값을 시도.
# gsrf = GridSearchCV(RandomForestClassifier(random_state = 42), params,n_jobs=-1)
# #cv parameter 기본값은 5. min_impurity_decrease 값마다 5-폴드 교차 검증.
# #>>  25개의 모델을 훈련.
# #n_jobs는 병렬 실행에 사용할 CPU 코어 수. default = 1, -1 : use all the cores.
# gsrf.fit(train_input, train_target)
# rf1 = gsrf.best_estimator_ #best models are stored in best_pestimator_
# print(rf1.score(train_input, train_target))
# print(gsrf.best_params_)
# print(gsrf.cv_results_['mean_test_score'])
# best_index = np.argmax(gsrf.cv_results_['mean_test_score'])
# print(gsrf.cv_results_['params'][best_index])
# print(rf1.score(test_input, test_target))

#randomforest randomsearch
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
params = {'min_impurity_decrease': uniform(0.0001, 0.001), 'max_depth': randint(20, 50), 'min_samples_split': randint(2, 25), 'min_samples_leaf': randint(1, 25)}
rsrf = RandomizedSearchCV(RandomForestClassifier(random_state=42), params, n_iter=100, n_jobs=-1, random_state=42)
rsrf.fit(train_input, train_target)
rf3 = rsrf.best_estimator_
print(rf3.score(test_input, test_target))

print(rsrf.best_params_)

rf4 = RandomForestClassifier(max_depth= 42,
                            min_impurity_decrease= 0.00037864646423661145,
                            min_samples_leaf= 1,
                            min_samples_split = 2)

rf4.fit(train_input, train_target)
print(rf4.score(train_input, train_target))
print(rf4.score(test_input, test_target))

#Decision Tree
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
import numpy as np
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))
importance = dt.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))

#그리드서치
from sklearn.model_selection import GridSearchCV
#gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
params = {'max_depth': [2,4,7,10]}
wine_tree = DecisionTreeClassifier(max_depth=2, random_state=13)

gridsearch = GridSearchCV(estimator=wine_tree, param_grid=params, cv=5)
gridsearch.fit(train_input, train_target)
print(gridsearch.cv_results_['mean_test_score'])
gridsearch.best_estimator_
gridsearch.best_score_
gridsearch.best_params_

#그리드서치
from sklearn.model_selection import GridSearchCV

params = {
    'max_depth': [14, 15, 16, 17, 18],
    'max_features': ["sqrt", "log2"],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4]
}
gs = GridSearchCV(RandomForestClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
print(gs.best_params_)
#{'max_depth': 15, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2}

rf2= RandomForestClassifier(max_depth= 15,
                            max_features= "sqrt",
                            min_samples_leaf= 1,
                            min_samples_split = 2)

rf2.fit(train_input, train_target)
print(rf2.score(train_input, train_target))
print(rf2.score(test_input, test_target))

"""##Gradio"""

pip install gradio

import gradio as gr

from sklearn import datasets
import joblib

train_input.shape

import pickle

#'saved_model'은 저장할 파일의 이름이다.
with open('saved_model', 'wb') as f:
    pickle.dump(rf2, f)

import pickle
#이제 "mod"라는 이름에 파일을 불러왔기 때문에 mod로 원하는 작업을 수행하면 된다.
with open('saved_model', 'rb') as f:
    mod = pickle.load(f)

mod.predict([[52, 1, 100, 100, 100, 150, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0]])

import gradio as gr

def predict(age, sex, resting_bp_s, cholesterol, fasting_blood_sugar, max_heart_rate, exercise_angina, oldpeak, ST_slope, chest_pain_type_1, chest_pain_type_2, chest_pain_type_3, chest_pain_type_4, resting_ecg_0, resting_ecg_1, resting_ecg_2):
    info = [[age, sex, resting_bp_s, cholesterol, fasting_blood_sugar, max_heart_rate, exercise_angina, oldpeak, ST_slope, chest_pain_type_1, chest_pain_type_2, chest_pain_type_3, chest_pain_type_4, resting_ecg_0, resting_ecg_1, resting_ecg_2]]
    # age, sex, resting bp s, cholesterol, fasting blood sugar, max heart rate, exercise angina, oldpeak, ST slope, chest pain type 1, chest pain type 2, chest pain type 3, chest pain type 4, resting ecg_0, resting ecg_1, resting ecg_2])
    result = mod.predict(info)
    if result[0] == 1:
      return "Patient has heart disease."
    else:
      return "Patient does not have heart disease"



demo = gr.Interface(
    fn=predict,
    inputs=["text", "text", "text", "text", "text", "text", "text", "text", "text", "text", "text", "text", "text", "text", "text", "text"],
    outputs=["text"],
)
demo.launch()

import gradio as gr

def predict(age, sex, resting_bp_s, cholesterol, fasting_blood_sugar, max_heart_rate, exercise_angina, oldpeak, ST_slope, chest_pain_type_1, chest_pain_type_2, chest_pain_type_3, chest_pain_type_4, resting_ecg_0, resting_ecg_1, resting_ecg_2):
    if sex == "female":
      sex = 0
    else:
      sex = 1
    if fasting_blood_sugar == "> 120 mg/dl":
      fasting_blood_sugar = 1
    else:
      fasting_blood_sugar = 0
    if exercise_angina == "yes":
      exercise_angina = 1
    else:
      exercise_angina = 0
    if ST_slope == "upsloping":
      ST_slope = 0
    elif ST_slope == "flat":
      ST_slope = 1
    else:
      ST_slope = 2
    info = [[age, sex, resting_bp_s, cholesterol, fasting_blood_sugar, max_heart_rate, exercise_angina, oldpeak, ST_slope, chest_pain_type_1, chest_pain_type_2, chest_pain_type_3, chest_pain_type_4, resting_ecg_0, resting_ecg_1, resting_ecg_2]]
    # age, sex, resting bp s, cholesterol, fasting blood sugar, max heart rate, exercise angina, oldpeak, ST slope, chest pain type 1, chest pain type 2, chest pain type 3, chest pain type 4, resting ecg_0, resting ecg_1, resting ecg_2])
    result = mod.predict(info)
    if result[0] == 1:
      return "Patient has heart disease."
    else:
      return "Patient does not have heart disease"



demo = gr.Interface(
    fn=predict,
    inputs=[gr.Slider(1, 100, value=20, step = 1), gr.Dropdown(["female", "male"]), "text", "text", gr.Dropdown(["> 120 mg/dl", "< 120 mg/dl"], label = "fasting blood sugar"), gr.Slider(71, 201, value=100, step = 1, label="maximum heart rate"), gr.Dropdown(["no", "yes"], label="exercise induced angina"), "text", gr.Dropdown(["upsloping", "flat", "downsloping"], label = "the slope of the peak exercise ST segment"),   gr.Checkbox(label="typical angina"), gr.Checkbox(label="atypical angina"), gr.Checkbox(label="non-anginal pain"), gr.Checkbox(label="asymptomatic"), gr.Checkbox(label="resting ecg - normal"), gr.Checkbox(label="resting ecg - ST elevation or depression"), gr.Checkbox(label="resting ecg - showing left ventricular hypertrophy")],
    outputs=["text"],
)
demo.launch()

