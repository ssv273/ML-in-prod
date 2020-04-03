import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split

import numpy as np



dataset = pd.read_csv('Data/dataset/dataset_train.csv', sep=';')
X = dataset.drop(['user_id', 'is_churned'], axis=1)
y = dataset['is_churned']

X_mm = MinMaxScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_mm, 
                                                    y, 
                                                    test_size=0.3,
                                                    shuffle=True, 
                                                    stratify=y, 
                                                    random_state=100)

with open('models/baseline_xgb.pcl', 'rb') as f:
    model = pickle.load(f)

predict_test = model.predict(X_test)
predict_test_probas = model.predict_proba(X_test)[:, 1]

