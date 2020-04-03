import time
from datetime import datetime, timedelta
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

from config import *

def time_format(sec):
    return str(timedelta(seconds=sec))

def prepare_dataset(dataset, 
                    dataset_type='train',
                    dataset_path='Data/dataset/'):
    print(dataset_type)
    start_t = time.time()
    print('Dealing with missing values, outliers, categorical features...')
    
    # Профили
    dataset['age'] = dataset['age'].fillna(dataset['age'].median())
    dataset['gender'] = dataset['gender'].fillna(dataset['gender'].mode()[0])
    dataset.loc[~dataset['gender'].isin(['M', 'F']), 'gender'] = dataset['gender'].mode()[0]
    dataset['gender'] = dataset['gender'].map({'M': 1., 'F':0.})
    dataset.loc[(dataset['age'] > 80) | (dataset['age'] < 7), 'age'] = round(dataset['age'].median())
    dataset.loc[dataset['days_between_fl_df'] < -1, 'days_between_fl_df'] = -1
    # Пинги
    for period in range(1,len(INTER_LIST)+1):
        col = 'avg_min_ping_{}'.format(period)
        dataset.loc[(dataset[col] < 0) | 
                    (dataset[col].isnull()), col] = dataset.loc[dataset[col] >= 0][col].median()
    # Сессии и прочее
    dataset.fillna(0, inplace=True)
    dataset.to_csv('{}dataset_{}.csv'.format(dataset_path, dataset_type), sep=';', index=False)
         
    print('Dataset is successfully prepared and saved to {}, run time (dealing with bad values): {}'.\
          format(dataset_path, time_format(time.time()-start_t))) 

train = pd.read_csv('Data/dataset/dataset_raw_train.csv', sep=';')
test = pd.read_csv('Data/dataset/dataset_raw_test.csv', sep=';')
print('train.shape is ',train.shape,'test.shape is', test.shape)

prepare_dataset(dataset=train, dataset_type='train')
prepare_dataset(dataset=test, dataset_type='test')

train_new = pd.read_csv('Data/dataset/dataset_train.csv', sep=';')
print(train_new['is_churned'].value_counts())

X_train = train_new.drop(['user_id', 'is_churned'], axis=1)
y_train = train_new['is_churned']

X_train_mm = MinMaxScaler().fit_transform(X_train)
print('Балансируем классы...')

X_train_balanced, y_train_balanced = SMOTE(random_state=42, ratio=0.3). \
                                        fit_sample(X_train_mm, y_train.values)


print('До:', Counter(y_train.values))
print('После:', Counter(y_train_balanced))