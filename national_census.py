import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_curve, auc, accuracy_score)

df = pd.read_csv('train.tsv', delimiter='\t')
# df = pd.read_csv('train.tsv', delimiter='\t', skipfooter=3280)13000

#前処理
df['workclass'] = df['workclass'].replace({'Private':0, 'Local-gov':1, 'State-gov':2, 'Federal-gov':3, 'Self-emp-not-inc':4, 'Self-emp-inc':5, 'Without-pay':6, 'Never-worked':7})
df['education'] = df['education'].replace({'11th':0, 'Assoc-acdm':1, 'HS-grad':2, 'Masters':3, 'Bachelors':4, '10th':5, 'Some-college':6, 'Assoc-voc':7, '7th-8th':8, 'Doctorate':9, '12th':10, 'Prof-school':11, '1st-4th':12, '5th-6th':13, '9th':14, 'Preschool':15})
df['marital-status'] = df['marital-status'].replace({'Never-married':0, 'Married-civ-spouse':1, 'Divorced':2, 'Widowed':3, 'Separated':4, 'Married-spouse-absent':5, 'Married-AF-spouse':6})
df['occupation'] = df['occupation'].replace({'Handlers-cleaners':0, 'Craft-repair':1, 'Adm-clerical':2, 'Prof-specialty':3, 'Tech-support':4, 'Machine-op-inspct':5, 'Other-service':6, 'Transport-moving':7, 'Exec-managerial':8, 'Sales':9, 'Farming-fishing':10, 'Protective-serv':11, 'Priv-house-serv':12, 'Armed-Forces':13})
df['relationship'] = df['relationship'].replace({'Own-child':0, 'Husband':1, 'Wife':2, 'Unmarried':3, 'Not-in-family':4, 'Other-relative':5})
df['race'] = df['race'].replace({'White':0, 'Black':1, 'Asian-Pac-Islander':2, 'Amer-Indian-Eskimo':3, 'Other':4})
df['native-country'] = df['native-country'].replace({'United-States':0, 'Canada':1, 'Mexico':2,'Jamaica':3, 'South':4, 'Hungary':5, 'India':6, 'Germany':7, 'Guatemala':8, 'China':9, 'Iran':10, 'El-Salvador':11, 'Greece':12, 'Philippines' :13,'Ireland':14, 'Taiwan':15, 'Cuba':16, 'Cambodia':17, 'England':18, 'Thailand':19, 'Nicaragua':20, 'Dominican-Republic':21, 'Puerto-Rico':22, 'France':23, 'Hong':24, 'Columbia':25, 'Poland':26, 'Peru':27, 'Yugoslavia':28, 'Outlying-US(Guam-USVI-etc)':29, 'Haiti':30, 'Japan':31, 'Ecuador':32, 'Italy':33, 'Trinadad&Tobago':34, 'Vietnam':35, 'Laos':36, 'Scotland':37, 'Portugal':38, 'Honduras':39})
df['sex'] = df['sex'].apply(lambda x: 1 if x == 'Male' else 0)
df['Y'] = df['Y'].apply(lambda x: 1 if x == '>50K' else 0)

#欠損値処理
df2 = df.replace('?', np.nan).dropna()

X = df2.drop('Y', axis=1)
Y = df2.Y

# # #データをトレーニング用，評価用に分割
(train_X, test_X, train_Y, test_Y) = train_test_split(X, Y, test_size=0.2, random_state=0)


# # # モデル構築
clf = RandomForestClassifier(random_state=0)
clf = clf.fit(train_X, train_Y)
# # # 予測値を計算
pred = clf.predict(test_X)

fpr, tpr, thresholds = roc_curve(test_Y, pred, pos_label=1)
auc(fpr, tpr)
print(accuracy_score(pred, test_Y))
