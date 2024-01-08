import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

df_train = pd.read_excel('train.xlsx')
df_test = pd.read_excel('test.xlsx')

df_train['JENIS KELAMIN'] = df_train['JENIS KELAMIN'].apply(lambda x: 1 if x == "LAKI - LAKI" else 0)
df_train['STATUS MAHASISWA'] = df_train['STATUS MAHASISWA'].apply(lambda x: 1 if x == "MAHASISWA" else 0)
df_train['STATUS NIKAH'] = df_train['STATUS NIKAH'].apply(lambda x: 1 if x == "BELUM MENIKAH" else 0)

df_test['JENIS KELAMIN'] = df_test['JENIS KELAMIN'].apply(lambda x: 1 if x == "LAKI - LAKI" else 0)
df_test['STATUS MAHASISWA'] = df_test['STATUS MAHASISWA'].apply(lambda x: 1 if x == "MAHASISWA" else 0)
df_test['STATUS NIKAH'] = df_test['STATUS NIKAH'].apply(lambda x: 1 if x == "BELUM MENIKAH" else 0)


#select independet and dependent variable
X_train = df_train[['JENIS KELAMIN', 'STATUS MAHASISWA', 'STATUS NIKAH','UMUR', 'IPS 1', 'IPS 2', 'IPS 3', 'IPS 4', 'IPS 5', 'IPS 6', 'IPS 7', 'IPS 8']]
y_train = df_train['STATUS KELULUSAN']

X_test = df_test[['JENIS KELAMIN', 'STATUS MAHASISWA', 'STATUS NIKAH', 'UMUR', 'IPS 1', 'IPS 2', 'IPS 3', 'IPS 4', 'IPS 5', 'IPS 6', 'IPS 7', 'IPS 8']]
y_test = df_test['STATUS KELULUSAN']

# print(df_train.head())

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

lr = LogisticRegression()
lr.fit(X_train, y_train)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

models = {
    'RandomForest': rf,
    'LogisticRegression': lr,
    'DecisionTree': tree
}

with open('models.pkl', 'wb') as file:
    pickle.dump(models, file)