import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("D:/22B2492@iitb.ac.in/PythonProjects/Heat Flux Predictor/Data_CHF_Zhao_2020_ATE.csv")

print(data)
data.info()

# Preprocessing
def preprocess_inputs(df):
    df=df.copy()

    # Drop id and author columns
    df = df.drop(['id','author'],axis=1)

    # Shuffle the dataset
    df = df.sample(frac=1.0, random_state=1)

    # Split dataframe into X and y
    y = df['chf_exp [MW/m2]']
    X = df.drop('chf_exp [MW/m2]',axis=1)

    return X, y

X, y = preprocess_inputs(data)

print(X)
print(y)

# Building Pipeline
def build_model():

    nominal_transformer = Pipeline(steps=[('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[('nominal', nominal_transformer,['geometry'])], remainder='passthrough')

    model = Pipeline(steps=[('preprocessor', preprocessor),('regressor', RandomForestRegressor(random_state=1))])

    return model

# Training
kf = KFold(n_splits=5)

rmses = []

for train_idx, test_idx in kf.split(X):
    X_train = X.iloc[train_idx, :]
    X_test = X.iloc[test_idx, :]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    
    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(np.mean((y_test-y_pred)**2))

    rmses.append(rmse)

final_rmse = np.mean(rmses)

print(y_pred)
# print(rmses)
# print('RMSE: {:.2f}'.format(final_rmse))
# y_test.plot(kind='hist')
# plt.savefig('Frequency.png')
# plt.show()