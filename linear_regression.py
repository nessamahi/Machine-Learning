import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import datasets

df = pd.read_excel('data.xlsx')
length = df.shape[0]
test_data = df.loc[length-1].tolist()
print(test_data)

df = df.drop([df.shape[0]-1]).sample(frac=1).reset_index(drop=True)

print(df[0:10], "\n")

train_data = df.iloc[0:10][['PopulationAbove65', 'HospitalBeds', 'DeathsPer100k']].values
# print(train_data)

train_X = train_data[:, 0:2]
train_Y = np.squeeze(train_data[:, 2])  # .squeeze(1)

print("Trained value of X : \n", train_X, "\n")
print("Trained Value of Y : \n", train_Y, "\n")

model = LinearRegression().fit(train_X, train_Y)

test_X = np.array(test_data[1:3]).reshape(1, 2)
pred = model.predict(test_X)

print("Predicted Data : ", pred, "\nAnd actual Data : ", test_data[-1])
