#1.load processed data from the processed folder
#2.create a model and train the model
#3.save the model in the artifacts folder
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

X_train=pd.read_csv("../data/processed/X_train.csv")
y_train=pd.read_csv("../data/processed/y_train.csv")
X_test=pd.read_csv("../data/processed/X_test.csv")
y_test=pd.read_csv("../data/processed/y_test.csv")

print(X_train)

model=LinearRegression()
model.fit(X_train,y_train)

with open("../artifacts/model.pkl","wb") as f:
    pickle.dump(model,f)