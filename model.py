import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
import xgboost as xg
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor


df = pd.read_csv("df1.csv")

for i in df.columns:
    df[i] = df[i].astype('float64')
    
X = df.drop(['drug_perm_per','drug_perm_amt'], axis=1)
y = df[["drug_perm_per"]]

model_1 = xg.XGBRegressor(max_depth = 10, eta = 0.2, verbosity = 0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=1)
model_3 = GradientBoostingRegressor()
model_4 = neighbors.KNeighborsRegressor(n_neighbors = 1)
final_model = VotingRegressor(estimators=[('xgb', model_1), ('rf', model_2), ('gbr', model_3), ('knn',model_4)])

final_model.fit(X,y)

import pickle
pickle_out = open('vr_model.pkl','wb')
pickle.dump(final_model, pickle_out)
pickle_out.close()
