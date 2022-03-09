import os
import argparse
import time
from tkinter import E
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

def evaluate(actual, pred):
  rmse = np.sqrt(mean_absolute_error(actual, pred))
  mae = mean_absolute_error(actual, pred)
  r2 = r2_score(actual, pred)
  return rmse, mae, r2
def get_data():
  URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
  try:
    df = pd.read_csv(URL, sep=";")
    return df
  except Exception as e:
    raise e
  
def main(alpha, l1_ratio):
  df = get_data()
  train, test = train_test_split(df)
  TARGET = "quality"
  train_x = train.drop([TARGET], axis=1)
  test_x = test.drop([TARGET], axis=1)
  train_y = train[[TARGET]]
  test_y = test[[TARGET]]
  #MLflow logging
  with mlflow.start_run():
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    model_lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model_lr.fit(train_x, train_y)
    pred = model_lr.predict(test_x)
    rmse, mae, r2 = evaluate(test_y, pred)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    print(f"Params- alpha = {alpha}, l1_ration = {l1_ratio}")
    print(f"Eval Metrics - rmae = {rmse}, mae = {mae}, r2 = {r2}")

    mlflow.sklearn.log_model(model_lr, "model") #modelname,  folder to save
    
 

if __name__ == '__main__':
  args = argparse.ArgumentParser()
  args.add_argument("--alpha", "-a", type=float, default=0.5)
  args.add_argument("--l1_score", "-l1", type=float, default=0.5)
  parsed_args = args.parse_args()

  main(alpha=parsed_args.alpha, l1_ratio=parsed_args.l1_score)