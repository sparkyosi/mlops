import sys  
import json
import mlflow
import sklearn
import uvicorn
import numpy as np
import pandas as pd
from operator import index
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from preprocessing import date_processing, scale_data
from example_json.drought_info import DroughtModel
from best_model import model

#API
app = FastAPI(title = "Model Tracking")
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

loaded_model = model()

@app.get("/")
def read_root():
    return {"Welcome": "to Drought Prediction Using MLOps app version 1."}

# Predict 
@app.post("/predict")
def predict(data : DroughtModel):
    data = pd.DataFrame(data.dict(),index=[0])
    data = date_processing(data)
    df = scale_data(data)

    columns = ['fips', 'PRECTOT', 'PS', 'QV2M', 'T2M', 'T2MDEW', 'T2MWET',
           'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'TS', 'WS10M', 'WS10M_MAX',
           'WS10M_MIN', 'WS10M_RANGE', 'WS50M', 'WS50M_MAX', 'WS50M_MIN',
           'WS50M_RANGE', 'year', 'month', 'day']

    # Conversion en DataFrame
    df = pd.DataFrame(df, columns=columns)

    prediction = loaded_model.predict(df)
    return {"predictions": prediction.tolist()}



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)