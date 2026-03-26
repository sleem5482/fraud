from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware




class FraudInput(BaseModel):
    transaction_hour: int
    merchant_category: str
    foreign_transaction: int
    location_mismatch: int
    age_category: str
    amount_category: str
    velocity_category: str
    log_amount: float


def Load_model(path) : 
    model = joblib.load(path)
    return model 

app = FastAPI()

origins = ["https://fraud-frontend-henna.vercel.app"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def Infrence(data: FraudInput):

    columns = [
        "transaction_hour",
        "merchant_category",
        "foreign_transaction",
        "location_mismatch",
        "age_category",
        "amount_category",
        "velocity_category",
        "log_amount"
    ]

    cat_cols = [
        "merchant_category",
        "age_category",
        "amount_category",
        "velocity_category"
    ]

    data = pd.DataFrame([data], columns=columns)

    ohe = Load_model('./ohe.pkl')
    model = Load_model('./fraud_model.pkl')

    encoded = ohe.transform(data[cat_cols]).toarray()

    num = data.drop(columns=cat_cols)

    X = np.hstack([num.values, encoded])

    prediction = model.predict(X)

    return prediction


@app.get('/')
def check():
    return " hello"

@app.post("/predict")
def predict(data: FraudInput):

    input_data = [
        data.transaction_hour,
        data.merchant_category,
        data.foreign_transaction,
        data.location_mismatch,
        data.age_category,
        data.amount_category,
        data.velocity_category,
        data.log_amount
    ]

    prediction = Infrence(input_data)

    return {"prediction": int(prediction[0])}
