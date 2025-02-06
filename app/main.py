from fastapi import FastAPI, File, UploadFile
import pandas as pd
import io
from xgboost import XGBClassifier

app = FastAPI()

final_features = [
    " Net Value Growth Rate",
    " Equity to Liability",
    " Borrowing dependency",
    " Interest-bearing debt interest rate",
    " Total debt/Total net worth",
    " Non-industry income and expenditure/revenue",
    " After-tax net Interest Rate",
    " Total income/Total expense",
    " Interest Coverage Ratio (Interest expense to EBIT)",
    " Degree of Financial Leverage (DFL)",
    " Cash Turnover Rate",
    " Quick Ratio",
    " Quick Assets/Current Liability",
    " Current Liability to Current Assets",
    " Operating profit per person",
    " Net Value Per Share (A)",
    " Research and development expense rate",
    " Allocation rate per person",
    " Quick Asset Turnover Rate",
    " Average Collection Days",
    " Total Asset Growth Rate",
    " Cash flow rate",
    " No-credit Interval",
    " Inventory/Working Capital",
    " Net Income to Total Assets",
    " Operating Profit Growth Rate",
    " Inventory Turnover Rate (times)",
    " Cash/Total Assets",
    " Revenue per person",
    " Operating Expense Rate",
]

model = XGBClassifier()
model.load_model("model.json")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    print(df.columns)

    missing = set(final_features) - set(df.columns)
    if missing:
        return {"error": f"Missing columns in CSV: {missing}"}

    df_final = df[final_features]
    predictions = model.predict(df_final)
    return {"predictions": predictions.tolist()}
