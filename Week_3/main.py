from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(
    title="Steel Industry Energy Consumption AI",
    description="Real-time energy consumption forecasting dashboard using PCA & Random Forest.",
    version="1.0.0"
)

# Mount static and templates folders
# This handles paths whether running from workspace root or inside Week_3 folder
base_path = os.path.dirname(os.path.abspath(__file__))
static_path = os.path.join(base_path, "static")
templates_path = os.path.join(base_path, "templates")

app.mount("/static", StaticFiles(directory=static_path), name="static")
templates = Jinja2Templates(directory=templates_path)

# Load the saved model pipeline
model_path = os.path.join(base_path, "model.joblib")
if os.path.exists(model_path):
    try:
        pipeline = joblib.load(model_path)
        print(f"Loaded model pipeline successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model pipeline: {e}")
        pipeline = None
else:
    print(f"Model pipeline not found at {model_path}. Make sure to generate it in the notebook first.")
    pipeline = None

# Day of week mapping helper
DAY_MAPPING = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html", 
        context={"active_page": "home"}
    )

@app.get("/dashboard", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html", 
        context={"active_page": "dashboard"}
    )

@app.get("/predict", response_class=HTMLResponse)
async def predict_get(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="predict.html", 
        context={"active_page": "predict", "prediction": None, "form_values": {}}
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict_post(
    request: Request,
    lagging_reactive: float = Form(...),
    leading_reactive: float = Form(...),
    co2: float = Form(...),
    lagging_pf: float = Form(...),
    leading_pf: float = Form(...),
    nsm: int = Form(...),
    hour: int = Form(...),
    month: int = Form(...),
    day_of_week: str = Form(...),
    week_status: str = Form(...),
    load_type: str = Form(...)
):
    # Form values to return to the template
    form_values = {
        "lagging_reactive": lagging_reactive,
        "leading_reactive": leading_reactive,
        "co2": co2,
        "lagging_pf": lagging_pf,
        "leading_pf": leading_pf,
        "nsm": nsm,
        "hour": hour,
        "month": month,
        "day_of_week": day_of_week,
        "week_status": week_status,
        "load_type": load_type
    }

    # Reconstruct engineered features
    # 1. Day of Week index (0=Monday, 6=Sunday)
    day_of_week_idx = DAY_MAPPING.get(day_of_week, 0)
    
    # 2. IsWeekend flag
    is_weekend = 1 if day_of_week_idx in [5, 6] else 0
    
    # 3. Power Factor Ratio
    calculated_pf_ratio = leading_pf / lagging_pf if lagging_pf != 0 else 0.0

    # 4. Construct dummy encoding columns exactly matching X_train.columns (20 features)
    features_dict = {
        "Lagging_Current_Reactive.Power_kVarh": lagging_reactive,
        "Leading_Current_Reactive_Power_kVarh": leading_reactive,
        "CO2(tCO2)": co2,
        "Lagging_Current_Power_Factor": lagging_pf,
        "Leading_Current_Power_Factor": leading_pf,
        "NSM": nsm,
        "Hour": hour,
        "DayOfWeek": day_of_week_idx,
        "Month": month,
        "IsWeekend": is_weekend,
        "Power_Factor_Ratio": calculated_pf_ratio,
        "WeekStatus_Weekend": 1 if week_status == "Weekend" else 0,
        "Day_of_week_Monday": 1 if day_of_week == "Monday" else 0,
        "Day_of_week_Saturday": 1 if day_of_week == "Saturday" else 0,
        "Day_of_week_Sunday": 1 if day_of_week == "Sunday" else 0,
        "Day_of_week_Thursday": 1 if day_of_week == "Thursday" else 0,
        "Day_of_week_Tuesday": 1 if day_of_week == "Tuesday" else 0,
        "Day_of_week_Wednesday": 1 if day_of_week == "Wednesday" else 0,
        "Load_Type_Maximum_Load": 1 if load_type == "Maximum_Load" else 0,
        "Load_Type_Medium_Load": 1 if load_type == "Medium_Load" else 0
    }

    # Align columns in the exact order used during training
    columns_order = [
        "Lagging_Current_Reactive.Power_kVarh",
        "Leading_Current_Reactive_Power_kVarh",
        "CO2(tCO2)",
        "Lagging_Current_Power_Factor",
        "Leading_Current_Power_Factor",
        "NSM",
        "Hour",
        "DayOfWeek",
        "Month",
        "IsWeekend",
        "Power_Factor_Ratio",
        "WeekStatus_Weekend",
        "Day_of_week_Monday",
        "Day_of_week_Saturday",
        "Day_of_week_Sunday",
        "Day_of_week_Thursday",
        "Day_of_week_Tuesday",
        "Day_of_week_Wednesday",
        "Load_Type_Maximum_Load",
        "Load_Type_Medium_Load"
    ]

    # Create DataFrame for prediction
    input_df = pd.DataFrame([features_dict])[columns_order]

    # Predict using the pipeline
    prediction = None
    if pipeline is not None:
        try:
            # Predict returns a numpy array, take the first value
            prediction = float(pipeline.predict(input_df)[0])
        except Exception as e:
            print(f"Prediction failed: {e}")
            prediction = -1.0
    else:
        print("Pipeline is not loaded. Prediction skipped.")
        prediction = 0.0

    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={
            "active_page": "predict",
            "prediction": prediction,
            "form_values": form_values,
            "calculated_pf_ratio": calculated_pf_ratio,
            "is_weekend": is_weekend,
            "day_of_week_idx": day_of_week_idx
        }
    )
