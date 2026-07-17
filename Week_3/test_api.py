import requests
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_endpoint(path, method="GET", data=None):
    url = f"{BASE_URL}{path}"
    try:
        if method == "GET":
            response = requests.get(url)
        else:
            response = requests.post(url, data=data)
        
        print(f"[{method}] {path} -> Status Code: {response.status_code}")
        if response.status_code != 200:
            print(f"Error response body: {response.text[:200]}")
            return False
        return response
    except Exception as e:
        print(f"Connection failed to {url}: {e}")
        return None

def main():
    print("Starting FastAPI Dashboard Endpoint verification...")
    
    # 1. Test Home Route
    res_home = test_endpoint("/")
    if not res_home:
        print("Verification failed: Home page unreachable.")
        sys.exit(1)
    assert "SteelEnergy.AI" in res_home.text, "Logo/Text missing from Home page"
    print("Home page contains logo and text correctly.")
    
    # 2. Test Dashboard Route
    res_dash = test_endpoint("/dashboard")
    if not res_dash:
        print("Verification failed: Dashboard page unreachable.")
        sys.exit(1)
    assert "energy_by_hour.png" in res_dash.text, "Hour chart missing from dashboard"
    assert "energy_by_load_type.png" in res_dash.text, "Load level chart missing from dashboard"
    assert "correlation_heatmap.png" in res_dash.text, "Correlation chart missing from dashboard"
    print("Dashboard page contains all 3 required chart images.")
    
    # 3. Test Predict GET Route
    res_pred_get = test_endpoint("/predict")
    if not res_pred_get:
        sys.exit(1)
    assert "Awaiting Input Parameters" in res_pred_get.text, "Predict page placeholder missing"
    print("Predict page loads placeholder correctly for GET requests.")
    
    # 4. Test Predict POST Route
    post_data = {
        "lagging_reactive": 15.2,
        "leading_reactive": 0.2,
        "co2": 0.02,
        "lagging_pf": 75.3,
        "leading_pf": 98.4,
        "nsm": 36000,
        "hour": 10,
        "month": 6,
        "day_of_week": "Wednesday",
        "week_status": "Weekday",
        "load_type": "Medium_Load"
    }
    res_pred_post = test_endpoint("/predict", method="POST", data=post_data)
    if not res_pred_post:
        sys.exit(1)
    
    assert "Predicted Active Energy" in res_pred_post.text, "Predict page result heading missing"
    assert "kWh" in res_pred_post.text, "Predict unit kWh missing"
    
    # Try to extract the predicted value
    # In predict.html: <div class="result-value" id="prediction-val">45.23</div>
    import re
    val_match = re.search(r'id="prediction-val">([^<]+)<', res_pred_post.text)
    if val_match:
        pred_val = val_match.group(1).strip()
        print(f"Real-time prediction successful! Predicted Energy: {pred_val} kWh")
        try:
            val_float = float(pred_val)
            print(f"Verified prediction value is numeric: {val_float}")
            assert val_float > 0, "Prediction value should be positive"
        except ValueError:
            print("Error: prediction value is not a number!")
            sys.exit(1)
    else:
        print("Error: Could not locate prediction value in returned HTML!")
        sys.exit(1)
        
    print("\nAll endpoints verified successfully! The FastAPI Dashboard is ready and correct.")

if __name__ == "__main__":
    main()
