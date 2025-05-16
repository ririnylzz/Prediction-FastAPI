from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Inisialisasi FastAPI
app = FastAPI(title="Product Category Prediction API (GnB + Scaler)")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Alternatively use ["http://localhost"] to be more restrictive
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = r"D:\xampp\htdocs\capstone\backend\WebMinnersgnb.pkl"
# Load model dan scaler
try:
    with open(model_path, "rb") as f:
        saved_objects = pickle.load(f)
        model = saved_objects['model']
        scaler = saved_objects['scaler']
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Skema input sesuai fitur data - Perbaiki tipe data
class InputData(BaseModel):
    Product_Name: str  # Changed to string to match what frontend sends
    Product_Price: float
    Quantity: float
    Total: float  # Changed to float to match what frontend sends
    Month: int
    Quantity_Monthly: float
    Day: int
    Year: int

# Preprocessing input
def preprocess_input(data: InputData):
    # Convert Product_Name to numeric representation if model requires it
    data_dict = data.dict()
    
    # If your model expects Product_Name as numeric, convert it using hash or lookup
    data_dict['Product_Name'] = abs(hash(data_dict['Product_Name']) % 10000)
    
    df = pd.DataFrame([data_dict])
    
    # Ensure all expected columns are present and in correct order
    expected_cols = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else df.columns
    
    # Reorder columns to match model expectations
    df = df[expected_cols]
    
    # Apply scaling
    df_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    return df_scaled

# Endpoint root
@app.get("/")
def read_root():
    return {"message": "✅ Product Category Prediction API is running"}

# Endpoint prediksi
@app.post("/predict")
def predict_category(data: InputData):
    try:
        processed = preprocess_input(data)
        prediction = model.predict(processed)[0]
        
        # Get prediction probability if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(processed)[0]
            confidence = round(max(proba) * 100, 2)
        
        label_map = {0: "Sedikit", 1: "Sedang", 2: "Banyak"}
        
        return {
            "predicted_category": label_map.get(prediction, "Unknown"),
            "model_used": type(model).__name__,
            "confidence": confidence,
            # You can add historical and predicted data here if available
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")