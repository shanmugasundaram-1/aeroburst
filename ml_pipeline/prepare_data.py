import pandas as pd
import numpy as np
import os

DATA_PATH = "../data/india_2000_2024_daily_weather.csv"
OUTPUT_PATH = "../data/processed_aerobust.csv"

def synthesize_forecast(actual_rain):
    """
    Since we only have actuals, this function mathematically fakes what a 
    NWP weather model would have predicted, introducing realistic noise and rare 'busts'.
    """
    # Base noise (usually models get it somewhat close)
    noise = np.random.normal(loc=0.0, scale=3.0, size=len(actual_rain))
    forecast = actual_rain + noise
    forecast = np.clip(forecast, 0, None)  # Prevent negative rainfall forecasts
    
    # Inject Artificial "Forecast Busts" (Where the model completely failed)
    # E.g. 5% of the time, the model is totally wrong by ~20mm
    bust_mask = np.random.rand(len(actual_rain)) < 0.05
    forecast[bust_mask] = np.abs(forecast[bust_mask] - 20.0) 
    
    return forecast

def process_data():
    print(f"Loading immense historical dataset: {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print("Data file not found. Ensure the path is correct.")
        return
    
    # Clean the data by filling missing values
    df.fillna(0, inplace=True)
    
    print("Synthesizing Forecast baseline to pair with actuals...")
    df['forecast_rain_sum'] = synthesize_forecast(df['rain_sum'].values)
    
    # Calculate the Error representing how badly the model missed
    df['forecast_error_mm'] = np.abs(df['rain_sum'] - df['forecast_rain_sum'])
    
    # Convert absolute error to a Probability of a 'Bust' (0 to 100%)
    # Let's say a 15mm error means a 100% bust prediction
    df['bust_probability'] = np.clip((df['forecast_error_mm'] / 15.0) * 100.0, 0, 100).astype(int)
    
    # Keep only the features relevant for the ML algorithm
    features = [
        'city', 'date', 'temperature_2m_max', 'wind_speed_10m_max', 
        'rain_sum', 'forecast_rain_sum', 'forecast_error_mm', 'bust_probability'
    ]
    
    processed_df = df[features]
    processed_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"SUCCESS: Synthesized {len(processed_df)} forecast/actual paired data points.")
    print(f"Saved processed ML dataset to: {OUTPUT_PATH}")

if __name__ == '__main__':
    # Initialize random seeder to make sure bugs are reproducible for the hackathon
    np.random.seed(42)
    process_data()
