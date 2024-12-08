import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime, timedelta

# --------------------------
# Step 1: Load and Clean the Citibike Dataset
# --------------------------

# Load the Citibike dataset
citibike_df = pd.read_csv("data/cleaned_citibike.csv")

# Convert starttime to datetime
citibike_df['starttime'] = pd.to_datetime(citibike_df['starttime'])
citibike_df['date'] = citibike_df['starttime'].dt.date

# --------------------------
# Step 2: Load and Clean the Crash Dataset
# --------------------------

# Load the crash dataset
crash_df = pd.read_csv("data/accidents.csv")

# Use only January crashes
crash_df['CRASH DATE'] = pd.to_datetime(crash_df['CRASH DATE'])
crash_df = crash_df[crash_df['CRASH DATE'].dt.month == 1]
crash_df['CRASH TIME'] = pd.to_datetime(crash_df['CRASH TIME'], format='%H:%M:%S', errors='coerce').dt.time

# Create a datetime column for crash events
crash_df['CRASH DATETIME'] = pd.to_datetime(crash_df['CRASH DATE'].astype(str) + ' ' + crash_df['CRASH TIME'].astype(str), errors='coerce')

# --------------------------
# Step 3: Load the Weather Dataset
# --------------------------

# Load the weather dataset
weather_df = pd.read_csv("data/cleaned_weather.csv")

# Convert 'date' column to datetime
weather_df['date'] = pd.to_datetime(weather_df['date'])

# --------------------------
# Step 4: Merge All Datasets
# --------------------------

# Ensure 'date' column is in datetime format for all datasets
citibike_df['date'] = pd.to_datetime(citibike_df['date'])
weather_df['date'] = pd.to_datetime(weather_df['date'])

# Merge weather data into Citibike data
citibike_weather_df = pd.merge(citibike_df, weather_df, on='date', how='left')

# --------------------------
# Step 5: Fill Missing Data After Merging
# --------------------------

# List of columns with missing data
missing_cols = ['tmax','tmin','tavg', 'departure', 'HDD', 'CDD', 'precipitation', 'new_snow', 'snow_depth']

# Fill missing values using forward-fill, backward-fill, and mean imputation
for col in missing_cols:
    citibike_weather_df[col] = citibike_weather_df[col].replace('T', 0).astype(float)  # Replace 'T' with 0
    citibike_weather_df[col].fillna(method='ffill', inplace=True)  # Forward fill
    citibike_weather_df[col].fillna(method='bfill', inplace=True)  # Backward fill
    citibike_weather_df[col].fillna(citibike_weather_df[col].mean(), inplace=True)  # Mean imputation

# --------------------------
# Step 6: Calculate Nearby Crashes with Haversine Formula
# --------------------------

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance between two points."""
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def count_nearby_crashes(row, crash_df, radius_km=5, time_window_hours=2):
    """Count crashes near a given station within a time window."""
    start_lat, start_lon = row['start station latitude'], row['start station longitude']
    start_time = row['starttime']
    time_window_start = start_time - pd.Timedelta(hours=time_window_hours)
    time_window_end = start_time + pd.Timedelta(hours=time_window_hours)
    
    # Filter crashes within time window
    crashes_in_time = crash_df[
        (crash_df['CRASH DATETIME'] >= time_window_start) &
        (crash_df['CRASH DATETIME'] <= time_window_end)
    ]
    
    # Calculate distances and count nearby crashes
    count = 0
    for _, crash_row in crashes_in_time.iterrows():
        crash_lat, crash_lon = crash_row['LATITUDE'], crash_row['LONGITUDE']
        if not np.isnan(crash_lat) and not np.isnan(crash_lon):
            distance = haversine(start_lat, start_lon, crash_lat, crash_lon)
            if distance <= radius_km:
                count += 1
    return count

# Apply the nearby crash calculation
citibike_weather_df['nearby_crashes'] = citibike_weather_df.apply(
    lambda row: count_nearby_crashes(row, crash_df), axis=1
)

# --------------------------
# Step 7: Save the Final Combined Dataset
# --------------------------

citibike_weather_df.to_csv("data/bigdata.csv", index=False)
print("Final combined dataset with filled weather data and nearby crashes saved as 'bigdata.csv'")