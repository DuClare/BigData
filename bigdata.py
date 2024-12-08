import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix

# Load the data
df = pd.read_csv('bigdata.csv')

# Enhanced Exploratory Data Analysis
def enhanced_exploratory_analysis(df):
    # Weather and Crash Impact Analysis
    print("Weather and Nearby Crashes Summary:")
    weather_crash_cols = ['tmax', 'tmin', 'tavg', 'HDD', 'CDD', 'precipitation', 'new_snow', 'snow_depth', 'nearby_crashes']
    print(df[weather_crash_cols].describe())
    
    # Correlation with Nearby Crashes
    plt.figure(figsize=(12,8))
    crash_correlation = df[['nearby_crashes', 'ride_duration', 'distance_km', 'tmax', 'tmin', 'precipitation']].corr()
    sns.heatmap(crash_correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation with Nearby Crashes')
    plt.tight_layout()
    plt.savefig('crash_correlation_heatmap.png')
    plt.close()

# Enhanced Weather Impact Visualization
def weather_impact_analysis(df):
    # Prepare multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(20,12))
    fig.suptitle('Weather Conditions Impact on Bike Sharing', fontsize=16)
    
    # 1. Ride Duration vs Average Temperature
    sns.scatterplot(x='tavg', y='ride_duration', hue='nearby_crashes', data=df, ax=axes[0,0])
    axes[0,0].set_title('Ride Duration vs Average Temperature')
    axes[0,0].set_xlabel('Average Temperature')
    axes[0,0].set_ylabel('Ride Duration')
    
    # 2. Ride Distance vs Precipitation
    sns.boxplot(x=pd.cut(df['precipitation'], bins=5), y='distance_km', data=df, ax=axes[0,1])
    axes[0,1].set_title('Ride Distance by Precipitation Levels')
    axes[0,1].set_xlabel('Precipitation Levels')
    axes[0,1].set_ylabel('Ride Distance (km)')
    axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=45)
    
    # 3. Ride Count by Heating Degree Days (HDD)
    df['HDD_bins'] = pd.cut(df['HDD'], bins=5)
    hdd_ride_count = df.groupby('HDD_bins')['tripduration'].count()
    hdd_ride_count.plot(kind='bar', ax=axes[0,2])
    axes[0,2].set_title('Ride Frequency by Heating Degree Days')
    axes[0,2].set_xlabel('Heating Degree Days')
    axes[0,2].set_ylabel('Number of Rides')
    
    # 4. Crash Impact on Ride Characteristics
    crash_grouped = df.groupby('nearby_crashes')[['ride_duration', 'distance_km']].mean()
    crash_grouped.plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title('Average Ride Metrics by Nearby Crashes')
    axes[1,0].set_xlabel('Number of Nearby Crashes')
    
    # 5. Snow Depth Impact
    sns.scatterplot(x='snow_depth', y='ride_duration', hue='usertype', data=df, ax=axes[1,1])
    axes[1,1].set_title('Ride Duration vs Snow Depth')
    
    # 6. Temperature Variation Impact
    df['temp_range'] = df['tmax'] - df['tmin']
    sns.boxplot(x='usertype', y='temp_range', data=df, ax=axes[1,2])
    axes[1,2].set_title('Temperature Range by User Type')
    
    plt.tight_layout()
    plt.savefig('weather_impact_analysis.png')
    plt.close()

# Advanced Predictive Modeling
def advanced_predictive_modeling(df):
    # Prepare features
    features = [
        'distance_km', 'tavg', 'tmax', 'tmin', 
        'HDD', 'CDD', 'precipitation', 
        'new_snow', 'snow_depth', 
        'nearby_crashes', 'gender'
    ]
    
    # Separate numeric and categorical features
    numeric_features = [
        'distance_km', 'tavg', 'tmax', 'tmin', 
        'HDD', 'CDD', 'precipitation', 
        'new_snow', 'snow_depth', 'nearby_crashes'
    ]
    categorical_features = ['gender']
    
    # Prepare X and y
    X = df[features]
    y_duration = df['ride_duration']
    y_usertype = (df['usertype'] == 'Subscriber').astype(int)
    
    # Split data
    X_train_duration, X_test_duration, y_train_duration, y_test_duration = train_test_split(
        X, y_duration, test_size=0.2, random_state=42)
    
    X_train_usertype, X_test_usertype, y_train_usertype, y_test_usertype = train_test_split(
        X, y_usertype, test_size=0.2, random_state=42)
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Ride Duration Prediction Pipeline
    duration_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # User Type Prediction Pipeline
    usertype_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    # Fit and evaluate Duration Prediction
    duration_pipeline.fit(X_train_duration, y_train_duration)
    y_pred_duration = duration_pipeline.predict(X_test_duration)
    
    print("Ride Duration Prediction Results:")
    print(f"Mean Squared Error: {mean_squared_error(y_test_duration, y_pred_duration)}")
    print(f"R² Score: {r2_score(y_test_duration, y_pred_duration)}")
    
    # Cross-validation for robustness
    duration_cv_scores = cross_val_score(duration_pipeline, X, y_duration, cv=5, scoring='neg_mean_squared_error')
    print("Cross-validation MSE scores:", -duration_cv_scores)
    
    # Fit and evaluate User Type Prediction
    usertype_pipeline.fit(X_train_usertype, y_train_usertype)
    y_pred_usertype = usertype_pipeline.predict(X_test_usertype)
    
    print("\nUser Type Classification Results:")
    print(classification_report(y_test_usertype, y_pred_usertype))
    
    # Feature Importance for Duration Prediction
    feature_names = (
        numeric_features + 
        list(usertype_pipeline.named_steps['preprocessor']
             .named_transformers_['cat']
             .get_feature_names_out(categorical_features))
    )
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': duration_pipeline.named_steps['regressor'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12,6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Features Importance for Ride Duration')
    plt.tight_layout()
    plt.savefig('advanced_feature_importance.png')
    plt.close()

# Main Execution
def main():
    enhanced_exploratory_analysis(df)
    weather_impact_analysis(df)
    advanced_predictive_modeling(df)

if __name__ == "__main__":
    main()