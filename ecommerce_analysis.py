import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
import joblib
import pickle
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Set style for better visualizations
sns.set_style('whitegrid')
sns.set_palette("husl")

def load_data(file_path):
    """Load the e-commerce dataset"""
    print("Loading data...")
    df = pd.read_excel(file_path)
    return df

def clean_data(df):
    """Clean and preprocess the data"""
    print("Cleaning data...")
    
    # Remove rows with missing values
    df = df.dropna()
    
    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Remove canceled orders (negative quantity)
    df = df[df['Quantity'] > 0]
    
    # Calculate total amount for each transaction
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    
    return df

def analyze_revenue_over_time(df):
    """Analyze revenue trends over time"""
    print("Analyzing revenue over time...")
    
    # Monthly revenue
    monthly_revenue = df.groupby(df['InvoiceDate'].dt.to_period('M'))['TotalAmount'].sum()
    
    plt.figure(figsize=(12, 6))
    monthly_revenue.plot(kind='line', marker='o')
    plt.title('Monthly Revenue Over Time')
    plt.xlabel('Month')
    plt.ylabel('Revenue')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('monthly_revenue.png')
    plt.close()

def analyze_top_products(df):
    """Analyze most purchased products"""
    print("Analyzing top products...")
    
    # Top 10 products by quantity
    top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    top_products.plot(kind='bar')
    plt.title('Top 10 Products by Quantity')
    plt.xlabel('Product')
    plt.ylabel('Quantity')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('top_products.png')
    plt.close()

def analyze_country_distribution(df):
    """Analyze revenue distribution by country"""
    print("Analyzing country distribution...")
    
    country_revenue = df.groupby('Country')['TotalAmount'].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 10))
    plt.pie(country_revenue, labels=country_revenue.index, autopct='%1.1f%%')
    plt.title('Revenue Distribution by Country')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('country_distribution.png')
    plt.close()

def analyze_customer_patterns(df):
    """Analyze customer purchase patterns"""
    print("Analyzing customer patterns...")
    
    # Extract time components
    df['Hour'] = df['InvoiceDate'].dt.hour
    df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
    
    # Create heatmap of purchase patterns
    pivot_table = df.pivot_table(
        values='TotalAmount',
        index='DayOfWeek',
        columns='Hour',
        aggfunc='sum'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt='.0f')
    plt.title('Purchase Patterns by Day and Hour')
    plt.tight_layout()
    plt.savefig('purchase_patterns.png')
    plt.close()

def add_engineered_features(df):
    print("Adding engineered features...")

    # RFM features
    latest_date = df['InvoiceDate'].max()
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (latest_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalAmount': 'Monetary'
    }).reset_index()

    # Merge RFM back to main df
    df = df.merge(rfm, on='CustomerID', how='left')

    # Weekend feature
    df['IsWeekend'] = df['InvoiceDate'].dt.dayofweek >= 5

    # Time of day
    df['Hour'] = df['InvoiceDate'].dt.hour
    df['TimeOfDay'] = pd.cut(df['Hour'],
                              bins=[0, 6, 12, 18, 24],
                              labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                              right=False)

    return df

def predict_future_sales(df):
    """Predict future sales using enhanced XGBoost with parameter tuning and lag features"""
    print("Predicting future sales...")
    
    # Prepare features
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['Day'] = df['InvoiceDate'].dt.day
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    df['Quarter'] = df['InvoiceDate'].dt.quarter
    df['WeekOfYear'] = df['InvoiceDate'].dt.isocalendar().week
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Fix holiday check
    holiday_dates = pd.date_range('2011-12-24', '2011-12-26')
    df['IsHoliday'] = df['InvoiceDate'].dt.date.isin(holiday_dates.date).astype(int)
    
    # Aggregate daily sales
    daily_sales = df.groupby(['Year', 'Month', 'Day'])['TotalAmount'].sum().reset_index()
    daily_sales['Date'] = pd.to_datetime(daily_sales[['Year', 'Month', 'Day']])
    daily_sales['DayOfWeek'] = daily_sales['Date'].dt.dayofweek
    daily_sales['Quarter'] = daily_sales['Date'].dt.quarter
    daily_sales['WeekOfYear'] = daily_sales['Date'].dt.isocalendar().week
    daily_sales['IsWeekend'] = daily_sales['DayOfWeek'].isin([5, 6]).astype(int)
    daily_sales['IsHoliday'] = daily_sales['Date'].dt.date.isin(holiday_dates.date).astype(int)
    
    # Sort by date
    daily_sales = daily_sales.sort_values(by='Date')
    
    # Add lag features
    for lag in [1, 2, 3, 7, 14, 30]:
        daily_sales[f'Lag_{lag}'] = daily_sales['TotalAmount'].shift(lag)
    
    # Add rolling mean features
    for window in [3, 7, 14, 30]:
        daily_sales[f'Rolling_Mean_{window}'] = daily_sales['TotalAmount'].rolling(window=window).mean()
        daily_sales[f'Rolling_Std_{window}'] = daily_sales['TotalAmount'].rolling(window=window).std()
    
    # Drop NaNs caused by lagging and rolling
    daily_sales = daily_sales.dropna()
    
    # Define features and target
    feature_columns = ['Year', 'Month', 'Day', 'DayOfWeek', 'Quarter', 'WeekOfYear', 
                      'IsWeekend', 'IsHoliday'] + \
                     [f'Lag_{i}' for i in [1, 2, 3, 7, 14, 30]] + \
                     [f'Rolling_Mean_{i}' for i in [3, 7, 14, 30]] + \
                     [f'Rolling_Std_{i}' for i in [3, 7, 14, 30]]
    
    X = daily_sales[feature_columns]
    y = daily_sales['TotalAmount']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Enhanced XGBoost with Grid Search
    param_grid = {
        'n_estimators': [200, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=5, 
        scoring='r2', 
        verbose=1, 
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Predict and evaluate
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

def forecast_with_prophet(df):
    """Enhanced forecasting with Prophet"""
    print("Forecasting with Prophet...")

    # Prepare daily aggregated data
    df['Date'] = df['InvoiceDate'].dt.date
    daily = df.groupby('Date')['TotalAmount'].sum().reset_index()
    daily.columns = ['ds', 'y']

    # Add additional regressors
    daily['is_weekend'] = pd.to_datetime(daily['ds']).dt.dayofweek.isin([5, 6]).astype(int)
    holiday_dates = pd.date_range('2011-12-24', '2011-12-26')
    daily['is_holiday'] = pd.to_datetime(daily['ds']).dt.date.isin(holiday_dates.date).astype(int)
    daily['month'] = pd.to_datetime(daily['ds']).dt.month
    daily['quarter'] = pd.to_datetime(daily['ds']).dt.quarter

    # Create holidays dataframe
    holidays = pd.DataFrame({
        'holiday': 'christmas',
        'ds': pd.to_datetime(['2011-12-24', '2011-12-25', '2011-12-26']),
        'lower_window': -1,
        'upper_window': 1,
    })

    # Initialize and fit the model with enhanced parameters
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        seasonality_mode='multiplicative',
        holidays=holidays  # Pass holidays directly in constructor
    )

    # Add regressors
    model.add_regressor('is_weekend')
    model.add_regressor('is_holiday')
    model.add_regressor('month')
    model.add_regressor('quarter')

    # Add custom seasonality
    model.add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=5
    )

    # Fit model
    model.fit(daily)

    # Create future dataframe
    future = model.make_future_dataframe(periods=30)
    future['is_weekend'] = future['ds'].dt.dayofweek.isin([5, 6]).astype(int)
    future['is_holiday'] = future['ds'].dt.date.isin(holiday_dates.date).astype(int)
    future['month'] = future['ds'].dt.month
    future['quarter'] = future['ds'].dt.quarter

    # Make predictions
    forecast = model.predict(future)

    # Plot the forecast
    fig = model.plot(forecast)
    plt.title('Enhanced Sales Forecast (30 Days)')
    plt.savefig('prophet_forecast.png')
    plt.close()

    # Plot components
    fig2 = model.plot_components(forecast)
    plt.savefig('prophet_components.png')
    plt.close()

    # Calculate and print metrics
    rmse = np.sqrt(mean_squared_error(daily['y'], forecast['yhat'][:len(daily)]))
    mae = np.mean(np.abs(daily['y'] - forecast['yhat'][:len(daily)]))
    
    print("\nProphet Forecast Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    # Print forecast for next 30 days
    print("\nForecast for next 30 days:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))

    return model

def segment_customers(df):
    """Segment customers using RFM analysis and K-means clustering"""
    print("Segmenting customers using RFM...")

    # Calculate RFM metrics
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (df['InvoiceDate'].max() - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalAmount': 'Monetary'
    })

    # Normalize RFM values
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    # Apply KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Calculate cluster statistics
    cluster_stats = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).round(2)

    # Plot Clusters
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Recency vs Monetary
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Cluster', palette='Set2')
    plt.title('Recency vs Monetary Value')
    
    # Plot 2: Frequency vs Monetary
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=rfm, x='Frequency', y='Monetary', hue='Cluster', palette='Set2')
    plt.title('Frequency vs Monetary Value')
    
    # Plot 3: Recency vs Frequency
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=rfm, x='Recency', y='Frequency', hue='Cluster', palette='Set2')
    plt.title('Recency vs Frequency')
    
    # Plot 4: Cluster Sizes
    plt.subplot(2, 2, 4)
    cluster_sizes = rfm['Cluster'].value_counts().sort_index()
    sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values, palette='Set2')
    plt.title('Cluster Sizes')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Customers')

    plt.tight_layout()
    plt.savefig('customer_segments.png')
    plt.close()

    # Print cluster characteristics
    print("\nCluster Characteristics:")
    print(cluster_stats)
    
    # Assign segment names based on RFM values
    segment_names = {
        0: 'High-Value Loyal Customers',
        1: 'At-Risk Customers',
        2: 'New Customers',
        3: 'Regular Customers'
    }
    
    rfm['Segment'] = rfm['Cluster'].map(segment_names)
    
    # Print segment distribution
    print("\nCustomer Segment Distribution:")
    print(rfm['Segment'].value_counts())
    
    return rfm

def predict_with_stacked_model(df):
    """Predict future sales using enhanced stacked model with sophisticated feature engineering"""
    print("Predicting with stacked model...")
    
    # Prepare basic time features
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['Day'] = df['InvoiceDate'].dt.day
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    df['Quarter'] = df['InvoiceDate'].dt.quarter
    df['WeekOfYear'] = df['InvoiceDate'].dt.isocalendar().week
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Enhanced holiday features
    holiday_dates = pd.date_range('2011-12-24', '2011-12-26')
    df['IsHoliday'] = df['InvoiceDate'].dt.date.isin(holiday_dates.date).astype(int)
    df['DaysToHoliday'] = (holiday_dates[0] - df['InvoiceDate']).dt.days
    df['DaysToHoliday'] = df['DaysToHoliday'].apply(lambda x: min(abs(x), 7) if x <= 7 else 0)
    
    # Aggregate daily sales
    daily_sales = df.groupby(['Year', 'Month', 'Day'])['TotalAmount'].sum().reset_index()
    daily_sales['Date'] = pd.to_datetime(daily_sales[['Year', 'Month', 'Day']])
    daily_sales['DayOfWeek'] = daily_sales['Date'].dt.dayofweek
    daily_sales['Quarter'] = daily_sales['Date'].dt.quarter
    daily_sales['WeekOfYear'] = daily_sales['Date'].dt.isocalendar().week
    daily_sales['IsWeekend'] = daily_sales['DayOfWeek'].isin([5, 6]).astype(int)
    daily_sales['IsHoliday'] = daily_sales['Date'].dt.date.isin(holiday_dates.date).astype(int)
    daily_sales['DaysToHoliday'] = (holiday_dates[0] - daily_sales['Date']).dt.days
    daily_sales['DaysToHoliday'] = daily_sales['DaysToHoliday'].apply(lambda x: min(abs(x), 7) if x <= 7 else 0)
    
    # Sort by date
    daily_sales = daily_sales.sort_values(by='Date')
    
    # Create a copy to avoid fragmentation warnings
    daily_sales = daily_sales.copy()
    
    # Add seasonal features
    daily_sales['Month_Sin'] = np.sin(2 * np.pi * daily_sales['Month']/12)
    daily_sales['Month_Cos'] = np.cos(2 * np.pi * daily_sales['Month']/12)
    daily_sales['DayOfWeek_Sin'] = np.sin(2 * np.pi * daily_sales['DayOfWeek']/7)
    daily_sales['DayOfWeek_Cos'] = np.cos(2 * np.pi * daily_sales['DayOfWeek']/7)
    daily_sales['WeekOfYear_Sin'] = np.sin(2 * np.pi * daily_sales['WeekOfYear']/52)
    daily_sales['WeekOfYear_Cos'] = np.cos(2 * np.pi * daily_sales['WeekOfYear']/52)
    
    # Add trend features
    daily_sales['Trend'] = np.arange(len(daily_sales))
    daily_sales['Trend_Squared'] = daily_sales['Trend'] ** 2
    
    # Add lag features
    for lag in [1, 2, 3, 7, 30]:
        daily_sales[f'Lag_{lag}'] = daily_sales['TotalAmount'].shift(lag)
        daily_sales[f'Lag_{lag}_Ratio'] = daily_sales['TotalAmount'] / daily_sales[f'Lag_{lag}'].replace(0, 1)
        if lag <= 3:
            daily_sales[f'Lag_{lag}_Diff'] = daily_sales['TotalAmount'] - daily_sales[f'Lag_{lag}']
            daily_sales[f'Lag_{lag}_Pct_Change'] = (daily_sales['TotalAmount'] - daily_sales[f'Lag_{lag}']) / daily_sales[f'Lag_{lag}'].replace(0, 1)
    
    # Add rolling features
    for window in [3, 7, 14]:
        daily_sales[f'Rolling_Mean_{window}'] = daily_sales['TotalAmount'].rolling(window=window, min_periods=1).mean()
        if window <= 7:
            daily_sales[f'Rolling_Std_{window}'] = daily_sales['TotalAmount'].rolling(window=window, min_periods=1).std()
            daily_sales[f'Rolling_Min_{window}'] = daily_sales['TotalAmount'].rolling(window=window, min_periods=1).min()
            daily_sales[f'Rolling_Max_{window}'] = daily_sales['TotalAmount'].rolling(window=window, min_periods=1).max()
            daily_sales[f'Rolling_Range_{window}'] = daily_sales[f'Rolling_Max_{window}'] - daily_sales[f'Rolling_Min_{window}']
            if window == 3:
                daily_sales[f'Rolling_Median_{window}'] = daily_sales['TotalAmount'].rolling(window=window, min_periods=1).median()
                daily_sales[f'Rolling_Skew_{window}'] = daily_sales['TotalAmount'].rolling(window=window, min_periods=1).skew()
    
    # Add interaction features
    daily_sales['Weekend_Holiday'] = daily_sales['IsWeekend'] * daily_sales['IsHoliday']
    daily_sales['Weekend_Lag1_Ratio'] = daily_sales['IsWeekend'] * daily_sales['Lag_1_Ratio']
    daily_sales['Holiday_Lag1_Ratio'] = daily_sales['IsHoliday'] * daily_sales['Lag_1_Ratio']
    
    # Fill NaN values
    daily_sales = daily_sales.fillna(0)
    
    # Define features and target
    feature_columns = [
        # Time features
        'Year', 'Month', 'Day', 'DayOfWeek', 'Quarter', 'WeekOfYear',
        'IsWeekend', 'IsHoliday', 'DaysToHoliday',
        
        # Seasonal features
        'Month_Sin', 'Month_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos',
        'WeekOfYear_Sin', 'WeekOfYear_Cos',
        
        # Trend features
        'Trend', 'Trend_Squared',
        
        # Lag features
        'Lag_1', 'Lag_2', 'Lag_3', 'Lag_7', 'Lag_30',
        'Lag_1_Ratio', 'Lag_2_Ratio', 'Lag_3_Ratio', 'Lag_7_Ratio', 'Lag_30_Ratio',
        'Lag_1_Diff', 'Lag_2_Diff', 'Lag_3_Diff',
        'Lag_1_Pct_Change', 'Lag_2_Pct_Change', 'Lag_3_Pct_Change',
        
        # Rolling features
        'Rolling_Mean_3', 'Rolling_Mean_7', 'Rolling_Mean_14',
        'Rolling_Std_3', 'Rolling_Std_7',
        'Rolling_Min_3', 'Rolling_Max_3',
        'Rolling_Range_3', 'Rolling_Range_7',
        'Rolling_Median_3', 'Rolling_Skew_3',
        
        # Interaction features
        'Weekend_Holiday', 'Weekend_Lag1_Ratio', 'Holiday_Lag1_Ratio'
    ]
    
    X = daily_sales[feature_columns]
    y = daily_sales['TotalAmount']
    
    # Create and fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Define base models with simpler configurations
    base_models = [
        ('xgb', XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )),
        ('rf', RandomForestRegressor(
            n_estimators=100,
            max_depth=3,
            random_state=42
        )),
        ('lr', LinearRegression())
    ]
    
    # Create stacking model
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=LinearRegression(),
        cv=3
    )
    
    # Train stacking model
    stacking_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = stacking_model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nEnhanced Stacked Model Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    
    return stacking_model, feature_columns, scaler

def save_models(stacked_model, prophet_model, scaler):
    """Save trained models to disk"""
    print("Saving models...")
    
    # Save stacked model
    joblib.dump(stacked_model, 'models/stacked_model.joblib')
    
    # Save Prophet model
    with open('models/prophet_model.pkl', 'wb') as f:
        pickle.dump(prophet_model, f)
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.joblib')
    
    print("Models saved successfully!")

def load_models():
    """Load trained models from disk"""
    print("Loading models...")
    
    try:
        # Load stacked model
        stacked_model = joblib.load('models/stacked_model.joblib')
        
        # Load Prophet model
        with open('models/prophet_model.pkl', 'rb') as f:
            prophet_model = pickle.load(f)
        
        # Load scaler
        scaler = joblib.load('models/scaler.joblib')
        
        print("Models loaded successfully!")
        return stacked_model, prophet_model, scaler
    
    except FileNotFoundError:
        print("Error: Model files not found. Please train the models first.")
        return None, None, None

def main():
    # Load and clean data
    df = load_data('Online Retail.xlsx')
    df = clean_data(df)
    
    # Perform analysis
    analyze_revenue_over_time(df)
    analyze_top_products(df)
    analyze_country_distribution(df)
    analyze_customer_patterns(df)
    predict_future_sales(df)
    
    # Add feature engineering
    df = add_engineered_features(df)
    
    # Add Prophet forecasting
    prophet_model = forecast_with_prophet(df)
    
    # Add customer segmentation
    customer_segments = segment_customers(df)
    
    # Add stacked model prediction
    stacked_model, feature_columns, scaler = predict_with_stacked_model(df)
    
    # Save models
    save_models(stacked_model, prophet_model, scaler)
    
    print("Analysis complete! Check the generated plots in the current directory.")

if __name__ == "__main__":
    main() 