import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

def load_data(filepath='Online Retail.xlsx'):
    df = pd.read_excel(filepath)
    return df

def clean_data(df):
    df = df.dropna(subset=['CustomerID'])
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    return df

def load_models():
    stacked_model = joblib.load(os.path.join('models', 'stacked_model.joblib'))
    prophet_model = joblib.load(os.path.join('models', 'prophet_model.pkl'))
    scaler = joblib.load(os.path.join('models', 'scaler.joblib'))
    return stacked_model, prophet_model, scaler

def get_regions(df):
    return ['All'] + sorted(df['Country'].unique().tolist())

def get_metrics(df):
    return [
        {'label': 'Total Revenue', 'value': f"${df['TotalAmount'].sum():,.2f}"},
        {'label': 'Total Orders', 'value': f"{df['InvoiceNo'].nunique():,}"},
        {'label': 'Avg Order Value', 'value': f"${(df['TotalAmount'].sum()/df['InvoiceNo'].nunique()):,.2f}"},
        {'label': 'Total Customers', 'value': f"{df['CustomerID'].nunique():,}"},
    ]

def get_revenue_trend_plot(df):
    daily_revenue = df.groupby(df['InvoiceDate'].dt.date)['TotalAmount'].sum().reset_index()
    fig = px.line(daily_revenue, x='InvoiceDate', y='TotalAmount', title='', template='plotly_white', markers=True)
    fig.update_traces(line=dict(color='#00b894', width=3))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=350, hovermode='x unified')
    fig.update_xaxes(rangeslider_visible=True)
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn', div_id='revenue-plot')

def get_top_products_plot(df):
    top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10).reset_index()
    fig = px.bar(top_products, x='Description', y='Quantity', title='', template='plotly_white', color='Quantity', color_continuous_scale='Viridis')
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300, coloraxis_showscale=False, hovermode='x')
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn', div_id='top-products-plot')

def get_country_revenue_plot(df):
    country_revenue = df.groupby('Country')['TotalAmount'].sum().sort_values(ascending=False).head(10).reset_index()
    fig = px.pie(country_revenue, values='TotalAmount', names='Country', title='', template='plotly_white', hole=0.4)
    fig.update_traces(textinfo='percent+label', pull=[0.1]+[0]*9)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn', div_id='country-revenue-plot')

def get_customer_patterns_plot(df):
    df['Hour'] = df['InvoiceDate'].dt.hour
    df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
    pivot_table = df.pivot_table(values='TotalAmount', index='DayOfWeek', columns='Hour', aggfunc='sum').fillna(0)
    fig = px.imshow(pivot_table, title='', template='plotly_white', aspect='auto', color_continuous_scale='Blues')
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn', div_id='customer-patterns-plot')

def build_forecast_features(df, future_dates):
    # Compute the trend (days since first date)
    first_date = df['InvoiceDate'].min()
    trend = (future_dates - first_date).days
    trend_squared = trend ** 2

    # Days to next holiday (example: Christmas)
    holidays = pd.to_datetime(['2011-12-25'])
    days_to_holiday = future_dates.map(lambda d: np.min(np.abs((holidays - d).days)))

    # Cyclical features
    month_sin = np.sin(2 * np.pi * future_dates.month / 12)
    month_cos = np.cos(2 * np.pi * future_dates.month / 12)
    dayofweek_sin = np.sin(2 * np.pi * future_dates.dayofweek / 7)
    dayofweek_cos = np.cos(2 * np.pi * future_dates.dayofweek / 7)
    weekofyear = future_dates.isocalendar().week.to_numpy()
    weekofyear_sin = np.sin(2 * np.pi * weekofyear / 52)
    weekofyear_cos = np.cos(2 * np.pi * weekofyear / 52)

    # Basic time features
    features = pd.DataFrame({
        'Year': future_dates.year,
        'Month': future_dates.month,
        'Day': future_dates.day,
        'DayOfWeek': future_dates.dayofweek,
        'Quarter': future_dates.quarter,
        'WeekOfYear': weekofyear,
        'IsWeekend': future_dates.dayofweek.isin([5, 6]).astype(int),
        'IsHoliday': future_dates.isin(holidays).astype(int),
        'DaysToHoliday': days_to_holiday,
        'Month_Sin': month_sin,
        'Month_Cos': month_cos,
        'DayOfWeek_Sin': dayofweek_sin,
        'DayOfWeek_Cos': dayofweek_cos,
        'WeekOfYear_Sin': weekofyear_sin,
        'WeekOfYear_Cos': weekofyear_cos,
        'Trend': trend,
        'Trend_Squared': trend_squared,
    })

    # Use last available values for lag/rolling features
    last = df['TotalAmount']
    for lag in [1, 2, 3, 7, 30]:
        features[f'Lag_{lag}'] = last.iloc[-lag] if len(last) >= lag else last.iloc[-1]
    for lag in [1, 2, 3, 7, 30]:
        # Ratio: current/lag (use last value for both)
        curr = last.iloc[-1]
        prev = last.iloc[-lag] if len(last) >= lag else last.iloc[-1]
        features[f'Lag_{lag}_Ratio'] = curr / prev if prev != 0 else 0
    for lag in [1, 2, 3]:
        curr = last.iloc[-1]
        prev = last.iloc[-lag] if len(last) >= lag else last.iloc[-1]
        features[f'Lag_{lag}_Diff'] = curr - prev
        features[f'Lag_{lag}_Pct_Change'] = (curr - prev) / prev if prev != 0 else 0
    # Rolling features
    for window in [3, 7, 14]:
        features[f'Rolling_Mean_{window}'] = last.iloc[-window:].mean() if len(last) >= window else last.mean()
    for window in [3, 7]:
        features[f'Rolling_Std_{window}'] = last.iloc[-window:].std() if len(last) >= window else last.std()
    for window in [3]:
        features[f'Rolling_Min_{window}'] = last.iloc[-window:].min() if len(last) >= window else last.min()
        features[f'Rolling_Max_{window}'] = last.iloc[-window:].max() if len(last) >= window else last.max()
        features[f'Rolling_Range_{window}'] = features[f'Rolling_Max_{window}'] - features[f'Rolling_Min_{window}']
        features[f'Rolling_Median_{window}'] = last.iloc[-window:].median() if len(last) >= window else last.median()
        features[f'Rolling_Skew_{window}'] = last.iloc[-window:].skew() if len(last) >= window else 0
    for window in [7]:
        features[f'Rolling_Range_{window}'] = last.iloc[-window:].max() - last.iloc[-window:].min() if len(last) >= window else last.max() - last.min()
    # Weekend/Holiday interaction
    features['Weekend_Holiday'] = features['IsWeekend'] * features['IsHoliday']
    # Weekend_Lag1_Ratio: IsWeekend * Lag_1_Ratio
    features['Weekend_Lag1_Ratio'] = features['IsWeekend'] * features['Lag_1_Ratio']
    # Holiday_Lag1_Ratio: IsHoliday * Lag_1_Ratio
    features['Holiday_Lag1_Ratio'] = features['IsHoliday'] * features['Lag_1_Ratio']

    # Ensure column order matches scaler/model
    feature_columns = [
        'Year', 'Month', 'Day', 'DayOfWeek', 'Quarter', 'WeekOfYear', 'IsWeekend',
        'IsHoliday', 'DaysToHoliday', 'Month_Sin', 'Month_Cos', 'DayOfWeek_Sin',
        'DayOfWeek_Cos', 'WeekOfYear_Sin', 'WeekOfYear_Cos', 'Trend', 'Trend_Squared',
        'Lag_1', 'Lag_2', 'Lag_3', 'Lag_7', 'Lag_30', 'Lag_1_Ratio', 'Lag_2_Ratio',
        'Lag_3_Ratio', 'Lag_7_Ratio', 'Lag_30_Ratio', 'Lag_1_Diff', 'Lag_2_Diff',
        'Lag_3_Diff', 'Lag_1_Pct_Change', 'Lag_2_Pct_Change', 'Lag_3_Pct_Change',
        'Rolling_Mean_3', 'Rolling_Mean_7', 'Rolling_Mean_14', 'Rolling_Std_3',
        'Rolling_Std_7', 'Rolling_Min_3', 'Rolling_Max_3', 'Rolling_Range_3',
        'Rolling_Range_7', 'Rolling_Median_3', 'Rolling_Skew_3', 'Weekend_Holiday',
        'Weekend_Lag1_Ratio', 'Holiday_Lag1_Ratio'
    ]
    features = features[feature_columns]
    return features

def get_forecast_plot(df):
    stacked_model, prophet_model, scaler = load_models()
    last_date = df['InvoiceDate'].max()
    future_dates = pd.date_range(start=last_date, periods=30, freq='D')
    X_future = build_forecast_features(df, future_dates)
    X_future_scaled = scaler.transform(X_future)
    stacked_pred = stacked_model.predict(X_future_scaled)
    
    # Create Prophet future dataframe with required regressors
    prophet_future = prophet_model.make_future_dataframe(periods=30)
    
    # Add all required regressors
    prophet_future['is_weekend'] = prophet_future['ds'].dt.dayofweek.isin([5, 6]).astype(int)
    prophet_future['is_holiday'] = prophet_future['ds'].isin(pd.to_datetime(['2011-12-25'])).astype(int)
    prophet_future['month'] = prophet_future['ds'].dt.month
    prophet_future['day_of_week'] = prophet_future['ds'].dt.dayofweek
    prophet_future['day_of_month'] = prophet_future['ds'].dt.day
    prophet_future['quarter'] = prophet_future['ds'].dt.quarter
    prophet_future['year'] = prophet_future['ds'].dt.year
    
    prophet_forecast = prophet_model.predict(prophet_future)
    
    actual_data = df.groupby('InvoiceDate')['TotalAmount'].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_data['InvoiceDate'], y=actual_data['TotalAmount'], name='Actual', line=dict(color='#6c5ce7', width=2)))
    fig.add_trace(go.Scatter(x=future_dates, y=stacked_pred, name='Stacked Model Forecast', line=dict(color='#00cec9', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=prophet_forecast['ds'].iloc[-30:], y=prophet_forecast['yhat'].iloc[-30:], name='Prophet Forecast', line=dict(color='#fdcb6e', width=2, dash='dot')))
    fig.update_layout(title='', xaxis_title='Date', yaxis_title='Revenue', height=400, template='plotly_white', hovermode='x unified', showlegend=True)
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn', div_id='forecast-plot')

def get_forecast_metrics(df):
    stacked_model, prophet_model, scaler = load_models()
    last_date = df['InvoiceDate'].max()
    future_dates = pd.date_range(start=last_date, periods=30, freq='D')
    X_future = build_forecast_features(df, future_dates)
    X_future_scaled = scaler.transform(X_future)
    stacked_pred = stacked_model.predict(X_future_scaled)
    return [
        {"label": "Average Daily Forecast", "value": f"${stacked_pred.mean():,.2f}"},
        {"label": "Maximum Forecast", "value": f"${stacked_pred.max():,.2f}"},
        {"label": "Minimum Forecast", "value": f"${stacked_pred.min():,.2f}"},
    ]

def get_forecast_csv(df):
    stacked_model, prophet_model, scaler = load_models()
    last_date = df['InvoiceDate'].max()
    future_dates = pd.date_range(start=last_date, periods=30, freq='D')
    X_future = build_forecast_features(df, future_dates)
    X_future_scaled = scaler.transform(X_future)
    stacked_pred = stacked_model.predict(X_future_scaled)
    return pd.DataFrame({'Date': future_dates, 'Stacked Forecast': stacked_pred})