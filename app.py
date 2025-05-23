from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
import json
from utils import load_data, clean_data, load_models, get_regions, get_metrics, get_revenue_trend_plot, get_top_products_plot, get_country_revenue_plot, get_customer_patterns_plot, get_forecast_plot, get_forecast_metrics

app = Flask(__name__)

# Placeholder: Load data and models at startup (or per request if needed)
# df = load_data('Online Retail.xlsx')
# df = clean_data(df)
# stacked_model, prophet_model, scaler = load_models()

def get_filtered_df(region):
    df = clean_data(load_data())
    if region and region != "All":
        df = df[df['Country'] == region]
    return df

@app.route('/')
def landing():
    # Render landing page with animation/modal and Enter Dashboard button
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    # Default to overview section
    return redirect(url_for('overview'))

@app.route('/overview')
def overview():
    df = clean_data(load_data())
    region = request.args.get('region', 'All')
    regions = get_regions(df)
    if region != 'All':
        df = df[df['Country'] == region]
    metrics = get_metrics(df)
    plot_json = get_revenue_trend_plot(df)
    return render_template(
        'overview.html',
        metrics=metrics,
        plot_json=plot_json,
        regions=regions,
        selected_region=region
    )

@app.route('/analysis')
def analysis():
    region = request.args.get('region', 'All')
    df = get_filtered_df(region)
    regions = get_regions(clean_data(load_data()))
    top_products_plot = get_top_products_plot(df)
    country_revenue_plot = get_country_revenue_plot(df)
    customer_patterns_plot = get_customer_patterns_plot(df)
    return render_template(
        'analysis.html',
        top_products_plot=top_products_plot,
        country_revenue_plot=country_revenue_plot,
        customer_patterns_plot=customer_patterns_plot,
        regions=regions,
        selected_region=region
    )

@app.route('/forecast')
def forecast():
    region = request.args.get('region', 'All')
    df = get_filtered_df(region)
    regions = get_regions(clean_data(load_data()))
    forecast_plot = get_forecast_plot(df)
    forecast_metrics = get_forecast_metrics(df)
    return render_template(
        'forecast.html',
        forecast_plot=forecast_plot,
        forecast_metrics=forecast_metrics,
        regions=regions,
        selected_region=region
    )

# Download endpoints (CSV)
@app.route('/download/<datatype>')
def download_csv(datatype):
    df = clean_data(load_data())
    region = request.args.get('region', 'All')
    if region != 'All':
        df = df[df['Country'] == region]
    if datatype == 'revenue':
        data = df.groupby(df['InvoiceDate'].dt.date)['TotalAmount'].sum().reset_index()
        filename = 'revenue_data.csv'
    # Add logic for other datatypes...
    data.to_csv(filename, index=False)
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True) 