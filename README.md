# E-commerce Data Analysis Project

This project analyzes an e-commerce dataset to gain insights into sales patterns, customer behavior, and product performance.

## Project Structure

- `ecommerce_analysis.py`: Main analysis script
- `requirements.txt`: Python package dependencies
- `Online Retail.xlsx`: Input dataset
- Generated plots:
  - `monthly_revenue.png`: Monthly revenue trends
  - `top_products.png`: Top 10 products by quantity
  - `country_distribution.png`: Revenue distribution by country
  - `purchase_patterns.png`: Purchase patterns by day and hour

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Analysis

Execute the main script:
```bash
python ecommerce_analysis.py
```

## Analysis Components

1. Data Collection and Cleaning
   - Loads the Excel dataset
   - Removes missing values
   - Converts date formats
   - Removes canceled orders

2. Exploratory Data Analysis
   - Revenue trends over time
   - Top products analysis
   - Country-wise revenue distribution
   - Customer purchase patterns

3. Predictive Analysis
   - Uses Random Forest to predict future sales
   - Provides model performance metrics

## Output

The script generates several visualization files and prints analysis results to the console. 