"""
Synthetic Data Generator for Retail Promotional Event Analysis
==============================================================

This script generates synthetic data that mimics the statistical properties
of retail promotional event data without exposing any real company information.

The generated data preserves:
- Similar distributions for numeric metrics (sales, costs, ROI)
- Realistic categorical proportions (departments, regions, event types)
- Correlations between related variables
- Enough variation to demonstrate meaningful statistical analysis

Author: Jacob Aguillard
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_synthetic_retail_data(n_records: int = 100000) -> pd.DataFrame:
    """
    Generate synthetic retail promotional event data.
    
    Parameters
    ----------
    n_records : int
        Number of records to generate (default 100,000 for demo purposes)
    
    Returns
    -------
    pd.DataFrame
        Synthetic dataset with realistic promotional event metrics
    """
    
    # =========================================================================
    # CATEGORICAL DIMENSIONS
    # =========================================================================
    
    # Regions (generic names)
    regions = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'Northwest', 'West']
    region_weights = [0.18, 0.20, 0.17, 0.15, 0.15, 0.15]
    
    # Departments (generic retail categories)
    departments = [
        'Snacks & Candy', 'Beverages', 'Frozen Foods', 'Dairy & Refrigerated',
        'Bakery', 'Deli & Prepared', 'Health & Beauty', 'Household',
        'Electronics', 'Clothing', 'Home Goods', 'Pharmacy',
        'Meat & Seafood', 'Produce', 'Automotive'
    ]
    # Weighted to reflect typical retail distribution
    dept_weights = [0.12, 0.11, 0.10, 0.09, 0.08, 0.08, 0.07, 0.07, 
                    0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.01]
    
    # Department divisions
    dept_to_division = {
        'Snacks & Candy': 'Food', 'Beverages': 'Food', 'Frozen Foods': 'Food',
        'Dairy & Refrigerated': 'Food', 'Bakery': 'Food', 'Deli & Prepared': 'Food',
        'Meat & Seafood': 'Food', 'Produce': 'Food',
        'Health & Beauty': 'Non-Food', 'Household': 'Non-Food', 
        'Electronics': 'Non-Food', 'Clothing': 'Non-Food',
        'Home Goods': 'Non-Food', 'Pharmacy': 'Non-Food', 'Automotive': 'Non-Food'
    }
    
    # Event types
    event_types = ['Standard', 'Premium', 'Split', 'Special']
    event_type_weights = [0.60, 0.25, 0.10, 0.05]
    
    # Event detail types
    event_detail_types = ['Demo & Sample', 'Display Only', 'Coupon Distribution', '']
    event_detail_weights = [0.45, 0.30, 0.15, 0.10]
    
    # Weather conditions
    weather_conditions = ['sunny', 'cloudy', 'rainy', 'snowy']
    weather_weights = [0.50, 0.30, 0.15, 0.05]
    
    # =========================================================================
    # GENERATE BASE CATEGORICAL DATA
    # =========================================================================
    
    print("Generating categorical dimensions...")
    
    data = {
        'region': np.random.choice(regions, n_records, p=region_weights),
        'department': np.random.choice(departments, n_records, p=dept_weights),
        'event_type': np.random.choice(event_types, n_records, p=event_type_weights),
        'event_detail_type': np.random.choice(event_detail_types, n_records, p=event_detail_weights),
        'weather': np.random.choice(weather_conditions, n_records, p=weather_weights),
    }
    
    df = pd.DataFrame(data)
    
    # Add division based on department
    df['division'] = df['department'].map(dept_to_division)
    
    # Generate store IDs (500 stores across regions)
    stores_per_region = {r: list(range(i*100+1, i*100+85)) for i, r in enumerate(regions)}
    df['store_id'] = df['region'].apply(lambda r: random.choice(stores_per_region[r]))
    
    # Generate item IDs (random 6-7 digit numbers)
    df['item_id'] = np.random.randint(100000, 2000000, n_records)
    
    # Generate work order numbers
    df['work_order_id'] = np.random.randint(1000000, 2000000, n_records)
    
    # =========================================================================
    # GENERATE DATE FIELDS
    # =========================================================================
    
    print("Generating date fields...")
    
    # Events span 2 years
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = (end_date - start_date).days
    
    df['event_date'] = [start_date + timedelta(days=random.randint(0, date_range)) 
                        for _ in range(n_records)]
    
    # Execution date is same day or up to 5 days after
    df['execution_date'] = df['event_date'] + pd.to_timedelta(
        np.random.randint(0, 6, n_records), unit='D'
    )
    
    # =========================================================================
    # GENERATE NUMERIC METRICS
    # =========================================================================
    
    print("Generating numeric metrics...")
    
    # Base parameters vary by department (some departments have higher sales)
    dept_base_sales = {
        'Electronics': 800, 'Meat & Seafood': 500, 'Clothing': 400,
        'Health & Beauty': 350, 'Home Goods': 350, 'Pharmacy': 300,
        'Beverages': 250, 'Frozen Foods': 250, 'Dairy & Refrigerated': 250,
        'Snacks & Candy': 200, 'Deli & Prepared': 200, 'Bakery': 180,
        'Household': 180, 'Produce': 150, 'Automotive': 300
    }
    
    # Store door count (foot traffic proxy)
    df['door_count'] = np.random.normal(4000, 800, n_records).clip(1500, 8000).astype(int)
    
    # Event count (typically 1, sometimes 0.5 for split events)
    df['event_count'] = np.where(df['event_type'] == 'Split', 0.5, 1.0)
    
    # Base sales (what would have sold without event)
    base_multiplier = df['department'].map(dept_base_sales)
    df['base_sales'] = (base_multiplier * np.random.lognormal(0, 0.5, n_records)).clip(0, 5000)
    
    # Base units
    avg_price = base_multiplier / 10  # rough average price per unit
    df['base_units'] = (df['base_sales'] / avg_price * np.random.uniform(0.8, 1.2, n_records)).clip(0, 500).astype(int)
    
    # Event units (lifted units from the promotional event)
    # Events typically lift sales by 50-300%
    lift_multiplier = np.random.lognormal(0.5, 0.6, n_records).clip(0.5, 10)
    df['event_units'] = (df['base_units'] * lift_multiplier).clip(1, 1000).astype(int)
    
    # Total event day sales
    df['total_event_sales'] = df['event_units'] * (base_multiplier / 10) * np.random.uniform(0.9, 1.1, n_records)
    
    # Sales lift
    df['sales_lift'] = df['total_event_sales'] - df['base_sales']
    df['sales_lift_pct'] = np.where(df['base_sales'] > 0, 
                                      df['sales_lift'] / df['base_sales'], 
                                      1.0)
    
    # Unit lift
    df['unit_lift'] = df['event_units'] - df['base_units']
    df['unit_lift_pct'] = np.where(df['base_units'] > 0,
                                    df['unit_lift'] / df['base_units'],
                                    1.0)
    
    # =========================================================================
    # GENERATE COST METRICS
    # =========================================================================
    
    print("Generating cost metrics...")
    
    # Product charge (cost of demo products)
    df['product_charge'] = np.random.lognormal(4, 0.8, n_records).clip(10, 1000)
    
    # Labor charge
    df['labor_charge'] = np.random.normal(180, 50, n_records).clip(100, 400)
    
    # Supply charge
    df['supply_charge'] = np.random.choice([5, 10, 15, 20], n_records, p=[0.3, 0.4, 0.2, 0.1])
    
    # Tax on labor & supply
    df['labor_supply_tax'] = (df['labor_charge'] + df['supply_charge']) * np.random.uniform(0, 0.08, n_records)
    
    # Enhancements (optional add-ons)
    df['enhancements'] = np.where(np.random.random(n_records) < 0.2,
                                   np.random.uniform(10, 100, n_records), 0)
    
    # Coupons (negative = cost to company)
    df['coupons'] = np.where(np.random.random(n_records) < 0.25,
                              -np.random.uniform(5, 50, n_records), 0)
    
    # Other charges
    df['other_charges'] = np.where(np.random.random(n_records) < 0.15,
                                    np.random.uniform(5, 30, n_records), 0)
    
    # Total event cost
    df['total_event_cost'] = (df['product_charge'] + df['labor_charge'] + 
                               df['supply_charge'] + df['labor_supply_tax'] + 
                               df['enhancements'] - df['coupons'] + df['other_charges'])
    
    # =========================================================================
    # GENERATE ROI METRICS
    # =========================================================================
    
    print("Generating ROI metrics...")
    
    # Gross sales
    df['gross_sales'] = df['total_event_sales'] + df['product_charge']
    
    # ROI per dollar spent
    df['roi_per_dollar'] = np.where(df['total_event_cost'] > 0,
                                     df['gross_sales'] / df['total_event_cost'],
                                     df['gross_sales'])
    
    # ROI lift
    df['roi_lift'] = np.where(df['total_event_cost'] > 0,
                               (df['gross_sales'] - df['total_event_cost']) / df['total_event_cost'],
                               0)
    
    # Conversion rate (event units / typical customer exposure)
    df['conversion_rate'] = df['event_units'] / 240  # assuming 240 customer exposures per event
    
    # =========================================================================
    # GENERATE PROMOTIONAL TACTIC FLAGS
    # =========================================================================
    
    print("Generating promotional tactic flags...")
    
    # End cap placement (premium shelf location)
    df['end_cap'] = np.random.choice([0, 1], n_records, p=[0.6, 0.4])
    
    # Product proximity (placed near related products)
    df['product_proximity'] = np.random.choice([0, 1], n_records, p=[0.3, 0.7])
    
    # Has coupon
    df['has_coupon'] = (df['coupons'] < 0).astype(int)
    
    # Digital signage
    df['digital_signage'] = np.random.choice([0, 1], n_records, p=[0.85, 0.15])
    
    # QR code on materials
    df['qr_code'] = np.random.choice([0, 1], n_records, p=[0.90, 0.10])
    
    # =========================================================================
    # ADD CORRELATED EFFECTS (make tactics actually impact ROI)
    # =========================================================================
    
    print("Adding correlated effects for tactics...")
    
    # End cap improves ROI by ~15% on average for food departments
    food_mask = df['division'] == 'Food'
    endcap_effect = np.where((df['end_cap'] == 1) & food_mask, 
                              np.random.uniform(1.05, 1.25, n_records), 1.0)
    df['roi_per_dollar'] = df['roi_per_dollar'] * endcap_effect
    
    # Coupons improve ROI by ~20% for certain departments
    coupon_boost_depts = ['Snacks & Candy', 'Frozen Foods', 'Beverages', 'Dairy & Refrigerated']
    coupon_mask = df['department'].isin(coupon_boost_depts)
    coupon_effect = np.where((df['has_coupon'] == 1) & coupon_mask,
                              np.random.uniform(1.10, 1.30, n_records), 1.0)
    df['roi_per_dollar'] = df['roi_per_dollar'] * coupon_effect
    
    # Recalculate ROI lift after effects
    df['roi_lift'] = df['roi_per_dollar'] - 1
    
    # =========================================================================
    # WINSORIZE ROI FOR CLEANER ANALYSIS
    # =========================================================================
    
    print("Winsorizing extreme values...")
    
    # Cap extreme ROI values
    roi_lower = df['roi_per_dollar'].quantile(0.05)
    roi_upper = df['roi_per_dollar'].quantile(0.95)
    df['roi_winsorized'] = df['roi_per_dollar'].clip(roi_lower, roi_upper)
    
    # =========================================================================
    # FINAL CLEANUP
    # =========================================================================
    
    print("Finalizing dataset...")
    
    # Round numeric columns appropriately
    money_cols = ['base_sales', 'total_event_sales', 'sales_lift', 'product_charge',
                  'labor_charge', 'labor_supply_tax', 'enhancements', 'coupons',
                  'other_charges', 'total_event_cost', 'gross_sales']
    for col in money_cols:
        df[col] = df[col].round(2)
    
    pct_cols = ['sales_lift_pct', 'unit_lift_pct', 'roi_per_dollar', 'roi_lift', 
                'conversion_rate', 'roi_winsorized']
    for col in pct_cols:
        df[col] = df[col].round(4)
    
    # Reorder columns for clarity
    column_order = [
        # Dimensions
        'region', 'store_id', 'department', 'division', 'item_id',
        'event_date', 'execution_date', 'event_type', 'event_detail_type',
        'work_order_id', 'weather', 'door_count',
        # Event metrics
        'event_count', 'base_sales', 'total_event_sales', 'sales_lift', 'sales_lift_pct',
        'base_units', 'event_units', 'unit_lift', 'unit_lift_pct',
        # Cost metrics
        'product_charge', 'labor_charge', 'supply_charge', 'labor_supply_tax',
        'enhancements', 'coupons', 'other_charges', 'total_event_cost',
        # ROI metrics
        'gross_sales', 'roi_per_dollar', 'roi_lift', 'roi_winsorized', 'conversion_rate',
        # Promotional tactics
        'end_cap', 'product_proximity', 'has_coupon', 'digital_signage', 'qr_code'
    ]
    
    df = df[column_order]
    
    print(f"\nGenerated {len(df):,} synthetic records")
    print(f"Departments: {df['department'].nunique()}")
    print(f"Regions: {df['region'].nunique()}")
    print(f"Date range: {df['event_date'].min()} to {df['event_date'].max()}")
    print(f"ROI range: {df['roi_winsorized'].min():.2f} to {df['roi_winsorized'].max():.2f}")
    
    return df


if __name__ == "__main__":
    # Generate the synthetic dataset
    df = generate_synthetic_retail_data(n_records=100000)
    
    # Save to CSV
    output_file = "synthetic_retail_promotions.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")
    
    # Display sample
    print("\nSample records:")
    print(df.head(10).to_string())
    
    # Display summary statistics
    print("\n\nSummary Statistics:")
    print(df.describe().round(2).to_string())
