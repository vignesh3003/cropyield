#!/usr/bin/env python3
"""
Statistical Calculations for Crop Production Dataset

This script calculates mean, median, mode, and variance for:
- All categorical variables (Seasons, Crops, States, Districts, Years)
- Associated numerical values (Area, Production, Yield)
- Group-wise statistics for each category
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the crop production dataset."""
    print("Loading and preprocessing data...")
    
    # Load dataset
    df = pd.read_csv("crop_production.csv")
    
    # Basic preprocessing
    df['State_Name'] = df['State_Name'].str.strip()
    df['District_Name'] = df['District_Name'].str.strip()
    df['Season'] = df['Season'].str.strip()
    df['Crop'] = df['Crop'].str.strip()
    
    # Handle missing values
    df['Production'] = df['Production'].fillna(0)
    
    # Remove invalid data
    df = df[df['Area'] > 0]
    df = df[df['Production'] >= 0]
    
    # Calculate yield
    df['Yield'] = df['Production'] / df['Area']
    df = df[~np.isinf(df['Yield']) & ~np.isnan(df['Yield'])]
    
    print(f"Dataset loaded: {len(df):,} records")
    return df

def calculate_basic_statistics(df, column_name, value_column):
    """Calculate basic statistics for a categorical column with associated numerical values."""
    print(f"\n{'='*60}")
    print(f"STATISTICS FOR {column_name.upper()}")
    print(f"{'='*60}")
    
    # Get unique categories
    categories = df[column_name].unique()
    print(f"Number of unique {column_name}: {len(categories)}")
    
    # Calculate statistics for each category
    stats_data = []
    
    for category in categories:
        category_data = df[df[column_name] == category][value_column]
        
        if len(category_data) > 0:
            mean_val = category_data.mean()
            median_val = category_data.median()
            mode_val = category_data.mode().iloc[0] if len(category_data.mode()) > 0 else "No mode"
            variance_val = category_data.var()
            std_val = category_data.std()
            min_val = category_data.min()
            max_val = category_data.max()
            count_val = len(category_data)
            
            stats_data.append({
                'Category': category,
                'Count': count_val,
                'Mean': mean_val,
                'Median': median_val,
                'Mode': mode_val,
                'Variance': variance_val,
                'Std Dev': std_val,
                'Min': min_val,
                'Max': max_val
            })
            
            print(f"\n{category}:")
            print(f"  Count: {count_val:,}")
            print(f"  Mean: {mean_val:.4f}")
            print(f"  Median: {median_val:.4f}")
            print(f"  Mode: {mode_val}")
            print(f"  Variance: {variance_val:.4f}")
            print(f"  Std Dev: {std_val:.4f}")
            print(f"  Min: {min_val:.4f}")
            print(f"  Max: {max_val:.4f}")
    
    return pd.DataFrame(stats_data)
 
def calculate_overall_statistics(df):
    """Calculate overall statistics for numerical columns."""
    print(f"\n{'='*60}")
    print("OVERALL DATASET STATISTICS")
    print(f"{'='*60}")
    
    numerical_columns = ['Area', 'Production', 'Yield']
    
    for col in numerical_columns:
        print(f"\n{col.upper()} Statistics:")
        print("-" * 40)
        
        data = df[col].dropna()
        
        mean_val = data.mean()
        median_val = data.median()
        mode_val = data.mode().iloc[0] if len(data.mode()) > 0 else "No mode"
        variance_val = data.var()
        std_val = data.std()
        min_val = data.min()
        max_val = data.max()
        count_val = len(data)
        
        print(f"Count: {count_val:,}")
        print(f"Mean: {mean_val:.4f}")
        print(f"Median: {median_val:.4f}")
        print(f"Mode: {mode_val}")
        print(f"Variance: {variance_val:.4f}")
        print(f"Std Dev: {std_val:.4f}")
        print(f"Min: {min_val:.4f}")
        print(f"Max: {max_val:.4f}")
        
        # Additional statistics
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        
        print(f"Skewness: {skewness:.4f}")
        print(f"Kurtosis: {kurtosis:.4f}")
        print(f"Q1 (25th percentile): {q25:.4f}")
        print(f"Q3 (75th percentile): {q75:.4f}")
        print(f"IQR: {iqr:.4f}")

def calculate_season_statistics(df):
    """Calculate detailed statistics for seasons."""
    print(f"\n{'='*60}")
    print("DETAILED SEASON STATISTICS")
    print(f"{'='*60}")
    
    seasons = df['Season'].unique()
    
    for season in seasons:
        season_data = df[df['Season'] == season]
        print(f"\n{season.upper()} Season:")
        print("-" * 30)
        
        # Area statistics
        area_stats = season_data['Area']
        print(f"Area - Mean: {area_stats.mean():.2f}, Median: {area_stats.median():.2f}, Variance: {area_stats.var():.2f}")
        
        # Production statistics
        prod_stats = season_data['Production']
        print(f"Production - Mean: {prod_stats.mean():.2f}, Median: {prod_stats.median():.2f}, Variance: {prod_stats.var():.2f}")
        
        # Yield statistics
        yield_stats = season_data['Yield']
        print(f"Yield - Mean: {yield_stats.mean():.2f}, Median: {yield_stats.median():.2f}, Variance: {yield_stats.var():.2f}")
        
        # Count
        print(f"Record Count: {len(season_data):,}")

def calculate_crop_statistics(df):
    """Calculate detailed statistics for top crops."""
    print(f"\n{'='*60}")
    print("DETAILED CROP STATISTICS (TOP 20)")
    print(f"{'='*60}")
    
    top_crops = df['Crop'].value_counts().head(20)
    
    for crop in top_crops.index:
        crop_data = df[df['Crop'] == crop]
        print(f"\n{crop.upper()}:")
        print("-" * 30)
        
        # Area statistics
        area_stats = crop_data['Area']
        print(f"Area - Mean: {area_stats.mean():.2f}, Median: {area_stats.median():.2f}, Variance: {area_stats.var():.2f}")
        
        # Production statistics
        prod_stats = crop_data['Production']
        print(f"Production - Mean: {prod_stats.mean():.2f}, Median: {prod_stats.median():.2f}, Variance: {prod_stats.var():.2f}")
        
        # Yield statistics
        yield_stats = crop_data['Yield']
        print(f"Yield - Mean: {yield_stats.mean():.2f}, Median: {yield_stats.median():.2f}, Variance: {yield_stats.var():.2f}")
        
        # Count
        print(f"Record Count: {len(crop_data):,}")

def calculate_state_statistics(df):
    """Calculate detailed statistics for top states."""
    print(f"\n{'='*60}")
    print("DETAILED STATE STATISTICS (TOP 15)")
    print(f"{'='*60}")
    
    top_states = df['State_Name'].value_counts().head(15)
    
    for state in top_states.index:
        state_data = df[df['State_Name'] == state]
        print(f"\n{state.upper()}:")
        print("-" * 30)
        
        # Area statistics
        area_stats = state_data['Area']
        print(f"Area - Mean: {area_stats.mean():.2f}, Median: {area_stats.median():.2f}, Variance: {area_stats.var():.2f}")
        
        # Production statistics
        prod_stats = state_data['Production']
        print(f"Production - Mean: {prod_stats.mean():.2f}, Median: {prod_stats.median():.2f}, Variance: {prod_stats.var():.2f}")
        
        # Yield statistics
        yield_stats = state_data['Yield']
        print(f"Yield - Mean: {yield_stats.mean():.2f}, Median: {yield_stats.median():.2f}, Variance: {yield_stats.var():.2f}")
        
        # Count
        print(f"Record Count: {len(state_data):,}")

def calculate_year_statistics(df):
    """Calculate detailed statistics for years."""
    print(f"\n{'='*60}")
    print("DETAILED YEAR STATISTICS")
    print(f"{'='*60}")
    
    years = sorted(df['Crop_Year'].unique())
    
    for year in years:
        year_data = df[df['Crop_Year'] == year]
        print(f"\nYear {year}:")
        print("-" * 20)
        
        # Area statistics
        area_stats = year_data['Area']
        print(f"Area - Mean: {area_stats.mean():.2f}, Median: {area_stats.median():.2f}, Variance: {area_stats.var():.2f}")
        
        # Production statistics
        prod_stats = year_data['Production']
        print(f"Production - Mean: {prod_stats.mean():.2f}, Median: {prod_stats.median():.2f}, Variance: {prod_stats.var():.2f}")
        
        # Yield statistics
        yield_stats = year_data['Yield']
        print(f"Yield - Mean: {yield_stats.mean():.2f}, Median: {yield_stats.median():.2f}, Variance: {yield_stats.var():.2f}")
        
        # Count
        print(f"Record Count: {len(year_data):,}")

def create_summary_dataframe(df):
    """Create a summary dataframe with all statistics."""
    print(f"\n{'='*60}")
    print("CREATING SUMMARY STATISTICS DATAFRAME")
    print(f"{'='*60}")
    
    # Calculate statistics for each categorical column
    season_stats = calculate_basic_statistics(df, 'Season', 'Yield')
    crop_stats = calculate_basic_statistics(df, 'Crop', 'Yield')
    state_stats = calculate_basic_statistics(df, 'State_Name', 'Yield')
    
    # Save to CSV files
    season_stats.to_csv('season_statistics.csv', index=False)
    crop_stats.to_csv('crop_statistics.csv', index=False)
    state_stats.to_csv('state_statistics.csv', index=False)
    
    print(f"\n✅ Statistics saved to CSV files:")
    print(f"   • season_statistics.csv")
    print(f"   • crop_statistics.csv")
    print(f"   • state_statistics.csv")
    
    return season_stats, crop_stats, state_stats

def print_formula_explanations():
    """Print explanations of the mathematical formulas used."""
    print(f"\n{'='*60}")
    print("MATHEMATICAL FORMULAS EXPLANATION")
    print(f"{'='*60}")
    
    formulas = [
        ("Mean (Arithmetic Average)", "μ = (Σx) / n", "Sum of all values divided by count of values"),
        ("Median", "Middle value when ordered", "Central value that divides data into two equal halves"),
        ("Mode", "Most frequent value", "Value that appears most often in the dataset"),
        ("Variance", "σ² = Σ(x - μ)² / n", "Average of squared differences from the mean"),
        ("Standard Deviation", "σ = √(Variance)", "Square root of variance, measures spread of data"),
        ("Skewness", "Measure of distribution asymmetry", "Positive = right-skewed, Negative = left-skewed"),
        ("Kurtosis", "Measure of distribution tail weight", "High = heavy tails, Low = light tails"),
        ("Quartiles", "Q1 = 25th percentile, Q3 = 75th percentile", "Divide data into four equal parts"),
        ("IQR (Interquartile Range)", "IQR = Q3 - Q1", "Range containing middle 50% of data")
    ]
    
    for name, formula, explanation in formulas:
        print(f"\n{name}:")
        print(f"  Formula: {formula}")
        print(f"  Explanation: {explanation}")

def main():
    """Main function to execute all statistical calculations."""
    print("Starting comprehensive statistical analysis for crop production dataset...")
    print("="*80)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Calculate overall statistics
    calculate_overall_statistics(df)
    
    # Calculate detailed statistics for each category
    calculate_season_statistics(df)
    calculate_crop_statistics(df)
    calculate_state_statistics(df)
    calculate_year_statistics(df)
    
    # Create summary dataframes
    season_stats, crop_stats, state_stats = create_summary_dataframe(df)
    
    # Print formula explanations
    print_formula_explanations()
    
    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print("Generated files:")
    print("• season_statistics.csv - Detailed season statistics")
    print("• crop_statistics.csv - Detailed crop statistics")
    print("• state_statistics.csv - Detailed state statistics")
    print("\nAll calculations include:")
    print("• Mean, Median, Mode")
    print("• Variance and Standard Deviation")
    print("• Min, Max, Count")
    print("• Skewness and Kurtosis")
    print("• Quartiles and IQR")

if __name__ == "__main__":
    main()
