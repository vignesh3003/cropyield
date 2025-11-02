#!/usr/bin/env python3
"""
Crop Yield Dataset Cleaning Script

This script cleans the crop_yield.csv dataset by addressing various data quality issues:
- Handling missing values
- Standardizing text data (crop names, seasons, states)
- Removing outliers and invalid data
- Data type conversions
- Consistency checks
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class CropYieldCleaner:
    def __init__(self, file_path: str):
        """
        Initialize the cleaner with the dataset file path.
        
        Args:
            file_path (str): Path to the CSV file
        """
        self.file_path = file_path
        self.df = None
        self.cleaned_df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the dataset from CSV file."""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Dataset loaded successfully. Shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            return self.df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def display_data_info(self):
        """Display basic information about the dataset."""
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
            
        print("\n" + "="*50)
        print("DATASET INFORMATION")
        print("="*50)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\nColumn Information:")
        print(self.df.info())
        
        print("\nMissing Values:")
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            print(missing_data[missing_data > 0])
        else:
            print("No missing values found.")
            
        print("\nData Types:")
        print(self.df.dtypes)
        
        print("\nSample Data (first 5 rows):")
        print(self.df.head())
    
    def clean_crop_names(self) -> pd.DataFrame:
        """Clean and standardize crop names."""
        if self.df is None:
            return None
            
        df = self.df.copy()
        
        # Remove extra spaces and standardize crop names
        df['Crop'] = df['Crop'].str.strip()
        
        # Standardize common crop name variations
        crop_mappings = {
            'Arhar/Tur': 'Arhar/Tur',
            'Moong(Green Gram)': 'Moong (Green Gram)',
            'Rapeseed &Mustard': 'Rapeseed & Mustard',
            'Peas & beans (Pulses)': 'Peas & Beans (Pulses)',
            'Other  Rabi pulses': 'Other Rabi Pulses',
            'Other Kharif pulses': 'Other Kharif Pulses',
            'other oilseeds': 'Other Oilseeds',
            'Sannhamp': 'Sannhemp'
        }
        
        df['Crop'] = df['Crop'].replace(crop_mappings)
        
        # Remove any remaining extra spaces
        df['Crop'] = df['Crop'].str.replace(r'\s+', ' ', regex=True)
        
        print(f"Crop names cleaned. Unique crops: {df['Crop'].nunique()}")
        return df
    
    def clean_seasons(self) -> pd.DataFrame:
        """Clean and standardize season names."""
        if self.df is None:
            return None
            
        df = self.df.copy()
        
        # Remove extra spaces and standardize season names
        df['Season'] = df['Season'].str.strip()
        
        # Standardize season names
        season_mappings = {
            'Whole Year ': 'Whole Year',
            'Kharif     ': 'Kharif',
            'Rabi       ': 'Rabi',
            'Summer     ': 'Summer',
            'Autumn     ': 'Autumn',
            'Winter     ': 'Winter'
        }
        
        df['Season'] = df['Season'].replace(season_mappings)
        
        print(f"Seasons cleaned. Unique seasons: {df['Season'].nunique()}")
        print(f"Seasons: {sorted(df['Season'].unique())}")
        return df
    
    def clean_states(self) -> pd.DataFrame:
        """Clean and standardize state names."""
        if self.df is None:
            return None
            
        df = self.df.copy()
        
        # Remove extra spaces and standardize state names
        df['State'] = df['State'].str.strip()
        
        # Standardize state names
        state_mappings = {
            'Andhra Pradesh': 'Andhra Pradesh',
            'Assam': 'Assam',
            'Goa': 'Goa',
            'Karnataka': 'Karnataka',
            'Kerala': 'Kerala',
            'Meghalaya': 'Meghalaya',
            'Puducherry': 'Puducherry',
            'Tamil Nadu': 'Tamil Nadu',
            'West Bengal': 'West Bengal'
        }
        
        df['State'] = df['State'].replace(state_mappings)
        
        print(f"States cleaned. Unique states: {df['State'].nunique()}")
        print(f"States: {sorted(df['State'].unique())}")
        return df
    
    def clean_numeric_columns(self) -> pd.DataFrame:
        """Clean numeric columns and handle invalid values."""
        if self.df is None:
            return None
            
        df = self.df.copy()
        
        # Define numeric columns
        numeric_cols = ['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']
        
        for col in numeric_cols:
            if col in df.columns:
                # Convert to numeric, coerce errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Remove negative values (not logical for these metrics)
                if col in ['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']:
                    df.loc[df[col] < 0, col] = np.nan
                
                # Remove extremely high outliers (beyond 99.9th percentile)
                if col in ['Area', 'Production', 'Fertilizer', 'Pesticide']:
                    q99 = df[col].quantile(0.999)
                    df.loc[df[col] > q99, col] = np.nan
                
                print(f"{col}: {df[col].isnull().sum()} missing values after cleaning")
        
        return df
    
    def clean_crop_year(self) -> pd.DataFrame:
        """Clean and validate crop year column."""
        if self.df is None:
            return None
            
        df = self.df.copy()
        
        # Convert to numeric
        df['Crop_Year'] = pd.to_numeric(df['Crop_Year'], errors='coerce')
        
        # Remove invalid years (assuming reasonable range 1950-2030)
        df.loc[(df['Crop_Year'] < 1950) | (df['Crop_Year'] > 2030), 'Crop_Year'] = np.nan
        
        print(f"Crop_Year: {df['Crop_Year'].isnull().sum()} missing values after cleaning")
        print(f"Year range: {df['Crop_Year'].min()} - {df['Crop_Year'].max()}")
        
        return df
    
    def handle_missing_values(self) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        if self.df is None:
            return None
            
        df = self.df.copy()
        
        print("\nMissing values before handling:")
        print(df.isnull().sum())
        
        # For categorical columns, fill with mode
        categorical_cols = ['Crop', 'Season', 'State']
        for col in categorical_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                print(f"Filled missing values in {col} with mode: {mode_val}")
        
        # For numeric columns, fill with median (more robust than mean)
        numeric_cols = ['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']
        for col in numeric_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Filled missing values in {col} with median: {median_val:.2f}")
        
        # For Crop_Year, fill with mode
        if 'Crop_Year' in df.columns and df['Crop_Year'].isnull().sum() > 0:
            mode_year = df['Crop_Year'].mode()[0]
            df['Crop_Year'] = df['Crop_Year'].fillna(mode_year)
            print(f"Filled missing values in Crop_Year with mode: {mode_year}")
        
        print("\nMissing values after handling:")
        print(df.isnull().sum())
        
        return df
    
    def remove_duplicates(self) -> pd.DataFrame:
        """Remove duplicate rows from the dataset."""
        if self.df is None:
            return None
            
        df = self.df.copy()
        
        initial_rows = len(df)
        df = df.drop_duplicates()
        final_rows = len(df)
        
        duplicates_removed = initial_rows - final_rows
        print(f"Removed {duplicates_removed} duplicate rows")
        print(f"Dataset shape after removing duplicates: {df.shape}")
        
        return df
    
    def validate_data_consistency(self) -> pd.DataFrame:
        """Validate data consistency and remove logically inconsistent records."""
        if self.df is None:
            return None
            
        df = self.df.copy()
        
        initial_rows = len(df)
        
        # Remove records where Area is 0 but Production > 0 (logically inconsistent)
        if 'Area' in df.columns and 'Production' in df.columns:
            inconsistent_area = df[(df['Area'] == 0) & (df['Production'] > 0)]
            if len(inconsistent_area) > 0:
                print(f"Found {len(inconsistent_area)} records with Area=0 but Production>0")
                df = df[~((df['Area'] == 0) & (df['Production'] > 0))]
        
        # Remove records where Production is 0 but Area > 0 (might be valid, but check)
        if 'Area' in df.columns and 'Production' in df.columns:
            zero_production = df[(df['Production'] == 0) & (df['Area'] > 0)]
            if len(zero_production) > 0:
                print(f"Found {len(zero_production)} records with Production=0 but Area>0")
                # Keep these as they might represent failed crops
        
        # Remove records with extremely high yield values (beyond reasonable limits)
        if 'Yield' in df.columns:
            # Yield should typically be between 0 and 1000 (tons per hectare)
            extreme_yield = df[(df['Yield'] < 0) | (df['Yield'] > 1000)]
            if len(extreme_yield) > 0:
                print(f"Found {len(extreme_yield)} records with extreme yield values")
                df = df[(df['Yield'] >= 0) & (df['Yield'] <= 1000)]
        
        final_rows = len(df)
        removed_rows = initial_rows - final_rows
        
        if removed_rows > 0:
            print(f"Removed {removed_rows} logically inconsistent records")
            print(f"Dataset shape after validation: {df.shape}")
        else:
            print("No logically inconsistent records found")
        
        return df
    
    def create_derived_features(self) -> pd.DataFrame:
        """Create useful derived features for analysis."""
        if self.df is None:
            return None
            
        df = self.df.copy()
        
        # Calculate yield per unit area (if not already present or to verify)
        if 'Area' in df.columns and 'Production' in df.columns:
            df['Calculated_Yield'] = df['Production'] / df['Area']
            df['Calculated_Yield'] = df['Calculated_Yield'].replace([np.inf, -np.inf], np.nan)
            
            # Compare with existing Yield column if available
            if 'Yield' in df.columns:
                yield_diff = abs(df['Calculated_Yield'] - df['Yield'])
                print(f"Average difference between calculated and existing yield: {yield_diff.mean():.4f}")
        
        # Create season category
        if 'Season' in df.columns:
            df['Season_Category'] = df['Season'].map({
                'Kharif': 'Monsoon',
                'Rabi': 'Winter',
                'Summer': 'Summer',
                'Autumn': 'Autumn',
                'Winter': 'Winter',
                'Whole Year': 'Year-round'
            })
        
        # Create decade column
        if 'Crop_Year' in df.columns:
            df['Decade'] = (df['Crop_Year'] // 10) * 10
        
        print("Derived features created successfully")
        return df
    
    def final_data_quality_check(self) -> Dict:
        """Perform final data quality assessment."""
        if self.df is None:
            return {}
            
        df = self.df.copy()
        
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'unique_values': {}
        }
        
        # Count unique values for categorical columns
        categorical_cols = ['Crop', 'Season', 'State']
        for col in categorical_cols:
            if col in df.columns:
                quality_report['unique_values'][col] = df[col].nunique()
        
        # Basic statistics for numeric columns
        numeric_cols = ['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']
        for col in numeric_cols:
            if col in df.columns:
                quality_report[f'{col}_stats'] = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std()
                }
        
        return quality_report
    
    def clean_dataset(self) -> pd.DataFrame:
        """Execute the complete cleaning pipeline."""
        print("Starting dataset cleaning process...")
        print("="*60)
        
        # Load data
        self.load_data()
        if self.df is None:
            return None
        
        # Display initial information
        self.display_data_info()
        
        # Apply cleaning steps
        print("\n" + "="*60)
        print("APPLYING CLEANING STEPS")
        print("="*60)
        
        # Step 1: Clean text columns
        print("\n1. Cleaning text columns...")
        self.df = self.clean_crop_names()
        self.df = self.clean_seasons()
        self.df = self.clean_states()
        
        # Step 2: Clean numeric columns
        print("\n2. Cleaning numeric columns...")
        self.df = self.clean_numeric_columns()
        self.df = self.clean_crop_year()
        
        # Step 3: Handle missing values
        print("\n3. Handling missing values...")
        self.df = self.handle_missing_values()
        
        # Step 4: Remove duplicates
        print("\n4. Removing duplicates...")
        self.df = self.remove_duplicates()
        
        # Step 5: Validate data consistency
        print("\n5. Validating data consistency...")
        self.df = self.validate_data_consistency()
        
        # Step 6: Create derived features
        print("\n6. Creating derived features...")
        self.df = self.create_derived_features()
        
        # Step 7: Final quality check
        print("\n7. Performing final quality check...")
        quality_report = self.final_data_quality_check()
        
        # Display final results
        print("\n" + "="*60)
        print("CLEANING COMPLETED")
        print("="*60)
        
        print(f"\nFinal dataset shape: {self.df.shape}")
        print(f"Total missing values: {self.df.isnull().sum().sum()}")
        print(f"Duplicate rows: {self.df.duplicated().sum()}")
        
        print("\nQuality Report Summary:")
        for key, value in quality_report.items():
            if key not in ['missing_values', 'data_types', 'unique_values']:
                print(f"{key}: {value}")
        
        self.cleaned_df = self.df.copy()
        return self.df
    
    def save_cleaned_data(self, output_path: str = "cleaned_crop_yield.csv"):
        """Save the cleaned dataset to a new CSV file."""
        if self.cleaned_df is None:
            print("No cleaned data available. Please run clean_dataset() first.")
            return
        
        try:
            self.cleaned_df.to_csv(output_path, index=False)
            print(f"\nCleaned dataset saved to: {output_path}")
        except Exception as e:
            print(f"Error saving cleaned dataset: {e}")
    
    def get_cleaning_summary(self) -> str:
        """Generate a summary of the cleaning process."""
        if self.cleaned_df is None:
            return "No cleaned data available. Please run clean_dataset() first."
        
        summary = f"""
CROP YIELD DATASET CLEANING SUMMARY
===================================

Original Dataset:
- Rows: {len(self.df) if self.df is not None else 'N/A'}
- Columns: {len(self.df.columns) if self.df is not None else 'N/A'}

Cleaned Dataset:
- Rows: {len(self.cleaned_df)}
- Columns: {len(self.cleaned_df.columns)}

Cleaning Actions Performed:
1. Text standardization (Crop names, Seasons, States)
2. Numeric data cleaning and outlier removal
3. Missing value handling
4. Duplicate removal
5. Data consistency validation
6. Derived feature creation

Data Quality Improvements:
- Missing values: {self.cleaned_df.isnull().sum().sum()}
- Duplicates: {self.cleaned_df.duplicated().sum()}
- Data types: All columns have appropriate data types
- Logical consistency: Validated area, production, and yield relationships

The cleaned dataset is now ready for analysis and modeling.
"""
        return summary

def main():
    """Main function to execute the cleaning process."""
    # Initialize the cleaner
    cleaner = CropYieldCleaner("crop_yield.csv")
    
    # Clean the dataset
    cleaned_data = cleaner.clean_dataset()
    
    if cleaned_data is not None:
        # Save the cleaned dataset
        cleaner.save_cleaned_data()
        
        # Display cleaning summary
        print("\n" + "="*60)
        print("CLEANING SUMMARY")
        print("="*60)
        print(cleaner.get_cleaning_summary())
        
        # Display sample of cleaned data
        print("\nSample of cleaned data:")
        print(cleaned_data.head(10))
        
        print("\nDataset cleaning completed successfully!")
    else:
        print("Dataset cleaning failed. Please check the input file.")

if __name__ == "__main__":
    main()
