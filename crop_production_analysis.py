#!/usr/bin/env python3
"""
Crop Production Dataset Analysis Script

This script performs comprehensive analysis on the crop_production.csv dataset including:
- Data preprocessing and cleaning
- Dataset summary and statistics
- Multiple visualizations
- Key findings and insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

class CropProductionAnalyzer:
    def __init__(self, file_path: str):
        """
        Initialize the analyzer with the dataset file path.
        
        Args:
            file_path (str): Path to the CSV file
        """
        self.file_path = file_path
        self.df = None
        self.cleaned_df = None
        
    def load_data(self):
        """Load the dataset."""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Dataset loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def preprocess_data(self):
        """Preprocess and clean the dataset."""
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        df = self.df.copy()
        
        # 1. Clean text columns
        print("1. Cleaning text columns...")
        
        # Clean State names
        df['State_Name'] = df['State_Name'].str.strip()
        
        # Clean District names
        df['District_Name'] = df['District_Name'].str.strip()
        
        # Clean Season names (remove extra spaces)
        df['Season'] = df['Season'].str.strip()
        
        # Clean Crop names
        df['Crop'] = df['Crop'].str.strip()
        
        # 2. Handle missing values
        print("2. Handling missing values...")
        print(f"Missing values before cleaning: {df.isnull().sum().sum()}")
        
        # For Production column, fill missing values with 0 (no production recorded)
        df['Production'] = df['Production'].fillna(0)
        
        print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
        
        # 3. Remove invalid data
        print("3. Removing invalid data...")
        initial_rows = len(df)
        
        # Remove records with zero or negative area
        df = df[df['Area'] > 0]
        
        # Remove records with negative production
        df = df[df['Production'] >= 0]
        
        # Remove extreme outliers (beyond 99.9th percentile)
        area_q99 = df['Area'].quantile(0.999)
        production_q99 = df['Production'].quantile(0.999)
        
        df = df[(df['Area'] <= area_q99) & (df['Production'] <= production_q99)]
        
        final_rows = len(df)
        removed_rows = initial_rows - final_rows
        print(f"Removed {removed_rows} invalid/outlier records")
        
        # 4. Create derived features
        print("4. Creating derived features...")
        
        # Calculate yield (Production / Area)
        df['Yield'] = df['Production'] / df['Area']
        
        # Remove infinite and NaN yield values
        df = df[~np.isinf(df['Yield']) & ~np.isnan(df['Yield'])]
        
        # Create decade column
        df['Decade'] = (df['Crop_Year'] // 10) * 10
        
        # Create year category
        df['Year_Category'] = pd.cut(df['Crop_Year'], 
                                   bins=[1995, 2000, 2005, 2010, 2015, 2020], 
                                   labels=['1996-2000', '2001-2005', '2006-2010', '2011-2015', '2016-2020'])
        
        print(f"Final dataset shape: {df.shape}")
        
        self.cleaned_df = df.copy()
        return df
    
    def dataset_summary(self):
        """Generate comprehensive dataset summary."""
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        
        df = self.cleaned_df
        
        # Basic information
        print(f"\n1. Number of rows & columns:")
        print(f"   Rows: {df.shape[0]:,}")
        print(f"   Columns: {df.shape[1]}")
        
        # Data types
        print(f"\n2. Data types:")
        print(df.dtypes.value_counts())
        
        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        print(f"\n3. Memory usage: {memory_mb:.2f} MB")
        
        # Sample table
        print(f"\n4. Small sample table (first 5 rows):")
        print(df.head())
        
        # Missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\n5. Missing values:")
            print(missing_values[missing_values > 0])
        else:
            print(f"\n5. Missing values: None")
        
        # Unique values
        print(f"\n6. Unique values:")
        print(f"   States: {df['State_Name'].nunique()}")
        print(f"   Districts: {df['District_Name'].nunique()}")
        print(f"   Years: {df['Crop_Year'].nunique()} ({df['Crop_Year'].min()} to {df['Crop_Year'].max()})")
        print(f"   Seasons: {df['Season'].nunique()}")
        print(f"   Crops: {df['Crop'].nunique()}")
    
    def basic_statistics(self):
        """Calculate and display basic statistics."""
        print("\n" + "="*60)
        print("BASIC STATISTICS")
        print("="*60)
        
        df = self.cleaned_df
        numeric_cols = ['Area', 'Production', 'Yield']
        
        for col in numeric_cols:
            print(f"\n{col.upper()} Statistics:")
            print("-" * 40)
            
            # Basic statistics
            mean_val = df[col].mean()
            median_val = df[col].median()
            mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "No mode"
            
            min_val = df[col].min()
            max_val = df[col].max()
            range_val = max_val - min_val
            std_val = df[col].std()
            
            print(f"Mean: {mean_val:.4f}")
            print(f"Median: {median_val:.4f}")
            print(f"Mode: {mode_val}")
            print(f"Min: {min_val:.4f}")
            print(f"Max: {max_val:.4f}")
            print(f"Range: {range_val:.4f}")
            print(f"Standard Deviation: {std_val:.4f}")
            
            # Check for negative values
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                print(f"⚠️  Warning: {negative_count} negative values found in {col}")
            else:
                print(f"✅ No negative values in {col}")
            
            # Distribution shape
            skewness = stats.skew(df[col].dropna())
            kurtosis = stats.kurtosis(df[col].dropna())
            
            print(f"Skewness: {skewness:.3f} ({'Right-skewed' if skewness > 0.5 else 'Left-skewed' if skewness < -0.5 else 'Symmetric'})")
            print(f"Kurtosis: {kurtosis:.3f} ({'Heavy-tailed' if kurtosis > 3 else 'Light-tailed' if kurtosis < 3 else 'Normal'})")
    
    def create_visualizations(self):
        """Create multiple visualizations for the dataset."""
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        df = self.cleaned_df
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Histogram - Distribution of Yield
        plt.subplot(4, 3, 1)
        plt.hist(df['Yield'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Crop Yield', fontweight='bold', fontsize=14)
        plt.xlabel('Yield (tons per hectare)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 2. Bar Chart - Average Yield by Season
        plt.subplot(4, 3, 2)
        season_yield = df.groupby('Season')['Yield'].mean().sort_values(ascending=False)
        plt.bar(range(len(season_yield)), season_yield.values, color='lightgreen', alpha=0.8)
        plt.title('Average Yield by Season', fontweight='bold', fontsize=14)
        plt.xlabel('Season')
        plt.ylabel('Average Yield (tons per hectare)')
        plt.xticks(range(len(season_yield)), season_yield.index, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 3. Pie Chart - Distribution of Seasons
        plt.subplot(4, 3, 3)
        season_counts = df['Season'].value_counts()
        plt.pie(season_counts.values, labels=season_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Distribution of Seasons', fontweight='bold', fontsize=14)
        
        # 4. Boxplot - Yield Distribution by Season
        plt.subplot(4, 3, 4)
        season_data = [df[df['Season'] == season]['Yield'].values for season in df['Season'].unique()]
        plt.boxplot(season_data, labels=df['Season'].unique())
        plt.title('Yield Distribution by Season', fontweight='bold', fontsize=14)
        plt.xlabel('Season')
        plt.ylabel('Yield (tons per hectare)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 5. Line Chart - Yield Trend Over Years
        plt.subplot(4, 3, 5)
        yearly_yield = df.groupby('Crop_Year')['Yield'].mean().sort_index()
        plt.plot(yearly_yield.index, yearly_yield.values, marker='o', linewidth=2, markersize=4)
        plt.title('Yield Trend Over Years', fontweight='bold', fontsize=14)
        plt.xlabel('Year')
        plt.ylabel('Average Yield (tons per hectare)')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(yearly_yield.index, yearly_yield.values, 1)
        p = np.poly1d(z)
        plt.plot(yearly_yield.index, p(yearly_yield.index), "r--", alpha=0.8, linewidth=2, label=f'Trend (slope: {z[0]:.4f})')
        plt.legend()
        
        # 6. Scatter Plot - Area vs Production
        plt.subplot(4, 3, 6)
        plt.scatter(df['Area'], df['Production'], alpha=0.5, s=10)
        plt.title('Area vs Production', fontweight='bold', fontsize=14)
        plt.xlabel('Area (hectares)')
        plt.ylabel('Production (tons)')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['Area'], df['Production'], 1)
        p = np.poly1d(z)
        plt.plot(df['Area'], p(df['Area']), "r--", alpha=0.8, linewidth=2, label=f'Trend (slope: {z[0]:.4f})')
        plt.legend()
        
        # 7. Scatter Plot - Area vs Yield
        plt.subplot(4, 3, 7)
        plt.scatter(df['Area'], df['Yield'], alpha=0.5, s=10)
        plt.title('Area vs Yield', fontweight='bold', fontsize=14)
        plt.xlabel('Area (hectares)')
        plt.ylabel('Yield (tons per hectare)')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['Area'], df['Yield'], 1)
        p = np.poly1d(z)
        plt.plot(df['Area'], p(df['Area']), "r--", alpha=0.8, linewidth=2, label=f'Trend (slope: {z[0]:.6f})')
        plt.legend()
        
        # 8. Histogram - Distribution of Area
        plt.subplot(4, 3, 8)
        plt.hist(df['Area'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.title('Distribution of Area', fontweight='bold', fontsize=14)
        plt.xlabel('Area (hectares)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 9. Bar Chart - Top 10 States by Average Yield
        plt.subplot(4, 3, 9)
        state_yield = df.groupby('State_Name')['Yield'].mean().sort_values(ascending=False).head(10)
        plt.barh(range(len(state_yield)), state_yield.values, color='lightblue', alpha=0.8)
        plt.title('Top 10 States by Average Yield', fontweight='bold', fontsize=14)
        plt.xlabel('Average Yield (tons per hectare)')
        plt.yticks(range(len(state_yield)), state_yield.index)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        
        # 10. Scatter Plot - Production vs Yield
        plt.subplot(4, 3, 10)
        plt.scatter(df['Production'], df['Yield'], alpha=0.5, s=10)
        plt.title('Production vs Yield', fontweight='bold', fontsize=14)
        plt.xlabel('Production (tons)')
        plt.ylabel('Yield (tons per hectare)')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['Production'], df['Yield'], 1)
        p = np.poly1d(z)
        plt.plot(df['Production'], p(df['Production']), "r--", alpha=0.8, linewidth=2, label=f'Trend (slope: {z[0]:.6f})')
        plt.legend()
        
        # 11. Histogram - Distribution of Production
        plt.subplot(4, 3, 11)
        plt.hist(df['Production'], bins=50, alpha=0.7, color='lightyellow', edgecolor='black')
        plt.title('Distribution of Production', fontweight='bold', fontsize=14)
        plt.xlabel('Production (tons)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 12. Line Chart - Production Trend Over Years
        plt.subplot(4, 3, 12)
        yearly_production = df.groupby('Crop_Year')['Production'].mean().sort_index()
        plt.plot(yearly_production.index, yearly_production.values, marker='o', linewidth=2, markersize=4)
        plt.title('Production Trend Over Years', fontweight='bold', fontsize=14)
        plt.xlabel('Year')
        plt.ylabel('Average Production (tons)')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(yearly_production.index, yearly_production.values, 1)
        p = np.poly1d(z)
        plt.plot(yearly_production.index, p(yearly_production.index), "r--", alpha=0.8, linewidth=2, label=f'Trend (slope: {z[0]:.2f})')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('crop_production_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ All visualizations created and saved as 'crop_production_analysis.png'")
    
    def correlation_analysis(self):
        """Perform correlation analysis between numeric variables."""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        df = self.cleaned_df
        numeric_cols = ['Area', 'Production', 'Yield']
        
        # Create correlation matrix
        correlation_matrix = df[numeric_cols].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, fmt='.3f')
        plt.title('Correlation Matrix of Numeric Variables', fontweight='bold', fontsize=16)
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Correlation heatmap created and saved as 'correlation_matrix.png'")
        
        # Print correlation insights
        print("\nKey Correlation Insights:")
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:  # Avoid duplicate pairs
                    corr_val = correlation_matrix.loc[col1, col2]
                    if abs(corr_val) > 0.5:
                        print(f"• {col1} and {col2}: {corr_val:.3f} ({'Strong positive' if corr_val > 0 else 'Strong negative'} correlation)")
                    elif abs(corr_val) > 0.3:
                        print(f"• {col1} and {col2}: {corr_val:.3f} ({'Moderate positive' if corr_val > 0 else 'Moderate negative'} correlation)")
    
    def key_findings(self):
        """Generate key findings and insights."""
        print("\n" + "="*60)
        print("KEY FINDINGS")
        print("="*60)
        
        df = self.cleaned_df
        findings = []
        
        # 1. Yield statistics
        yield_mean = df['Yield'].mean()
        yield_median = df['Yield'].median()
        yield_std = df['Yield'].std()
        
        findings.append(f"• Average crop yield is {yield_mean:.2f} tons per hectare")
        findings.append(f"• Median yield is {yield_median:.2f} tons per hectare")
        findings.append(f"• Yield varies by ±{yield_std:.2f} tons per hectare (standard deviation)")
        
        # Yield range analysis
        yield_25 = df['Yield'].quantile(0.25)
        yield_75 = df['Yield'].quantile(0.75)
        findings.append(f"• 50% of yields fall between {yield_25:.2f} and {yield_75:.2f} tons per hectare")
        
        # 2. Area and Production analysis
        area_mean = df['Area'].mean()
        production_mean = df['Production'].mean()
        findings.append(f"• Average cultivated area is {area_mean:.0f} hectares")
        findings.append(f"• Average production is {production_mean:.0f} tons")
        
        # 3. Correlation findings
        area_prod_corr = df['Area'].corr(df['Production'])
        findings.append(f"• Area and Production have a correlation of {area_prod_corr:.3f}")
        
        if area_prod_corr > 0.7:
            findings.append("• Strong positive relationship between area cultivated and production output")
        elif area_prod_corr > 0.3:
            findings.append("• Moderate positive relationship between area cultivated and production output")
        else:
            findings.append("• Weak relationship between area cultivated and production output")
        
        # 4. Seasonal patterns
        best_season = df.groupby('Season')['Yield'].mean().idxmax()
        best_season_yield = df.groupby('Season')['Yield'].mean().max()
        findings.append(f"• {best_season} season shows highest average yield ({best_season_yield:.2f} tons/hectare)")
        
        # 5. Temporal trends
        yearly_yield = df.groupby('Crop_Year')['Yield'].mean().sort_index()
        if len(yearly_yield) > 1:
            z = np.polyfit(yearly_yield.index, yearly_yield.values, 1)
            trend_slope = z[0]
            if trend_slope > 0.01:
                findings.append("• Overall increasing trend in crop yields over time")
            elif trend_slope < -0.01:
                findings.append("• Overall decreasing trend in crop yields over time")
            else:
                findings.append("• Relatively stable crop yields over time")
        
        # 6. Data quality
        findings.append(f"• Dataset contains {len(df):,} records with no missing values")
        findings.append("• All numeric values are non-negative (logically consistent)")
        
        # Print findings
        for i, finding in enumerate(findings, 1):
            print(f"{i}. {finding}")
    
    def tools_used(self):
        """List the tools and libraries used for analysis."""
        print("\n" + "="*60)
        print("TOOLS USED")
        print("="*60)
        
        tools = [
            "• Pandas - Data manipulation and analysis",
            "• NumPy - Numerical computations",
            "• Matplotlib - Basic plotting and visualization",
            "• Seaborn - Statistical data visualization",
            "• SciPy - Statistical functions",
            "• Python - Programming language"
        ]
        
        for tool in tools:
            print(tool)
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting comprehensive crop production analysis...")
        print("="*60)
        
        # Load data
        self.load_data()
        if self.df is None:
            return
        
        # Preprocess data
        self.preprocess_data()
        
        # Run all analysis components
        self.dataset_summary()
        self.basic_statistics()
        self.create_visualizations()
        self.correlation_analysis()
        self.key_findings()
        self.tools_used()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Generated files:")
        print("• crop_production_analysis.png - Main visualizations")
        print("• correlation_matrix.png - Correlation analysis")

def main():
    """Main function to execute the analysis."""
    analyzer = CropProductionAnalyzer("crop_production.csv")
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
