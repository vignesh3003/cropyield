import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, HuberRegressor, BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Try to import CatBoost (optional)
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not installed. Install with: pip install catboost")
    print("Continuing without CatBoost...\n")

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('cleaned_crop_yield_outliers_handled.csv')

# Display basic information about the dataset
print("\nDataset Information:")
print(f"Shape: {df.shape}")
print("\nColumns:", df.columns.tolist())
print("\nSample data:")
print(df.head())

# Data preprocessing
print("\nPreprocessing data...")

# Define features and target
X = df.drop(['Yield', 'Calculated_Yield'], axis=1)  # Using actual yield as target
y = df['Yield']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical features: {categorical_cols}")
print(f"Numerical features: {numerical_cols}")

# Create preprocessing pipeline for sklearn models
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Define models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Lasso Regression': Lasso(alpha=1.0, random_state=42),
    'Robust Regression': HuberRegressor(max_iter=500),  # Increased iterations
    'Bayesian Ridge': BayesianRidge(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Add CatBoost if available
if CATBOOST_AVAILABLE:
    models['CatBoost'] = CatBoostRegressor(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        random_state=42,
        verbose=False
    )
    print("✓ CatBoost included in model comparison\n")
else:
    print("✗ CatBoost skipped (not installed)\n")

# Dictionary to store results
results = {}

# Train and evaluate each model
print("\nTraining and evaluating models...")
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # CatBoost handles categorical features differently
    if name == 'CatBoost':
        # Pass categorical feature names directly to fit
        model.fit(X_train, y_train, cat_features=categorical_cols)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Store the model directly (not in a pipeline)
        trained_model = model
    else:
        # Create pipeline for sklearn models
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Store the pipeline
        trained_model = pipeline
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'model': trained_model
    }
    
    print(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'RMSE': [results[model]['RMSE'] for model in results.keys()],
    'MAE': [results[model]['MAE'] for model in results.keys()],
    'R2': [results[model]['R2'] for model in results.keys()]
})
comparison_df = comparison_df.sort_values('RMSE')

print("\n" + "="*60)
print("MODEL COMPARISON RESULTS")
print("="*60)
print(comparison_df.to_string(index=False))
print("="*60)

# Find the best model based on RMSE
best_model_name = min(results, key=lambda x: results[x]['RMSE'])
best_model = results[best_model_name]['model']
best_rmse = results[best_model_name]['RMSE']
best_r2 = results[best_model_name]['R2']

print(f"\n🏆 Best model: {best_model_name} with RMSE: {best_rmse:.4f} and R2: {best_r2:.4f}")

# Save the best model
joblib.dump(best_model, 'crop_yield_prediction_model.joblib')
print(f"Best model ({best_model_name}) saved as 'crop_yield_prediction_model.joblib'")

# Visualize model comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# RMSE comparison
comparison_df.plot(x='Model', y='RMSE', kind='bar', ax=axes[0], legend=False, color='steelblue')
axes[0].set_title('RMSE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('RMSE')
axes[0].tick_params(axis='x', rotation=45)

# MAE comparison
comparison_df.plot(x='Model', y='MAE', kind='bar', ax=axes[1], legend=False, color='coral')
axes[1].set_title('MAE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('MAE')
axes[1].tick_params(axis='x', rotation=45)

# R2 comparison
comparison_df.plot(x='Model', y='R2', kind='bar', ax=axes[2], legend=False, color='seagreen')
axes[2].set_title('R² Score Comparison (Higher is Better)', fontsize=12, fontweight='bold')
axes[2].set_ylabel('R² Score')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("Model comparison plot saved as 'model_comparison.png'")

# Feature importance for tree-based models
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    # Get feature names after preprocessing
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'cat':
            # For categorical features, get the one-hot encoded feature names
            for i, col in enumerate(columns):
                categories = transformer.categories_[i]
                feature_names.extend([f"{col}_{cat}" for cat in categories])
        else:
            # For numerical features, keep the original names
            feature_names.extend(columns)
    
    # Get feature importances
    importances = best_model.named_steps['model'].feature_importances_
    
    # Create a DataFrame for visualization
    if len(importances) == len(feature_names):
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
        plt.title(f'Top 15 Feature Importances - {best_model_name}')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("Feature importance plot saved as 'feature_importance.png'")

elif best_model_name == 'CatBoost':
    # Get feature importances from CatBoost
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title(f'Top 15 Feature Importances - {best_model_name}')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance plot saved as 'feature_importance.png'")

# Visualize actual vs predicted values
plt.figure(figsize=(10, 6))
if best_model_name == 'CatBoost':
    predictions = best_model.predict(X_test)
else:
    predictions = results[best_model_name]['model'].predict(X_test)

plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual Yield', fontsize=12)
plt.ylabel('Predicted Yield', fontsize=12)
plt.title(f'Actual vs Predicted Yield - {best_model_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300)
print("Actual vs Predicted plot saved as 'actual_vs_predicted.png'")

# Function to make predictions for new data
def predict_yield(crop, year, season, state, area, rainfall, fertilizer, pesticide):
    """
    Make yield predictions for new crop data
    
    Parameters:
    -----------
    crop : str
        Name of the crop
    year : float
        Crop year
    season : str
        Growing season
    state : str
        State name
    area : float
        Cultivation area
    rainfall : float
        Annual rainfall
    fertilizer : float
        Fertilizer used
    pesticide : float
        Pesticide used
    
    Returns:
    --------
    float
        Predicted yield
    """
    # Create a DataFrame with the input data
    decade = (year // 10) * 10
    
    # Determine season category
    season_mapping = {
        'Kharif': 'Monsoon',
        'Rabi': 'Winter',
        'Autumn': 'Autumn',
        'Summer': 'Summer',
        'Whole Year': 'Year-round',
        'Winter': 'Winter'
    }
    season_category = season_mapping.get(season, 'Unknown')
    
    data = pd.DataFrame({
        'Crop': [crop],
        'Crop_Year': [year],
        'Season': [season],
        'State': [state],
        'Area': [area],
        'Annual_Rainfall': [rainfall],
        'Fertilizer': [fertilizer],
        'Pesticide': [pesticide],
        'Season_Category': [season_category],
        'Decade': [decade],
        'Production': [0]  # Placeholder, not used for prediction
    })
    
    # Make prediction
    prediction = best_model.predict(data)
    return prediction[0]

print("\n✓ Model training and evaluation complete!")
print(f"✓ Total models trained: {len(models)}")
print(f"✓ Best performing model: {best_model_name}")