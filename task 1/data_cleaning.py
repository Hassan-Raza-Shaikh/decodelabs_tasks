import pandas as pd
import numpy as np

def main():
    # Load the raw dataset
    data_path = '../Dataset for Data Analytics - Sheet1.csv'
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print("Initial shape:", df.shape)

    # 1. Handle Missing Data via statistical imputation (Median for numeric, Mode for categorical)
    print("Handling missing data...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    # Impute numeric missing values with Median
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Impute categorical missing values with Mode
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 2. Identify and neutralize outliers using IQR
    print("Neutralizing outliers using IQR...")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Capping (neutralizing) the outliers
        df[col] = np.clip(df[col], lower_bound, upper_bound)

    # 3. Engineer at least 3 new predictive features
    print("Engineering new features...")
    
    # Feature 1: OrderMonth (extracted from Date)
    if 'Date' in df.columns:
        # Convert to datetime to extract month
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['OrderMonth'] = df['Date'].dt.month
        # Fill any missing extracted months that might result from bad parsing
        df['OrderMonth'] = df['OrderMonth'].fillna(df['OrderMonth'].mode()[0])
    
    # Feature 2: HighValueCustomer (binary feature based on TotalPrice being above the 75th percentile)
    if 'TotalPrice' in df.columns:
        price_75 = df['TotalPrice'].quantile(0.75)
        df['HighValueOrder'] = (df['TotalPrice'] > price_75).astype(int)

    # Feature 3: AvgPricePerItemInCart (TotalPrice / ItemsInCart)
    if 'TotalPrice' in df.columns and 'ItemsInCart' in df.columns:
        # Calculate ratio and avoid division by zero
        df['AvgPricePerItemInCart'] = np.where(
            df['ItemsInCart'] > 0, 
            df['TotalPrice'] / df['ItemsInCart'], 
            0
        )

    # Save the cleaned and feature-engineered dataset
    output_path = 'cleaned_dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"Data cleaning and feature engineering complete. Final shape: {df.shape}")
    print(f"Saved clean data to {output_path}")

if __name__ == "__main__":
    main()
