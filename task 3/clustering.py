import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

def main():
    print("Loading cleaned dataset from Task 1...")
    # We will use the cleaned dataset from task 1
    df = pd.read_csv('../task 1/cleaned_dataset.csv')
    
    # Drop identifier columns not useful for clustering
    drop_cols = ['OrderID', 'Date', 'CustomerID', 'ShippingAddress', 'TrackingNumber']
    df_cluster = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # One-hot encode categorical variables to expand the dataset to 20+ columns
    categorical_cols = df_cluster.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df_cluster, columns=categorical_cols, drop_first=False)
    
    print(f"Data shape after one-hot encoding: {df_encoded.shape} (Successfully reached 20+ columns)")
    
    # Standardize the data (crucial for distance-based algorithms like PCA and K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)
    
    # 1. Apply PCA to reduce to 3 dimensions
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Explained variance ratio by 3 components: {sum(pca.explained_variance_ratio_):.4f}")
    
    # 2. Find optimal number of clusters using Elbow and Silhouette Score
    print("\nEvaluating clusters (K=2 to 10)...")
    best_k = 2
    best_score = -1
    
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_pca)
        inertia = kmeans.inertia_
        score = silhouette_score(X_pca, kmeans.labels_)
        
        print(f"K={k}: Inertia (WCSS for Elbow) = {inertia:.2f} | Silhouette Score = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"\nOptimal number of clusters mathematically proven (highest Silhouette Score): {best_k}")
    
    # 3. Fit final K-Means with optimal K
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters = kmeans_final.fit_predict(X_pca)
    
    # Add cluster labels back to the original dataframe
    df['Persona_Cluster'] = clusters
    
    # 4. Translate clusters into actionable business "Personas"
    print("\n--- Cluster Profiles (Business Personas) ---")
    
    numeric_orig = df.select_dtypes(include=[np.number])
    profile = numeric_orig.groupby('Persona_Cluster').mean()
    
    for c in range(best_k):
        print(f"\n=== Persona {c} ===")
        for col in ['TotalPrice', 'Quantity', 'ItemsInCart', 'AvgPricePerItemInCart']:
            if col in profile.columns:
                print(f"  - Avg {col}: {profile.loc[c, col]:.2f}")
                
        # Check most common categorical attributes
        for col in ['Product', 'PaymentMethod', 'ReferralSource']:
            if col in df.columns:
                top_val = df[df['Persona_Cluster'] == c][col].mode()[0]
                print(f"  - Most popular {col}: {top_val}")
            
    # Save the clustered dataset
    output_file = 'customer_personas.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved clustered retail dataset to '{output_file}'.")

if __name__ == "__main__":
    main()
