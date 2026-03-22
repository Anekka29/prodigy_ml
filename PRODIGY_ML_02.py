# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

# Display first rows
print("===== Dataset Preview =====")
print(data.head())

# Select features (important for clustering)
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Feature Scaling (IMPORTANT for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Find Optimal K using Elbow Method
# -------------------------------
wcss = []  # Within-cluster sum of squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure()
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method to Find Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.show()

# -------------------------------
# Apply K-Means (choose K=5)
# -------------------------------
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataset
data['Cluster'] = clusters

# Show clustered data
print("\n===== Clustered Data =====")
print(data.head())

# -------------------------------
# Visualization
# -------------------------------
plt.figure()

plt.scatter(
    X_scaled[:, 0], 
    X_scaled[:, 1], 
    c=clusters
)

# Plot centroids
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0], 
    centroids[:, 1], 
    s=200, 
    marker='X'
)

plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.title("Customer Segmentation using K-Means")
plt.show()

# -------------------------------
# Cluster Analysis (Mean values)
# -------------------------------
print("\n===== Cluster Summary =====")
print(data.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())

# -------------------------------
# Predict New Customer Cluster
# -------------------------------
print("\n===== Predict New Customer Segment =====")

try:
    income = float(input("Enter Annual Income (k$): "))
    score = float(input("Enter Spending Score (1-100): "))

    new_data = scaler.transform([[income, score]])
    cluster_pred = kmeans.predict(new_data)

    print(f"\n✅ Customer belongs to Cluster: {cluster_pred[0]}")

except:
    print("❌ Invalid input! Please enter numeric values.")