import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the California Housing dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add the target variable (MedHouseVal) as a column
df['MedHouseVal'] = data.target

# Step 2: Dataset preview
print("Dataset preview:")
print(df.head())  # Show first 5 rows of the dataset

# Step 3: Preprocess the data - Scale the features (excluding 'MedHouseVal')
features = df.drop('MedHouseVal', axis=1)  # Drop the target variable for clustering
scaler = MinMaxScaler()  # Min-Max scaler to scale the features
data_scaled = scaler.fit_transform(features)

# Step 4: Set up and train the Self-Organizing Map (SOM)
som = MiniSom(x=10, y=10, input_len=data_scaled.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(data_scaled)  # Initialize the SOM with random weights
som.train_batch(data_scaled, 100)  # Train the SOM for 100 iterations

# Step 5: Visualize the SOM clusters using heatmap
plt.figure(figsize=(10, 8))
plt.title("Self-Organizing Map Clusters")
sns.heatmap(som.distance_map().T, cmap='coolwarm', cbar=False, square=True)
plt.show()

# Step 6: Visualize the clusters on a 2D plane (U-Matrix)
plt.figure(figsize=(10, 8))
for i, x in enumerate(data_scaled):
    w = som.winner(x)  # Get the winning node for each data point
    plt.text(w[0], w[1], str(i), color=plt.cm.rainbow(i / len(data_scaled)), fontsize=10)

plt.title('SOM - Data Points in the Grid')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# Step 7: Cluster analysis - Add the cluster labels to the dataframe
win_map = som.win_map(data_scaled)  # Get the winning node for each data point
labels = np.zeros(len(data_scaled))  # Initialize an array to store the labels
for i, x in enumerate(data_scaled):
    w = som.winner(x)
    labels[i] = w[0] * 10 + w[1]  # Assign a label based on the SOM grid position

# Add the cluster labels to the dataframe
df['Cluster'] = labels.astype(int)

# Step 8: Analyze clusters - Calculate the mean of features per cluster
print("\nCluster analysis:")
cluster_analysis = df.groupby('Cluster').mean()  # Group by cluster and calculate mean of features
print(cluster_analysis)

# If you want to display all rows of cluster analysis (for 100 clusters, this will be all clusters)
print("\nFull Cluster Analysis:")
print(cluster_analysis.to_string())

# Step 9: Visualize the distribution of the target variable (MedHouseVal) per cluster
plt.figure(figsize=(10, 8))
sns.boxplot(x='Cluster', y='MedHouseVal', data=df)
plt.title("Distribution of House Value by Cluster")
plt.show()

# Step 10: Visualize the correlation matrix of features (optional)
plt.figure(figsize=(12, 8))
sns.heatmap(df.drop('MedHouseVal', axis=1).corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Features")
plt.show()

