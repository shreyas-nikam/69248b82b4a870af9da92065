
# Patient Cohort Discovery for Personalized Intervention: A Healthcare Administrator's Workflow

## 1. Setting Up the Environment: Required Libraries Installation

As a Healthcare Administrator, I often need advanced analytical tools to understand complex patient data. Before I can begin analyzing our patient population, I need to ensure all the necessary software libraries are installed. These libraries provide the mathematical and visualization capabilities required for cohort discovery.

```python
!pip install pandas numpy scikit-learn matplotlib seaborn plotly scipy
```

## 2. Importing Dependencies

With the libraries installed, the next crucial step is to import them into my analysis environment. This makes all the functions and classes from these libraries available for use in our patient cohort discovery workflow.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage
from IPython.display import display, Markdown

# Set aesthetic parameters for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
```

## 3. Simulating Patient Data and Initial Exploration

As a Healthcare Administrator, my first task is to get a clear picture of the patient data we're working with. Since real patient data is highly sensitive, I'll simulate a de-identified dataset that reflects the complexity and diversity of our actual patient population. This synthetic dataset will allow me to explore potential hidden patterns and understand the characteristics of different patient groups without compromising privacy. This step is crucial for identifying areas where personalized interventions could be beneficial.

```python
def generate_synthetic_patient_data(n_samples=750, random_state=42):
    """
    Generates a synthetic dataset of de-identified patient data with underlying clusters.

    Args:
        n_samples (int): The number of patient records to generate.
        random_state (int): Seed for random number generation for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame containing synthetic patient data.
    """
    np.random.seed(random_state)

    # Basic demographics and health metrics (some with underlying cluster tendencies)
    data = {
        'patient_id': range(1, n_samples + 1),
        'age': np.random.normal(loc=55, scale=15, size=n_samples).astype(int).clip(20, 90),
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.49, 0.03]),
        'medical_history_score': np.random.normal(loc=5, scale=2, size=n_samples).clip(1, 10),
        'num_diagnoses': np.random.poisson(lam=3, size=n_samples).clip(1, 10),
        'num_prescriptions': np.random.poisson(lam=5, size=n_samples).clip(0, 15),
        'visit_frequency_per_year': np.random.poisson(lam=4, size=n_samples).clip(1, 12),
        'insurance_type': np.random.choice(['Private', 'Public', 'None'], n_samples, p=[0.6, 0.35, 0.05]),
        'chronic_condition_A_present': np.random.randint(0, 2, n_samples),
        'chronic_condition_B_present': np.random.randint(0, 2, n_samples),
    }

    df = pd.DataFrame(data)

    # Introduce some subtle clustering structure for demonstration
    # Cluster 1: Younger, lower history score, public insurance, condition A more common
    df.loc[df['patient_id'] % 3 == 0, 'age'] = np.random.normal(loc=35, scale=7, size=(n_samples // 3)).astype(int).clip(20, 90)
    df.loc[df['patient_id'] % 3 == 0, 'medical_history_score'] = np.random.normal(loc=3, scale=1.5, size=(n_samples // 3)).clip(1, 10)
    df.loc[df['patient_id'] % 3 == 0, 'insurance_type'] = np.random.choice(['Public', 'Private'], size=(n_samples // 3), p=[0.7, 0.3])
    df.loc[df['patient_id'] % 3 == 0, 'chronic_condition_A_present'] = 1

    # Cluster 2: Older, higher visit freq, more prescriptions, private insurance
    df.loc[df['patient_id'] % 3 == 1, 'age'] = np.random.normal(loc=70, scale=8, size=(n_samples // 3)).astype(int).clip(20, 90)
    df.loc[df['patient_id'] % 3 == 1, 'visit_frequency_per_year'] = np.random.poisson(lam=7, size=(n_samples // 3)).clip(1, 12)
    df.loc[df['patient_id'] % 3 == 1, 'num_prescriptions'] = np.random.poisson(lam=8, size=(n_samples // 3)).clip(0, 15)
    df.loc[df['patient_id'] % 3 == 1, 'insurance_type'] = np.random.choice(['Private', 'Public'], size=(n_samples // 3), p=[0.8, 0.2])

    # Cluster 3: Middle-aged, moderate stats, mixed insurance, condition B more common
    df.loc[df['patient_id'] % 3 == 2, 'age'] = np.random.normal(loc=50, scale=10, size=(n_samples - (n_samples // 3)*2)).astype(int).clip(20, 90)
    df.loc[df['patient_id'] % 3 == 2, 'chronic_condition_B_present'] = 1
    
    return df

# Generate the data
patient_data = generate_synthetic_patient_data(n_samples=800)

# Display the first few rows
print("First 5 rows of the synthetic patient data:")
display(patient_data.head())

# Display basic statistics to understand data distribution
print("\nBasic statistics of the patient data:")
display(patient_data.describe(include='all'))

# Visualize distributions of key numerical features
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Distribution of Key Patient Attributes', y=1.02)
sns.histplot(patient_data['age'], bins=20, kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Age Distribution')
sns.histplot(patient_data['medical_history_score'], bins=20, kde=True, ax=axes[0, 1], color='lightcoral')
axes[0, 1].set_title('Medical History Score Distribution')
sns.histplot(patient_data['num_diagnoses'], bins=10, kde=True, ax=axes[0, 2], color='lightgreen')
axes[0, 2].set_title('Number of Diagnoses Distribution')
sns.histplot(patient_data['num_prescriptions'], bins=10, kde=True, ax=axes[1, 0], color='gold')
axes[1, 0].set_title('Number of Prescriptions Distribution')
sns.histplot(patient_data['visit_frequency_per_year'], bins=10, kde=True, ax=axes[1, 1], color='lightsteelblue')
axes[1, 1].set_title('Visit Frequency Per Year Distribution')
sns.countplot(data=patient_data, x='gender', ax=axes[1, 2], palette='viridis')
axes[1, 2].set_title('Gender Distribution')
plt.tight_layout()
plt.show()

# Visualize distribution of categorical features
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.countplot(data=patient_data, x='insurance_type', ax=axes[0], palette='magma')
axes[0].set_title('Insurance Type Distribution')
sns.countplot(data=patient_data, x='chronic_condition_A_present', ax=axes[1], palette='plasma')
axes[1].set_title('Chronic Condition A Presence')
plt.tight_layout()
plt.show()
```

### Explanation of Execution
The initial data generation and visualization give me a foundational understanding of our patient demographics and health metrics. I can see the range of ages, the distribution of medical history scores, and the prevalence of different chronic conditions. This early exploration helps me identify potential biases or areas of high variability that might influence our cohort discovery, ensuring the subsequent steps are well-informed. For instance, if certain age groups are overrepresented, I should consider how that might affect clustering.

## 4. Data Preprocessing for Clustering Algorithms

Our raw patient data contains both numerical and categorical information. For clustering algorithms to work effectively, all features must be in a numerical format and on a comparable scale. As a Healthcare Administrator, I know that accurate data preparation is paramount for reliable analytical insights. I will transform categorical variables using one-hot encoding and scale numerical features to prevent features with larger values from dominating the distance calculations. This process ensures that patient similarities are measured fairly across all attributes.

For scaling numerical features, we typically use standardization, where each feature's values are transformed such that they have a mean of 0 and a standard deviation of 1. The formula for standardization for a data point $x$ in a feature with mean $\mu$ and standard deviation $\sigma$ is:
$$ x_{\text{scaled}} = \frac{x - \mu}{\sigma} $$
For categorical features, one-hot encoding converts each category into a new binary feature (0 or 1).

```python
# Identify categorical and numerical features
categorical_features = ['gender', 'insurance_type']
numerical_features = ['age', 'medical_history_score', 'num_diagnoses', 
                      'num_prescriptions', 'visit_frequency_per_year', 
                      'chronic_condition_A_present', 'chronic_condition_B_present']

# Create a column transformer for preprocessing
# One-hot encode categorical features, scale numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop' # Drop 'patient_id' and other unselected columns
)

# Create a preprocessing pipeline
preprocessing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit and transform the data
patient_features_processed = preprocessing_pipeline.fit_transform(patient_data.drop('patient_id', axis=1))

# Get feature names after one-hot encoding for better interpretability
ohe_feature_names = preprocessing_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(ohe_feature_names)

# Convert the processed data back to a DataFrame for easier handling
patient_processed_df = pd.DataFrame(patient_features_processed, columns=all_feature_names)

print("Shape of the processed patient data:", patient_processed_df.shape)
print("\nFirst 5 rows of the processed patient data:")
display(patient_processed_df.head())
```

### Explanation of Execution
After preprocessing, our data is now in a uniform, scaled numerical format, ready for clustering. The increase in the number of columns reflects the one-hot encoding of categorical variables. By standardizing numerical features, I've ensured that 'age' doesn't disproportionately influence clustering compared to 'num_prescriptions' simply because its values are numerically larger. This standardized representation allows the clustering algorithms to accurately identify patient similarities based on true underlying patterns, not just differences in scale.

## 5. Dimension Reduction for Visualizing Patient Relationships

As a Healthcare Administrator, directly visualizing relationships within our high-dimensional patient data (many features) is challenging. To gain an intuitive understanding of how patients might naturally group together, I'll use dimension reduction techniques. Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) will transform our data into a lower-dimensional space (2D or 3D) that can be easily plotted. This visualization step is crucial for spotting preliminary groupings and confirming whether our chosen clustering algorithms are likely to find meaningful cohorts.

### Principal Component Analysis (PCA)
PCA identifies the principal components, which are new variables that are linear combinations of the original variables and are orthogonal to each other. These components capture the maximum variance in the data. The algorithm involves:
1.  **Centering the data:** $X_{\text{centered}} = X - \text{mean}(X)$
2.  **Computing covariance matrix:** $C = \frac{1}{n-1} X_{\text{centered}}^T X_{\text{centered}}$
3.  **Performing eigendecomposition:** $C = V \Lambda V^T$, where $V$ contains eigenvectors (principal components) and $\Lambda$ contains eigenvalues (variance explained).
4.  **Selecting top $k$ eigenvectors:** To reduce dimensions.
5.  **Transforming data:** Projecting $X_{\text{centered}}$ onto the selected eigenvectors.

### t-Distributed Stochastic Neighbor Embedding (t-SNE)
t-SNE is particularly effective for visualizing high-dimensional data by giving each data point a location in a two- or three-dimensional map. It models each high-dimensional object by a two- or three-dimensional point in such a way that similar objects are modeled by nearby points and dissimilar objects are modeled by distant points.

```python
def plot_dimension_reduction(data, method_name, n_components=2):
    """
    Applies dimension reduction and plots the result.
    """
    if method_name == 'PCA':
        reducer = PCA(n_components=n_components, random_state=42)
        reduced_data = reducer.fit_transform(data)
        print(f"PCA explained variance ratio for {n_components} components: {reducer.explained_variance_ratio_.sum():.2f}")
    elif method_name == 't-SNE':
        # t-SNE can be computationally intensive; reduce samples for larger datasets if needed
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=30, n_iter=1000, learning_rate='auto')
        reduced_data = reducer.fit_transform(data)
    else:
        raise ValueError("Method must be 'PCA' or 't-SNE'")

    reduced_df = pd.DataFrame(reduced_data, columns=[f'{method_name} Component {i+1}' for i in range(n_components)])

    if n_components == 2:
        fig = px.scatter(reduced_df, x=f'{method_name} Component 1', y=f'{method_name} Component 2',
                         title=f'{method_name} 2D Projection of Patient Data',
                         hover_name=patient_data['patient_id'],
                         height=600)
    elif n_components == 3:
        fig = px.scatter_3d(reduced_df, x=f'{method_name} Component 1', y=f'{method_name} Component 2', z=f'{method_name} Component 3',
                            title=f'{method_name} 3D Projection of Patient Data',
                            hover_name=patient_data['patient_id'],
                            height=700)
    
    fig.show()
    return reduced_data

# Apply PCA and t-SNE for 2D visualization
print("--- Applying PCA ---")
pca_2d = plot_dimension_reduction(patient_processed_df, 'PCA', n_components=2)
print("\n--- Applying t-SNE ---")
tsne_2d = plot_dimension_reduction(patient_processed_df, 't-SNE', n_components=2)
```

### Explanation of Execution
The 2D plots from PCA and t-SNE give me an initial visual sense of how our patient population is structured. PCA, being linear, might show broader trends, while t-SNE, often better at preserving local neighborhood structures, could reveal more intricate, non-linear groupings. Seeing distinct "blobs" or concentrations of points in these visualizations suggests that our patient data likely contains discernible cohorts, which is a positive sign for the subsequent clustering analysis. The explained variance for PCA tells me how much information is retained, which helps in judging its representativeness.

## 6. Discovering Patient Cohorts with K-Means Clustering

Now that our data is preprocessed and we've had a preliminary visual inspection, it's time to apply clustering algorithms. As a Healthcare Administrator, my goal is to identify distinct patient cohorts whose needs and responses to treatments differ. K-Means clustering is an excellent starting point due to its simplicity and efficiency for numerical data. A crucial decision for K-Means is determining the optimal number of clusters, $k$. I'll use the Elbow Method and Silhouette Score to guide this choice, ensuring we find meaningful, compact, and well-separated groups.

The K-Means algorithm (Figure 1 in the provided text) operates by:
1.  **Initializing $k$ centroids** randomly.
2.  **Iteratively assigning each data point** to the cluster whose centroid it is closest to. The distance $d(x_i, \mu_j)$ is typically Euclidean.
3.  **Updating the centroid** of each cluster to be the mean of all points assigned to it: $\mu_j = \text{mean of all points assigned to cluster } j$.
4.  **Repeating steps 2 and 3** until the centroids no longer change significantly (convergence) or a maximum number of iterations is reached.

The Silhouette Score for a data point $i$ is calculated as:
$$ s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}} $$
where $a(i)$ is the average distance from $i$ to other points in its own cluster (cohesion) and $b(i)$ is the minimum average distance from $i$ to points in any other cluster (separation). A higher Silhouette Score indicates better-defined clusters.

```python
def evaluate_kmeans_k(data, max_k):
    """
    Evaluates K-Means clustering for a range of k values using Elbow Method (SSE)
    and Silhouette Score.
    """
    sse = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
        score = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(score)

    # Plot Elbow Method
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, sse, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')

    # Plot Silhouette Scores
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Optimal K')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.tight_layout()
    plt.show()

    # Find k with the highest silhouette score
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal K based on highest Silhouette Score: {optimal_k_silhouette}")
    return optimal_k_silhouette

# Determine optimal k for K-Means
print("--- Determining Optimal K for K-Means ---")
optimal_k_kmeans = evaluate_kmeans_k(patient_processed_df, max_k=10)

# Apply K-Means with the chosen k
kmeans_model = KMeans(n_clusters=optimal_k_kmeans, random_state=42, n_init=10)
kmeans_labels = kmeans_model.fit_predict(patient_processed_df)
patient_data['kmeans_cluster'] = kmeans_labels

# Add cluster labels to the PCA and t-SNE reduced data for visualization
pca_2d_df = pd.DataFrame(pca_2d, columns=['PCA Component 1', 'PCA Component 2'])
pca_2d_df['Cluster'] = kmeans_labels.astype(str)

tsne_2d_df = pd.DataFrame(tsne_2d, columns=['t-SNE Component 1', 't-SNE Component 2'])
tsne_2d_df['Cluster'] = kmeans_labels.astype(str)

# Visualize K-Means clusters on PCA and t-SNE projections
fig_pca = px.scatter(pca_2d_df, x='PCA Component 1', y='PCA Component 2', color='Cluster',
                     title=f'K-Means Clusters (k={optimal_k_kmeans}) on PCA 2D Projection',
                     hover_name=patient_data['patient_id'], height=600)
fig_pca.show()

fig_tsne = px.scatter(tsne_2d_df, x='t-SNE Component 1', y='t-SNE Component 2', color='Cluster',
                      title=f'K-Means Clusters (k={optimal_k_kmeans}) on t-SNE 2D Projection',
                      hover_name=patient_data['patient_id'], height=600)
fig_tsne.show()

# Display cluster sizes
print(f"\nK-Means Cluster Sizes (k={optimal_k_kmeans}):")
display(patient_data['kmeans_cluster'].value_counts().sort_index())
```

### Explanation of Execution
The Elbow Method and Silhouette Score plots provided insights into the most suitable number of clusters, $k$. A clear 'elbow' in the SSE plot, combined with a peak in the Silhouette Score, indicated an optimal `k` value. The scatter plots, colored by K-Means clusters on the PCA and t-SNE projections, visually confirmed that the algorithm successfully identified distinct patient groups. This visual validation is critical for me to trust that the mathematical groupings align with observable separations in the data. The cluster sizes give me an initial understanding of the distribution of patients across these new cohorts.

## 7. Exploring Alternative Patient Cohorts with Hierarchical Clustering

While K-Means is effective, its assumption of spherical clusters might not always capture the true underlying structure of our patient data. As a Healthcare Administrator, I want to explore alternative clustering methods to ensure we have a robust understanding of our patient population. Hierarchical clustering, specifically agglomerative clustering, offers a different perspective by building a tree-like structure (dendrogram) that shows how patient groups merge at different levels of similarity. This allows for a more flexible definition of clusters and can reveal nested relationships.

Agglomerative hierarchical clustering (Figure 3 in the provided text) works "bottom-up":
1.  **Initialize:** Each data point is its own cluster.
2.  **Compute distance matrix:** Distances between all pairs of points.
3.  **Iteratively merge:** Find the closest pair of clusters and merge them into a new cluster. The "closest" is defined by a **linkage method**:
    *   **Single linkage:** $\min\{d(x, y) : x \in C_i, y \in C_j\}$ (minimum distance between points in two clusters).
    *   **Complete linkage:** $\max\{d(x, y) : x \in C_i, y \in C_j\}$ (maximum distance).
    *   **Average linkage:** $\text{mean}\{d(x, y) : x \in C_i, y \in C_j\}$ (average distance).
    *   **Ward linkage:** Minimizes the total within-cluster variance. This is often preferred as it tends to produce more compact, spherical clusters.
4.  **Repeat step 3** until all points belong to a single cluster or the desired number of clusters is reached. A dendrogram records the merge history.

```python
# Perform hierarchical clustering
# Using 'ward' linkage as it often produces well-balanced clusters and minimizes variance within clusters.
linked = linkage(patient_processed_df, method='ward')

# Plot the dendrogram to visualize the hierarchical structure
plt.figure(figsize=(18, 8))
# The `color_threshold` can be adjusted. We'll set it to cut at a distance that yields optimal_k_kmeans clusters.
# This requires inspecting the dendrogram or calculating the height for k clusters.
# For simplicity, we'll try to visually estimate a good threshold or just let matplotlib color automatically
# and then use AgglomerativeClustering with n_clusters.
# For consistency with optimal_k_kmeans, let's determine the threshold dynamically:
# The distance at which the last (optimal_k_kmeans - 1) merges happened before forming optimal_k_kmeans clusters.
# linked is (N-1)x4 matrix, column 2 is distance. sorted(linked[:,2]) gives distances at merges.
# The threshold for `k` clusters is typically `linked[-(k), 2]` if using an inverted dendrogram.
# However, for consistency and clear visualization, we just plot the full dendrogram and then extract clusters.
# For explicit coloring to show k clusters, color_threshold can be set to the distance of the (N-k)th merge.
# This might require some calculation for the `linked` array.
# For now, let's keep it simple and just let it auto-color and then extract clusters using n_clusters.

# To set color_threshold for exactly optimal_k_kmeans clusters, find the (N - optimal_k_kmeans)th merge distance
# This is typically linked[-(optimal_k_kmeans), 2] if linked is sorted by merge distance.
# linked is usually sorted by order of merges, so the N-k'th merge is linked[len(linked) - optimal_k_kmeans + 1, 2]
# If optimal_k_kmeans is 3, we want the distance of the (N-3)th merge. The last 2 merges form 3 clusters.
# So, the threshold is the distance of the (N-optimal_k_kmeans)th merge.
if optimal_k_kmeans > 1: # ensure there are enough merges to make sense
    threshold_distance = linked[-(optimal_k_kmeans-1), 2]
else:
    threshold_distance = 0 # Default if only 1 cluster is desired (not common)


dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=False, # Too many leaves for 800 patients, hide counts
           color_threshold=threshold_distance,
           above_threshold_color='grey' # Color merges above threshold differently
          )
plt.title(f'Hierarchical Clustering Dendrogram (cut for {optimal_k_kmeans} clusters)')
plt.xlabel('Patient Samples')
plt.ylabel('Distance')
plt.xticks([]) # Hide x-axis labels for individual samples
plt.show()

# Extract clusters from hierarchical clustering at the same 'k' as K-Means for comparison
hierarchical_model = AgglomerativeClustering(n_clusters=optimal_k_kmeans, linkage='ward')
hierarchical_labels = hierarchical_model.fit_predict(patient_processed_df)
patient_data['hierarchical_cluster'] = hierarchical_labels

# Add hierarchical cluster labels to the PCA and t-SNE reduced data for visualization
pca_2d_df_hac = pd.DataFrame(pca_2d, columns=['PCA Component 1', 'PCA Component 2'])
pca_2d_df_hac['Cluster'] = hierarchical_labels.astype(str)

tsne_2d_df_hac = pd.DataFrame(tsne_2d, columns=['t-SNE Component 1', 't-SNE Component 2'])
tsne_2d_df_hac['Cluster'] = hierarchical_labels.astype(str)

# Visualize Hierarchical Clusters on PCA and t-SNE projections
fig_pca_hac = px.scatter(pca_2d_df_hac, x='PCA Component 1', y='PCA Component 2', color='Cluster',
                         title=f'Hierarchical Clusters (k={optimal_k_kmeans}) on PCA 2D Projection',
                         hover_name=patient_data['patient_id'], height=600)
fig_pca_hac.show()

fig_tsne_hac = px.scatter(tsne_2d_df_hac, x='t-SNE Component 1', y='t-SNE Component 2', color='Cluster',
                          title=f'Hierarchical Clusters (k={optimal_k_kmeans}) on t-SNE 2D Projection',
                          hover_name=patient_data['patient_id'], height=600)
fig_tsne_hac.show()

# Display cluster sizes
print(f"\nHierarchical Cluster Sizes (k={optimal_k_kmeans}):")
display(patient_data['hierarchical_cluster'].value_counts().sort_index())
```

### Explanation of Execution
The dendrogram provided a visual representation of how patient clusters are formed by merging individual patients and smaller groups. This tree structure is invaluable for understanding the granularity of relationships. Cutting the dendrogram at a specific 'distance' level revealed distinct clusters, which I chose to align with the `k` value from K-Means for direct comparison. The subsequent scatter plots showed how these hierarchical clusters manifest in the reduced-dimensional space. Observing similarities or differences between the K-Means and Hierarchical clustering visualizations helps me assess the robustness of the identified cohorts, informing which clustering solution might be more appropriate for our specific healthcare planning needs.

## 8. Evaluating and Comparing Clustering Solutions

As a Healthcare Administrator, I can't solely rely on visual inspection to determine the best patient cohorts. I need quantitative metrics to evaluate the quality of our clustering solutions and to compare the results from K-Means and Hierarchical clustering. This will help me choose the most reliable grouping for developing targeted interventions. We'll use the Silhouette Score to assess how well-defined and separated the clusters are for each algorithm, and the Adjusted Rand Index (ARI) to compare the agreement between the two different clustering results.

### Silhouette Score
The Silhouette Score, as introduced previously, quantifies how similar an object is to its own cluster compared to other clusters. A higher score signifies better-defined clusters.

### Adjusted Rand Index (ARI)
The Adjusted Rand Index (ARI) is a measure of the similarity between two data clusterings. Unlike the raw Rand Index, ARI adjusts for chance, meaning a random assignment of clusters will yield an ARI close to 0, and perfect agreement will yield an ARI of 1. Negative values indicate worse-than-random agreement. This is particularly useful when comparing two different clustering algorithms applied to the same dataset.
$$ \text{ARI} = \frac{\text{RI} - E[\text{RI}]}{\max(\text{RI}) - E[\text{RI}]} $$
where $\text{RI}$ is the Rand Index, and $E[\text{RI}]$ is the expected Rand Index under the null hypothesis of random clustering. $\max(\text{RI})$ is the maximum possible value of the Rand Index.

```python
# Calculate Silhouette Score for K-Means
kmeans_silhouette = silhouette_score(patient_processed_df, patient_data['kmeans_cluster'])
print(f"K-Means Silhouette Score: {kmeans_silhouette:.3f}")

# Calculate Silhouette Score for Hierarchical Clustering
hierarchical_silhouette = silhouette_score(patient_processed_df, patient_data['hierarchical_cluster'])
print(f"Hierarchical Clustering Silhouette Score: {hierarchical_silhouette:.3f}")

# Compare K-Means and Hierarchical Clustering using Adjusted Rand Index (ARI)
# ARI measures the similarity between the two clustering assignments
ari_score = adjusted_rand_score(patient_data['kmeans_cluster'], patient_data['hierarchical_cluster'])
print(f"Adjusted Rand Index (ARI) between K-Means and Hierarchical Clustering: {ari_score:.3f}")

# Visualize individual Silhouette scores for chosen K-Means clusters
kmeans_sample_silhouette_values = silhouette_samples(patient_processed_df, patient_data['kmeans_cluster'])

plt.figure(figsize=(10, 7))
y_lower = 10
for i in range(optimal_k_kmeans):
    # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
    ith_cluster_silhouette_values = \
        kmeans_sample_silhouette_values[patient_data['kmeans_cluster'] == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = plt.cm.nipy_spectral(float(i) / optimal_k_kmeans)
    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    # Label the silhouette plots with their cluster numbers at the middle
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

plt.axvline(x=kmeans_silhouette, color="red", linestyle="--", label=f'Avg Silhouette Score: {kmeans_silhouette:.3f}')
plt.title("Silhouette plot for K-Means Clustering")
plt.xlabel("Silhouette coefficient values")
plt.ylabel("Cluster label")
plt.legend()
plt.yticks([]) # Clear the yaxis labels / ticks
plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
plt.show()
```

### Explanation of Execution
The Silhouette Scores quantified the compactness and separation of the clusters from both K-Means and Hierarchical algorithms. A higher score for one method indicates it formed more distinct and well-defined groups. The individual Silhouette plot for K-Means provided a deeper look, showing how well each patient fits into their assigned cluster, and highlighting any clusters with particularly low scores, which might suggest misclassified patients or fuzzy boundaries. The Adjusted Rand Index (ARI) provided a direct measure of agreement between the two different clustering solutions. If ARI is high, it means both algorithms are largely agreeing on the patient groupings, which boosts my confidence in the discovered cohorts. If it's low, it suggests the choice of algorithm significantly impacts the resulting patient segmentation, requiring further investigation to understand why. This quantitative evaluation helps me make an informed decision about which clustering result is most reliable for our strategic planning.

## 9. Characterizing Patient Cohorts for Actionable Insights

To move from abstract clusters to actionable healthcare strategies, I, as a Healthcare Administrator, need to understand the defining characteristics of each patient cohort. This involves analyzing the average values of key patient attributes within each cluster. By creating "profile charts," I can clearly visualize what makes each group unique – are they typically older, have more chronic conditions, visit frequently, or have a specific insurance type? This deep dive into cohort profiles is essential for developing highly targeted interventions that address the specific needs of each patient segment.

```python
def plot_cluster_profiles(df_with_clusters, cluster_column, numerical_features, categorical_features, optimal_k):
    """
    Generates profile charts for each cluster based on original (unscaled) features.
    """
    print(f"\n--- Profiling Clusters for {cluster_column.replace('_', ' ').title()} ---")
    
    # Calculate mean for numerical features
    cluster_means_numerical = df_with_clusters.groupby(cluster_column)[numerical_features].mean()
    
    # Calculate value counts for categorical features (proportions)
    categorical_proportions = {}
    for col in categorical_features:
        categorical_proportions[col] = df_with_clusters.groupby(cluster_column)[col].value_counts(normalize=True).unstack(fill_value=0)

    # Plotting numerical feature profiles
    num_cols_to_plot = len(numerical_features)
    n_rows_num = (num_cols_to_plot + 2) // 3
    fig, axes = plt.subplots(n_rows_num, 3, figsize=(18, n_rows_num * 5))
    axes = axes.flatten()

    for i, feature in enumerate(numerical_features):
        cluster_means_numerical[feature].plot(kind='bar', ax=axes[i], title=f'Mean {feature.replace("_", " ").title()} by Cohort', rot=45, color=sns.color_palette("tab10", optimal_k))
        axes[i].set_ylabel(f'Mean {feature.replace("_", " ").title()}')
        axes[i].set_xlabel('Cohort ID')
    
    for j in range(i + 1, len(axes)): # Hide empty subplots
        fig.delaxes(axes[j])
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'Numerical Feature Profiles by {cluster_column.replace("_", " ").title()}', fontsize=16)
    plt.show()

    # Plotting categorical feature profiles
    cat_cols_to_plot = len(categorical_features)
    n_rows_cat = (cat_cols_to_plot + 1) // 2
    fig_cat, axes_cat = plt.subplots(n_rows_cat, 2, figsize=(15, n_rows_cat * 5))
    axes_cat = axes_cat.flatten()

    for i, feature in enumerate(categorical_features):
        categorical_proportions[feature].plot(kind='bar', ax=axes_cat[i], stacked=True, title=f'Proportion of {feature.replace("_", " ").title()} by Cohort', rot=45, cmap='viridis')
        axes_cat[i].set_ylabel('Proportion')
        axes_cat[i].set_xlabel('Cohort ID')

    for j in range(i + 1, len(axes_cat)):
        fig_cat.delaxes(axes_cat[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'Categorical Feature Profiles by {cluster_column.replace("_", " ").title()}', fontsize=16)
    plt.show()

# Assuming K-Means was the primary choice, or whichever had the best silhouette score
# Ensure 'patient_data' DataFrame contains the 'kmeans_cluster' column
plot_cluster_profiles(patient_data, 'kmeans_cluster', numerical_features, categorical_features, optimal_k_kmeans)
```

### Explanation of Execution
The profile charts vividly illustrate the distinguishing characteristics of each patient cohort. For instance, I can immediately see if Cohort 0 is predominantly older patients with multiple chronic conditions, while Cohort 1 consists of younger patients with fewer prescriptions. These visual summaries transform raw data into clear, digestible insights. By comparing the mean values and proportions of features across clusters, I can confidently articulate what defines each patient group. This detailed characterization forms the backbone of any personalized intervention strategy, allowing us to tailor care based on the specific needs of each identified segment of our patient population.

## 10. Generating a Patient Cohort Report: Actionable Strategies

As a Healthcare Administrator, the ultimate goal of this analysis is to translate data insights into practical, actionable strategies for improving patient care and resource allocation. Based on the detailed profiles of our identified patient cohorts, I can now develop specific recommendations. This "Patient Cohort Report" will summarize the key characteristics of each cluster and outline potential personalized intervention strategies. This deliverable is critical for moving beyond a one-size-fits-all approach and for optimizing our healthcare services.

We will select the clustering solution (e.g., K-Means) that demonstrated better quality based on the Silhouette Scores and agreement with ARI. For this report, let's assume K-Means provided the most robust and interpretable clusters.

```python
def generate_cohort_report(df_with_clusters, processed_data_for_silhouette, cluster_column, numerical_features, categorical_features):
    """
    Generates a summary report for each patient cohort with actionable intervention strategies.
    """
    num_clusters = df_with_clusters[cluster_column].nunique()
    report_sections = []

    report_sections.append("# Patient Cohort Report: Personalized Intervention Strategies\n")
    report_sections.append("This report outlines distinct patient cohorts identified through unsupervised clustering, along with tailored recommendations for improving care pathways and resource allocation.\n")
    report_sections.append(f"**Total Patients Analyzed:** {len(df_with_clusters)}  \n")
    report_sections.append(f"**Clustering Method Used:** {cluster_column.replace('_', ' ').title()} (k={num_clusters})  \n")
    
    # Recalculate silhouette score here in case the dataframe changed
    current_silhouette_score = silhouette_score(processed_data_for_silhouette, df_with_clusters[cluster_column])
    report_sections.append(f"**Overall Silhouette Score:** {current_silhouette_score:.3f}\n")

    # Calculate mean for numerical features and mode for categorical features
    cluster_summary_numerical = df_with_clusters.groupby(cluster_column)[numerical_features].mean()
    cluster_summary_categorical_mode = {}
    for col in categorical_features:
        # Get the value counts for each category within each cluster
        prop_df = df_with_clusters.groupby(cluster_column)[col].value_counts(normalize=True).unstack(fill_value=0)
        # Find the mode (most frequent category) for each cluster
        cluster_summary_categorical_mode[col] = prop_df.idxmax(axis=1) # Get the column name (category) with max proportion
        
    for i in range(num_clusters):
        report_sections.append(f"\n## Cohort {i}:\n")
        report_sections.append(f"**Number of Patients:** {len(df_with_clusters[df_with_clusters[cluster_column] == i])}\n")

        report_sections.append("\n### Key Characteristics:\n")
        
        # Numerical characteristics
        report_sections.append("**Numerical Features (Mean):**\n")
        for feature in numerical_features:
            report_sections.append(f"- **{feature.replace('_', ' ').title()}:** {cluster_summary_numerical.loc[i, feature]:.2f}\n")

        # Categorical characteristics
        report_sections.append("\n**Categorical Features (Most Prevalent):**\n")
        for feature in categorical_features:
            report_sections.append(f"- **{feature.replace('_', ' ').title()}:** {cluster_summary_categorical_mode[feature].loc[i]}\n")
            
        report_sections.append("\n### Potential Personalized Intervention Strategies:\n")
        
        # Example logic for generating recommendations (customize based on observed clusters)
        # These are illustrative and should be adapted to real-world domain knowledge.
        
        # Get mean values for the current cluster
        age_mean = cluster_summary_numerical.loc[i, 'age']
        medical_score_mean = cluster_summary_numerical.loc[i, 'medical_history_score']
        num_prescriptions_mean = cluster_summary_numerical.loc[i, 'num_prescriptions']
        visit_freq_mean = cluster_summary_numerical.loc[i, 'visit_frequency_per_year']
        
        # Get proportion of chronic conditions for the current cluster
        cc_a_prop = df_with_clusters[df_with_clusters[cluster_column] == i]['chronic_condition_A_present'].mean()
        cc_b_prop = df_with_clusters[df_with_clusters[cluster_column] == i]['chronic_condition_B_present'].mean()

        if age_mean < 45 and medical_score_mean < 4 and num_prescriptions_mean < 3:
            report_sections.append("- **Focus on preventative care & health education:** Younger, healthier patients could benefit from programs promoting long-term wellness, healthy lifestyle choices, and early screening for potential risks (e.g., vaccination drives, healthy eating workshops).\n")
            report_sections.append("- **Digital engagement:** Utilize mobile apps or online portals for appointment reminders, health resources, and telehealth consultations, given their likely tech-savviness.\n")
        elif age_mean > 60 and (cc_a_prop > 0.6 or cc_b_prop > 0.6) and num_prescriptions_mean > 5:
            report_sections.append("- **Comprehensive chronic disease management:** Older patients with multiple chronic conditions and prescriptions require coordinated care, medication reconciliation, and support for managing complex conditions. Implement disease-specific management programs.\n")
            report_sections.append("- **Geriatric assessments & fall prevention programs:** Tailor services to address common issues in elderly populations, improving safety and quality of life.\n")
            report_sections.append("- **Home health support & community outreach:** Explore options for in-home care or support groups to reduce hospital readmission rates and provide social support.\n")
        elif visit_freq_mean > 7 and medical_score_mean > 6:
            report_sections.append("- **Care coordination for high-utilizers:** Patients with high visit frequency and complex medical histories need robust care coordination to avoid fragmented care and optimize treatment plans. Assign dedicated care navigators.\n")
            report_sections.append("- **Specialized clinics or diagnostic pathways:** Offer dedicated support for their complex needs, potentially reducing unnecessary visits by streamlining diagnostics.\n")
        else:
            report_sections.append("- **Standardized care pathways with personalization options:** For general cohorts, focus on efficient, evidence-based care while offering flexibility for individual needs identified through regular screenings.\n")
            report_sections.append("- **Patient satisfaction surveys:** Regularly collect feedback to identify emerging needs, improve service delivery, and enhance patient experience across all touchpoints.\n")
            
    return "".join(report_sections)

# Generate the report using K-Means clusters
# (Assuming K-Means was selected as the primary clustering result)
patient_cohort_report_content = generate_cohort_report(patient_data, patient_processed_df, 'kmeans_cluster', numerical_features, categorical_features)

# Display the report using Markdown
display(Markdown(patient_cohort_report_content))
```

### Explanation of Execution
The Patient Cohort Report provides a concise, yet comprehensive, summary of our patient population, broken down into distinct, actionable segments. For each cohort, the report details average attributes and suggests tailored intervention strategies. As a Healthcare Administrator, this is my key deliverable. For example, if Cohort 0 is characterized by young age and low medical history scores, the report recommends preventative care programs. If Cohort 1 consists of older patients with many prescriptions, it suggests comprehensive chronic disease management. This report serves as a strategic document, guiding our organization in allocating resources more effectively, designing targeted health campaigns, and ultimately, delivering more personalized and effective care to each patient. The explicit recommendations ensure that the data-driven insights translate directly into tangible improvements in patient outcomes and operational efficiency.
