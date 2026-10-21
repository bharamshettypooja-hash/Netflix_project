# Netflix Movies and TV Shows Clustering

## Project Overview

This project focuses on analyzing and clustering Netflix movies and TV shows using **Unsupervised Machine Learning** techniques. The goal is to discover similarities between titles based on their content type, genre, title, and description.

The project uses Natural Language Processing (NLP) and clustering algorithms to group similar Netflix titles into meaningful clusters without requiring predefined labels.

## Objectives

- Analyze the Netflix Movies and TV Shows dataset.
- Perform Exploratory Data Analysis (EDA) to understand the dataset.
- Identify and handle missing values and duplicate records.
- Clean and preprocess text data.
- Apply TF-IDF vectorization to convert text into numerical features.
- Use dimensionality reduction to simplify the feature space.
- Apply K-Means clustering to group similar titles.
- Explore Agglomerative Hierarchical Clustering.
- Evaluate clustering performance using the Silhouette Score.
- Visualize the results and understand the characteristics of each cluster.

## Dataset

**Dataset name:** `NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv`

The dataset contains information about Netflix movies and TV shows, including:

 Column Description
 `show_id`  Unique identifier for each title 
 `type`  Type of content: Movie or TV Show 
 `title`  Name of the movie or TV show 
 `director`  Director of the title 
 `cast`  Cast members 
 `country`  Country associated with the title 
 `date_added`  Date the title was added to Netflix 
 `release_year`  Year the title was released 
 `rating` Content maturity rating 
 `duration`  Movie duration or number of TV show seasons 
 `listed_in`  Genres or categories 
 `description`  Short description of the title 

The dataset is used to explore patterns and similarities among Netflix titles.

**Dataset source:** Add the original dataset source URL here if you obtained it from an online platform. Otherwise, mention the source from which you received the dataset.

## Technologies Used

- Python — Programming language
- Pandas — Data manipulation and analysis
- NumPy — Numerical operations
- Matplotlib — Data visualization
- Seaborn — Statistical visualization
- Scikit-learn — Machine learning and preprocessing
- TF-IDF Vectorizer — Text feature extraction
- Truncated SVD — Dimensionality reduction
- K-Means Clustering — Unsupervised clustering algorithm
- Agglomerative Clustering — Hierarchical clustering algorithm
- SciPy — Hierarchical clustering and dendrogram visualization
- Jupyter Notebook — Interactive development environment

## Project Workflow

### 1. Data Loading

The dataset is loaded into a Pandas DataFrame using `read_csv()`.

### 2. Exploratory Data Analysis (EDA)

EDA is performed to understand the dataset and identify patterns, including:

- Distribution of movies and TV shows
- Most common countries
- Distribution of content ratings
- Release-year trends
- Most frequent genres
- Missing values and duplicate records

### 3. Data Cleaning and Preprocessing

The dataset is checked for missing values and duplicate records. Text columns are cleaned, and missing text values are handled to prepare the data for analysis.

Date-related fields and numerical features are also processed where applicable.

### 4. Feature Engineering

Relevant information, including content type, title, genres, and description, is combined into a text feature.

Additional features, such as the year a title was added and description length, may be created for exploratory analysis.

### 5. Text Vectorization Using TF-IDF

The **Term Frequency–Inverse Document Frequency (TF-IDF)** technique converts text into numerical features.

This enables the machine learning algorithms to compare titles based on their textual information.

### 6. Dimensionality Reduction

Truncated Singular Value Decomposition (Truncated SVD) reduces the dimensionality of the TF-IDF feature matrix while retaining useful information.

This makes the data more manageable for clustering and visualization.

### 7. K-Means Clustering

K-Means clustering is applied to group similar Netflix titles.

Different values of K are evaluated using:

- **Elbow Method:** Helps identify a suitable number of clusters by examining inertia.
- **Silhouette Score:** Measures how well data points fit within their assigned clusters compared with other clusters.

The selected number of clusters is then used to train the final K-Means model.

### 8. Agglomerative Hierarchical Clustering

Agglomerative Clustering is explored as an alternative clustering technique.

A dendrogram is used to visualize hierarchical relationships among a manageable sample of titles.

### 9. Cluster Analysis

The resulting clusters are examined to identify common genres, content characteristics, and example titles.

This helps interpret the similarities discovered by the clustering algorithms.

### 10. Results Export

The clustered dataset is saved as a CSV file named:

`netflix_clustered_results.csv`

The output includes the available dataset information and the generated cluster labels.

## Machine Learning Algorithms

### K-Means Clustering

K-Means partitions data into a selected number of clusters. It attempts to group titles with similar feature representations together.

### Agglomerative Hierarchical Clustering

Agglomerative Clustering builds a hierarchy of clusters by progressively merging similar data points or groups.

Both algorithms are unsupervised learning techniques because they do not require predefined target labels.

## Evaluation Metrics

### Inertia

Inertia measures the sum of squared distances between data points and their assigned cluster centers. It is used with the Elbow Method to help select the number of clusters.

### Silhouette Score

The Silhouette Score measures how similar a data point is to its own cluster compared with other clusters. A higher score generally indicates better-separated clusters.

The score should be interpreted alongside visualizations and cluster contents.

## Results and Expected Outcomes

The project aims to:

- Discover groups of similar Netflix movies and TV shows.
- Identify similarities based on textual descriptions and genres.
- Visualize content patterns and cluster assignments.
- Compare K-Means and Agglomerative Clustering.
- Generate a CSV file containing the cluster labels.

Note: The actual number of clusters, evaluation scores, and detailed findings should be added after running the notebook and reviewing its output.

## Business Applications

The clustering results could help support:

- **Content discovery:** Grouping similar titles for easier exploration.
- **Recommendation systems:** Providing an initial grouping of titles that may be useful in recommendation workflows.
- **Content analysis:** Understanding similarities across genres and descriptions.
- **Catalog organization:** Exploring ways to organize a large entertainment catalog.

These are potential applications; this project does not by itself implement a complete Netflix recommendation system.

## Project Structure

```text
Netflix-Movies-TV-Shows-Clustering/

 Netflix_Clustering_Project.ipynb
 NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv
 netflix_clustered_results.csv
 README.md
```

The CSV output file is generated when the notebook is executed successfully. Include it in the repository if you want to share the results.

## Installation and Setup

### Step 1: Install Python

Install Python 3 if it is not already installed.

### Step 2: Install the Required Libraries

Run the following command in a terminal or notebook cell:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

### Step 3: Download or Obtain the Dataset

Place `NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv` in the project folder.

### Step 4: Open the Project

Open the folder in VS Code or launch Jupyter Notebook.

### Step 5: Run the Notebook

Open `Netflix_Clustering_Project.ipynb` and execute the cells from top to bottom.

Ensure the dataset path in the notebook matches the location of the CSV file.

### Step 6: View the Results

Review the visualizations, clustering evaluation, and cluster analysis. After the notebook runs successfully, check the project folder for `netflix_clustered_results.csv`.

## Limitations

- Clustering quality depends on the features selected and the preprocessing steps.
- TF-IDF mainly captures textual similarity and does not fully understand the meaning of a description.
- The selected number of clusters may not represent the only meaningful grouping.
- Cluster labels are identifiers and do not automatically represent categories such as action, comedy, or drama.
- The results depend on the quality and completeness of the dataset.
- The project does not use Netflix's private viewing history or recommendation data.

## Future Improvements

- Experiment with alternative text representations, such as sentence embeddings.
- Tune clustering parameters and compare additional algorithms.
- Develop an interactive dashboard for exploring clusters.
- Create a simple interface that suggests similar titles based on cluster membership or text similarity.
- Add more robust cluster evaluation and visualizations.
- Improve the interpretation and naming of clusters using their most representative titles and genres.

## Conclusion

This project demonstrates how Unsupervised Machine Learning and Natural Language Processing can be applied to a real-world entertainment dataset.

By combining data cleaning, exploratory analysis, TF-IDF vectorization, dimensionality reduction, and clustering techniques, the project explores similarities among Netflix movies and TV shows without relying on predefined target labels.

The project provides practical experience in data analysis, text preprocessing, machine learning, visualization, and interpreting clustering results
