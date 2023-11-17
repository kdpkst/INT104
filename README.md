# Classification and Clustering based on Machine Learning Approach

This repository consists of two main folders: 
- **project** folder: contain the source code and dataset for the machine learning project.
- **notes** folder: contain code examples for the usage of Python libraries (e.g., scikit-learn) related to machine learning, data visualization and etc.

## Introduction

The project aims to classify and cluster students from different majors based on their marks. The provided CSV file (Data.csv) contains student records, including the mark for each question and corresponding student information, which will be used for training the classification and clustering models.

The following shows the columns in the dataset: 

| Columns     | Description                                        |
|------------ |----------------------------------------------------|
| **ID**      | A unique identifier for each student.              |
| **Q1**      | Mark for question 1.                               |
| **Q2**      | Mark for question 2.                               |
| **Q3**      | Mark for question 3.                               |
| **Q4**      | Mark for question 4.                               |
| **Q5**      | Mark for question 5.                               |
| **Programme** | The major in which the student is enrolled.      |

## Workflow
### Data Exploration and Preprocessing
1. **Data Loading and Preprocessing:**
   - Load the dataset from the CSV file (Data.csv)
   - Perform data preprocessing 
2. **Descriptive Statistics:**
   - Print descriptive statistics
   - Analyze correlation between variables
   - Explore data distribution and skewness  
3. **Data Visualization**
   - Generate histograms, density plots, box-and-whisker plots, correlation matrices plot, and scatter matrices.
   - First gather students with the same programme in one list. Then, draw scatter plots for median, mean, standard deviation, minimum, and maximum marks for each programme.

### Dimensionality Reduction
- Apply Principal Component Analysis (PCA).
- Perform Independent Component Analysis (ICA).
- Utilize t-Distributed Stochastic Neighbor Embedding (t-SNE).
- Implement Linear Discriminant Analysis (LDA).
- Use Non-Negative Matrix Factorization (NMF).
- Filter features based on variance.

### Classification 
- Train and evaluate Decision Tree, Random Forests, Logistic Regression, k-Nearest Neighbors, Naive Bayes, Support Vector Machines (SVM).

### Clustering 
- Apply k-Means, Agglomerative Clustering, and DBSCAN clustering algorithms.

## Requirements
- Python3
- anaconda

## Getting Started
To quickly explore the project, follow these steps:

1. Clone the repository and cd to project folder
2. run the file classify_and_cluster.py and observe the results.
3. For detailed information about project specificaton, data exploration and preprocessing,  dimensionality reduction, classification and clustering, please [Click](./project/report.pdf) to read the report. Feel free to modify the codes as needed.





