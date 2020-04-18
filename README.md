# Segmentation for Arvato Financial Services

## Project Motivation

The goal of this project is to predict which individuals are most likely to convert into becoming customers for a mail-order sales company in Germany.

## File Description

- A Descriptive Jupyter Notebook

## Libraries/Dependencies Required
-Numpy, Pandas, Seaborn, ast, lightgbm , Matplotlib.pyplot
-Matplotlib.ticker
-random
-from helper import clean_data, clean_data_Kaggle, training_1, training_2, new_feature_1, new_feature_2, plot, print_importances, scree_plot
-from sklearn.model_selection import KFold, StratifiedKFold
-from sklearn.metrics import mean_squared_error
-from sklearn.preprocessing import Imputer
-from sklearn.preprocessing import StandardScaler
-from sklearn.metrics import roc_auc_score
-from sklearn.cluster import KMeans
-from sklearn.decomposition import PCA
-from sklearn.model_selection import train_test_split


## Results of the analysis

Results and discussion were published on Medium: [Segmentation Report for Arvato Financial Services](https://medium.com/@reni.sundrani/segmentation-report-for-arvato-financial-services-aa1f604c3fdf)

In this blog, we dive into a real life machine learning project provided by Arvato Financial Solutions, a Bertelsmann subsidiary:
- Investigated Demographics data of general population of Germany and data for customers of a mail-order company.
- Preprocessed the dataset based on column/feature property.
- Apply Unsupervised Learning Algorithms, namely PCA and KMeans to segment the population (into different clusters) to recommend the potential customers for the company.
- Took a deeper look at two main clusters and compare them by checking the differences of several randomly choose features.
- Apply Supervised Learning to predict whether or not a person became a customer of the company following the campaign.
- Investigate the most important feature trained by machine learning model and compare the feature distribution between target/non-target population.

