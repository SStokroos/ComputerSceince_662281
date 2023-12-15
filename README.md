# ComputerScience_662281
Duplicate detection
This project presents an advanced algorithm for product matching in e-commerce datasets. It focuses on aggregating and comparing key features, brand occurrences, and model identifiers, utilizing techniques like Jaccard similarity, MinHash, and Locality-Sensitive Hashing (LSH) for efficient and accurate product matching.

# Description
The script processes large e-commerce datasets, handling product features and matching similar products based on various attributes. It includes comprehensive data cleaning, feature standardization, similarity scoring, and binary matrix creation for LSH.

# Features
Data Standardization: Converts product features into a uniform format, including measurement units and brand names.
Feature Matching: Groups similar features using a similarity threshold, essential for identifying comparable products.
Brand and Shop Filtering: Excludes matches from the same shop and filters products based on brand consistency, ensuring diverse and relevant results.
Locality-Sensitive Hashing (LSH): Implements LSH to efficiently find similar items in large datasets.
Agglomerative Clustering: Utilizes clustering techniques based on Jaccard similarity to group similar products.
Performance Metrics: Calculates Precision, Recall, and F1-Score, providing insights into the accuracy and efficiency of the product matching process.

# What's Included
sstokroos_cs_66281.py: The main Python script with all the matching logic.
TVs-all-merged.json:  JSON dataset file illustrating the data.
