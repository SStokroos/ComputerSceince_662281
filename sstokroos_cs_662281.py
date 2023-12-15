#!/usr/bin/env python
# coding: utf-8

# In[27]:


import json
import pandas as pd
import re
from collections import Counter
from itertools import chain
from difflib import SequenceMatcher
import os
import numpy as np
from tqdm import tqdm
import random
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering
os.chdir('C:/Users/stens/OneDrive/Bureaublad/Master/Q2/CS')
# Load data
with open('TVs-all-merged.json') as data_json:  
    data = json.load(data_json)  

weight_1 = 0.6
weight_2 = 0.4

def standardize_measurements(text):
    units = {
        'inch': [r' inches', "'", '”', 'in', ' inch', ' inches', 'Inches', ' Inches', '-Inch', '-Inches', '-inch', '-inches', '"', '""'],
        'hz': [r' hz', 'hz', ' HZ.', ' Hz', 'Hz.', 'Hz'],
        'cdma': [r' cd/mâ²', ' cdm2', 'cdm2', 'lm', ' lm', ' cd/m²', 'cd/m²', ' cd/m2', 'nit'],
        'lb': [r' lb', ' lbs.', ' lb.', ' pounds', 'pounds', 'lb', 'lbs.', 'lb.', 'lb'],
        'watt': [r' w', 'w', ' watt', 'watt'],
        'kg': [r' kg', 'kg', 'KG', ' KG', 'Kg'],
        'p': [r' p', 'p', 'i/p', ' i/p', '/24p']
    }

    for unit, patterns in units.items():
        for pattern in patterns:
            text = text.replace(pattern, unit + ' ')
    return text

def format_brand(text):
    return text.lower()

def format_value(text):
    replacements = [('+', ''), ('-', ''), ('without', '-'), ('with', '+'), ('and', ' '), ('|', ' '), (' x ', 'x'), 
                    ('no', '0'), ('yes', '1'), ('false', '0'), ('true', '1'), (',', ''), ('.', ''), (')', ''), 
                    ('(', ''), ('/', ''), ('&#', '')]
    for old, new in replacements:
        text = text.replace(old, new)
    text = standardize_measurements(text)
    return text.lower()

def format_shop_name(text):
    return text.lower().replace('.', '').replace(' ', '')

def format_value_reading(text):
    replacements = [('-', ''), ('+', ''), ('with', '+'), ('without', '-'), ('|', ' '), (' and ', ' '), ('no', '0'), 
                    (' x ', 'x'), ('yes', '1'), ('false', '0'), ('true', '1'), (',', ''), ('.', ''), ('(', ''), 
                    (')', ''), ('/', ''), ('+', ''), ('&#', ''), ('-', '')]
    for old, new in replacements:
        text = text.replace(old, new)
    text = standardize_measurements(text)
    return text.lower()

def format_title(text):
    text = standardize_measurements(text)
    text = text.replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('.0', '').replace('/', '').replace(',', '').replace('inchwatt', 'inch')
    return ' '.join(word for word in text.split() if any(char.isdigit() for char in word))




from difflib import SequenceMatcher
import operator

def calculate_similarity_score(set_a, set_b):
    if min(len(set_a), len(set_b)) == 0:
        return 0.0
    else:
        return len(set_a.intersection(set_b)) / min(len(set_a), len(set_b))

def compare_feature_similarity(feature_a, feature_b, weight_1, weight_2):
    return weight_1 * SequenceMatcher(None, feature_a, feature_b).ratio() + weight_2 * calculate_similarity_score(set(feature_a.split()), set(feature_b.split()))

def aggregate_key_features(data):

    feature_counts = {}
    for products in data.values():
        for product in products:
            for feature in product['featuresMap'].keys():
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
    return {feature: count for feature, count in feature_counts.items() if count > 1}


def match_and_group_features(features, threshold, weight_1, weight_2):

    sorted_features = sorted(features.items(), key=operator.itemgetter(1), reverse=True)
    grouped_features = {}

    for i, (feature, _) in enumerate(sorted_features):
        if i == 0:
            grouped_features[feature] = []
            continue

        for j, (compared_feature, _) in enumerate(sorted_features):
            if compare_feature_similarity(feature, compared_feature, weight_1, weight_2) < threshold:
                continue

            if compared_feature in grouped_features:
                grouped_features[compared_feature].append(feature)
                break

        else:
            grouped_features[feature] = []

    return {feature: variations for feature, variations in grouped_features.items() if variations}


def standardize_feature_keys(text, feature_variations):

    for key, variations in feature_variations.items():
        for variation in variations:
            text = text.replace(variation, key)
    return text

def format_feature_key(text, feature_variations):

    text = standardize_feature_keys(text, feature_variations)
    return text.lower().replace(' ', '')


shop_names = ['amazoncom', 'neweggcom', 'bestbuycom', 'thenerdsnet']

key_features = aggregate_key_features(data)
grouped_features = match_and_group_features(key_features, 0.8, weight_1, weight_2)



def extract_potential_id(title, min_length):

    title = title.replace('(', '').replace(')', '')
    digit_including_words = ' '.join(word for word in title.split() if any(char.isdigit() for char in word)).split()
    longest_word = max(digit_including_words, key=len, default='')

    return longest_word if len(longest_word) >= min_length else 'None'

def standardize_shop_names(shops, standardize_function):

    return [standardize_function(shop) for shop in shops]

def aggregate_brand_occurrences(data, feature_key='Brand'):

    brand_occurrences = {}
    for products in data.values():
        for product in products:
            features_map = product.get('featuresMap', {})
            brand = features_map.get(feature_key) or features_map.get(feature_key + ' Name') or features_map.get(feature_key + ' Name:')
            if brand:
                brand_occurrences[brand] = brand_occurrences.get(brand, 0) + 1

    return list(brand_occurrences.keys())

cleaned_shop_names = standardize_shop_names(shop_names, format_shop_name)
list_of_brands = aggregate_brand_occurrences(data)
cleaned_brands = [format_brand(brand) for brand in list_of_brands]




import pandas as pd

def create_and_clean_dataset(data, find_model_id, clean_title, clean_shop, clean_brand, standardize_keys, clean_key, clean_value, feature_variations):
    all_product_data = []  # List to hold all product data dictionaries

    for key in data.keys():
        for product in data[key]:
            product_data = process_product(product, find_model_id, clean_title, clean_shop, clean_brand, standardize_keys, clean_key, clean_value, feature_variations)
            product_data['key'] = key
            all_product_data.append(product_data)

    # Use pandas.concat to create the DataFrame from the list of dictionaries
    dataset = pd.concat([pd.DataFrame([data]) for data in all_product_data], ignore_index=True)
    return dataset


def process_product(product, find_model_id, clean_title, clean_shop, clean_brand, standardize_keys, format_key, clean_value, feature_variations):
    potential_model_id = find_model_id(product['title'], 6)
    title = clean_title(product['title']).split()
    shop_name = clean_shop(product['shop'])
    key_value_pairs = []

    for key, value in product['featuresMap'].items():
        if key in ['Brand', 'Brand Name', 'Brand Name:']:
            key_value_pairs.append(clean_brand(value))
        else:
            standardized_key = standardize_keys(key, feature_variations)
            if standardized_key in feature_variations:
                cleaned_key = format_key(standardized_key, feature_variations)  # Use format_key with both arguments
                cleaned_value = clean_value(value)
                key_value_pairs.append(cleaned_key + ':' + cleaned_value)

    return {
        'potential_model_id': potential_model_id,
        'title': title,
        'shop': shop_name,
        'key_value_pairs': key_value_pairs
    }

dataframe = create_and_clean_dataset(data, extract_potential_id, format_title, format_shop_name, format_brand, standardize_feature_keys, format_feature_key, format_value, grouped_features)
dataframe

cleanShops = ['amazoncom', 'neweggcom', 'bestbuycom', 'thenerdsnet']

def create_word_count_dict(data, title_col, kvp_col):
    """Calculates the frequency of each word in the dataset."""
    word_count = Counter()

    # Update word count with words from 'title'
    for title in data[title_col]:
        word_count.update(title)

    # Update word count with both keys and values from 'key_value_pairs'
    if kvp_col in data.columns:
        for kvp_list in data[kvp_col]:
            for kvp in kvp_list:
                # Split the string into key and value, and count both
                parts = kvp.split(':')
                word_count.update(parts)

    return word_count


def remove_shop_names(word_count, shop_list):
    for shop in shop_list:
        word_count.pop(shop, None)
    return word_count

def retain_frequent_words(word_count, min_occurrence=2):
    return {word: count for word, count in word_count.items() if count >= min_occurrence}

def weight_brand_model_ids(modwords, brands, model_ids, brand_weight, model_id_weight):
    weighted_words = modwords.copy()
    for _ in range(brand_weight - 1):
        weighted_words.extend(brands)
    for _ in range(model_id_weight - 1):
        weighted_words.extend(model_ids)
    return weighted_words

def create_binary_binmat_2(data, modwords, title_col, kvp_col):
    product_indices = data.index
    binmat = np.zeros((len(modwords), len(product_indices)))

    for p_idx in (product_indices):
        title = data[title_col][p_idx]
        kvp = data[kvp_col][p_idx]
        for w_idx, word in enumerate(modwords):
            if word in title or any(word in k.split(':')[0] for k in kvp):
                binmat[w_idx][p_idx] = 1

    return binmat

def create_binary_binmat(data, modwords, title_col):
    product_indices = data.index
    binmat = np.zeros((len(modwords), len(product_indices)))

    for p_idx in product_indices:
        title = data[title_col][p_idx]
        for w_idx, word in enumerate(modwords):
            if word in title:
                binmat[w_idx][p_idx] = 1

    return binmat


def permfind(n, r):
    b = 1
    while r * b <= n * 0.5:
        b += 1
    p = r * b
    return p, r, b


def is_number_prime(num):
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True

def get_next_prime(start_num):
    current_num = start_num
    while not is_number_prime(current_num):
        current_num += 1
    return current_num

def generate_hash_functions(num_hashes, matrix_length):
    prime = get_next_prime(num_hashes)
    a_values = random.sample(range(matrix_length), num_hashes)
    b_values = random.sample(range(matrix_length), num_hashes)

    def single_hash(a, b, x):
        return (a * x + b) % prime

    return [lambda x, a=a, b=b: single_hash(a, b, x) for a, b in zip(a_values, b_values)]

def create_signature_matrix(binmat, num_hashes):
    permutations, rows, bands = permfind(len(binmat), num_hashes)
    signature_matrix = np.ones((permutations, binmat.shape[1])) * float('inf')
    hash_functions = generate_hash_functions(permutations, binmat.shape[0])

    for row_index in tqdm(range(binmat.shape[0]), desc= 'minhash'):
        hash_values = [hash_func(row_index) for hash_func in hash_functions]
        for col_index in range(binmat.shape[1]):
            if binmat[row_index][col_index] == 1:
                for perm_index in range(permutations):
                    signature_matrix[perm_index][col_index] = min(hash_values[perm_index], signature_matrix[perm_index][col_index])

    return signature_matrix, rows, bands

def generate_lsh_buckets(signature_matrix, reps, bands, rows):
    buckets = {}
    for repetition in range(reps):
        shuffled_matrix = signature_matrix.copy()
        np.random.shuffle(shuffled_matrix)
        for product_idx in range(shuffled_matrix.shape[1]):
            for band_idx in range(bands):
                band_signature = str(repetition) + ' ' + str(band_idx) + ' ' + str([round(item) for item in shuffled_matrix[:, product_idx][band_idx:band_idx + rows]])
                buckets.setdefault(band_signature, []).append(product_idx)

    return {key: value for key, value in buckets.items() if len(value) >= 2}

def define_candidate_pairs_from_buckets(buckets):
    candidates = {}
    for bucket_values in buckets.values():
        for pair in combinations(bucket_values, 2):
            ordered_pair = tuple(sorted(pair))
            candidates[ordered_pair] = candidates.get(ordered_pair, 0) + 1

    return candidates

def find_true_pairs(dataset, key_col):
    true_pairs = []
    real_count = 0
    for i in dataset.index:
        for j in dataset.index:
            if i >= j:
                continue
            if dataset[key_col][i] == dataset[key_col][j]:
                true_pairs.append((i, j))
                real_count += 1
    return true_pairs, real_count

def calculate_wlsh_performance(candidates, dataset, key_col, true_pairs_count): 
    correct_matches = 0
    for candidate_pair in candidates:
        if dataset[key_col][candidate_pair[0]] == dataset[key_col][candidate_pair[1]]:
            correct_matches += 1

    total_possible_combinations = sum(range(len(dataset) + 1))
    FOC = len(candidates)/ total_possible_combinations
    PQ = correct_matches / len(candidates) if candidates else 0
    PC = correct_matches / true_pairs_count if true_pairs_count else 0
    F1_star = (2 * PQ * PC) / (PQ + PC) if PQ + PC else 0

    return PQ, PC, F1_star, FOC, correct_matches, total_possible_combinations

def remove_same_shop_candidates(candidates_2, dataset, shop_col):
    candidates_to_remove = [key for key in candidates_2 if dataset[shop_col][key[0]] == dataset[shop_col][key[1]]]
    for key in candidates_to_remove:
        del candidates_2[key]

    return candidates_2


def remove_different_brand_candidates(candidates_3, dataset, title_col, kvp_col, brands):
    product_brands = {}
    for idx in dataset.index:
        product_brands[idx] = next((brand for brand in brands if brand in dataset[title_col][idx] or brand in dataset[kvp_col][idx]), 'none')

    candidates_to_remove = [key for key in candidates_3 if product_brands[key[0]] != 'none' and product_brands[key[1]] != 'none' and product_brands[key[0]] != product_brands[key[1]]]

    for key in candidates_to_remove:
        del candidates_3[key]

    return candidates_3

def calculate_jaccard_similarity(signature_matrix, a, b):
    v1, v2 = signature_matrix[:, a], signature_matrix[:, b]
    jaccard_distance = np.sum(v1 != v2) / len(v1)
    return 1 - jaccard_distance


def create_distance_matrix(signature_matrix, candidates):
    num_products = signature_matrix.shape[1]
    distance_matrix = np.ones((num_products, num_products)) * 1000

    for i in range(num_products):
        for j in range(num_products):
            if (i, j) in candidates:
                distance_matrix[i][j] = calculate_jaccard_similarity(signature_matrix, i, j)

    return distance_matrix

def perform_clustering_and_generate_results(distance_matrix, threshold):
    clustering = AgglomerativeClustering(metric='precomputed', linkage='single', distance_threshold=threshold, n_clusters=None)
    cluster_labels = clustering.fit_predict(distance_matrix)

    buckets_cl = {}
    for idx, cluster_label in enumerate(cluster_labels):
        buckets_cl.setdefault(cluster_label, []).append(idx)

    filtered_buckets = {k: v for k, v in buckets_cl.items() if len(v) >= 2}
    
    result_pairs = [tuple(sorted(pair)) for bucket in filtered_buckets.values() for pair in combinations(bucket, 2)]
    return result_pairs

def calculate_performance_metrics(true_pairs, result_pairs):
    TP = set(result_pairs).intersection(set(true_pairs))
    print(str(len(TP)))
    FP = set(result_pairs) - set(true_pairs)
    FN = set(true_pairs) - set(result_pairs)

    precision = len(TP) / (len(TP) + len(FP)) if TP or FP else 0
    recall = len(TP) / (len(TP) + len(FN)) if TP or FN else 0
    F1 = 2 * (precision * recall) / (precision + recall) if precision + recall else 0

    return precision, recall, F1


bootstrap_iterations = 5
rvals =  [3, 4, 5, 6, 7, 8, 10, 11, 12]  # example values for rpbLSH

# Lists to hold results for each bootstrap iteration
bootstrap_FOC_values = [[] for _ in range(bootstrap_iterations)]
bootstrap_PQ_values = [[] for _ in range(bootstrap_iterations)]
bootstrap_PC_values = [[] for _ in range(bootstrap_iterations)]
bootstrap_F1_star_values = [[] for _ in range(bootstrap_iterations)]
bootstrap_F1_val = [[] for _ in range(bootstrap_iterations)]

for i in range(bootstrap_iterations):
    print(f"Bootstrap Iteration: {i+1}")

    for rval in rvals:
        sample_size = int(len(dataframe) * 0.63)

        DataSetB = dataframe.sample(n=sample_size, replace=False).reset_index(drop=True)

        # Create word count dictionary
        word_count_dict = create_word_count_dict(DataSetB, 'title', 'key_value_pairs')

        # Remove shop names
        word_count_no_shops = remove_shop_names(word_count_dict, cleanShops)

        # Retain words occurring more than once
        frequent_words = retain_frequent_words(word_count_no_shops)

        print('Number of model words before weighting:', len(frequent_words))
        # Weight brand and model IDs
        w_modwords = weight_brand_model_ids(list(frequent_words.keys()), cleaned_brands, dataframe['potential_model_id'].tolist(), 2, 3)

        # Create binary input matrix for LSH
        binmat = create_binary_input_matrix_2(DataSetB, w_modwords, 'title', 'key_value_pairs')

        # Output the count of model words after weighting
        print('Number of model words after weighting:', len(w_modwords))

        signature_matrix, rows, bands = create_signature_matrix(binmat, rval)


        buckets = generate_lsh_buckets(signature_matrix, 2, bands, rows)
        candidates = define_candidate_pairs_from_buckets(buckets)
        
        # Print the number of candidate pairs
        print(f"Number of candidate pairs: {len(candidates)}")

        candidates_2 = remove_same_shop_candidates(candidates, DataSetB, 'shop')
        candidates_3 = remove_different_brand_candidates(candidates_2, DataSetB, 'title', 'key_value_pairs', cleaned_brands)
        print(f"PAIRS LEFT: {len(candidates_3)}")
        true_pairs, real_count = find_true_pairs(DataSetB, 'key')
        PQ, PC, F1_star, FOC, correct_matches, total_possible_combinations = calculate_wlsh_performance(candidates, DataSetB, 'key', real_count)
        bootstrap_FOC_values[i].append(FOC)
        bootstrap_PQ_values[i].append(PQ)
        bootstrap_PC_values[i].append(PC)
        bootstrap_F1_star_values[i].append(F1_star)
        bootstrap_F1_val[i].append(F1)
        DistanceMatrix = create_distance_matrix(signature_matrix, candidates)
        result_pairs = perform_clustering_and_generate_results(DistanceMatrix, 0.86)
        precision, recall, F1 = calculate_performance_metrics(true_pairs, result_pairs)
        bootstrap_F1_val[i].append(F1)    



# In[35]:





# In[33]:


import matplotlib.pyplot as plt

# Function to calculate means for each value across bootstrap iterations
def calculate_means(values):
    return np.mean(values, axis=0)

# Calculating means
mean_FOC = calculate_means(bootstrap_FOC_values)
mean_PQ = calculate_means(bootstrap_PQ_values)
mean_F1_star = calculate_means(bootstrap_F1_star_values)
mean_PC = calculate_means(bootstrap_PC_values)
mean_F1 = calculate_means(bootstrap_F1_val)

# Plotting the means

# PQ vs FOC
plt.figure(figsize=(10, 6))
plt.plot(mean_FOC, mean_PQ, marker='o', color='blue', label='PQ')
plt.xlabel('FOC')
plt.ylabel('PQ')
plt.grid(True)
plt.legend()
plt.show()

# F1_star vs FOC
plt.figure(figsize=(10, 6))
plt.plot(mean_FOC, mean_F1_star, marker='s', color='green', label='F1*')
plt.xlabel('FOC')
plt.ylabel('F1*')
plt.grid(True)
plt.legend()
plt.show()

# PC vs FOC
plt.figure(figsize=(10, 6))
plt.plot(mean_FOC, mean_PC, marker='^', color='red', label='PC')
plt.xlabel('FOC')
plt.ylabel('PC')
plt.grid(True)
plt.legend()
plt.show()

# F1 vs FOC
plt.figure(figsize=(10, 6))
plt.plot(mean_FOC, mean_F1, marker='^', color='orange', label='F1')
plt.xlabel('FOC')
plt.ylabel('F1')
plt.grid(True)
plt.xlim(0.00, 0.01)
plt.legend()
plt.show()



# In[39]:


mean_F1_no_duplicates = calculate_means(bootstrap_F1_val_no_duplicates)

# Ensuring both mean_F1_no_duplicates and mean_FOC have the same length for plotting
min_length = min(len(mean_F1_no_duplicates), len(mean_FOC))
mean_F1_no_duplicates = mean_F1_no_duplicates[:min_length]
mean_FOC = mean_FOC[:min_length]

# Plotting F1 vs FOC
plt.figure(figsize=(10, 6))
plt.plot(mean_FOC, mean_F1_no_duplicates, marker='^', color='orange', label='F1')
plt.xlabel('Mean FOC')
plt.ylabel('Mean F1')
plt.grid(True)
plt.legend()
plt.show()


# In[40]:


mean_FOC


# In[ ]:




