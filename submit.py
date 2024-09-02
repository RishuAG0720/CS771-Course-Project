import numpy as np
import sklearn
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import time as tm

def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################

    # Use this method to train your model using training CRPs
    # X_train has 32 columns containing the challeenge bits
    # y_train contains the responses

    # THE RETURNED MODEL SHOULD BE A SINGLE VECTOR AND A BIAS TERM
    # If you do not wish to use a bias term, set it to 0
    X_train = my_map(X_train)
    y_train = np.where( y_train > 0, 1, -1 )
    clf = LogisticRegression(C=100,penalty='l2', max_iter=20,tol=1e-4);
    clf.fit(X_train, y_train)
    # Extract weight and bias terms
    w = clf.coef_.reshape( (529,) )
    b = clf.intercept_

    return w, b


def custom_feature_engineering(data):
    # Function for custom feature engineering
    flipped_data = np.flip(2 * data - 1, axis=1)
    cumulative_data = np.cumprod(flipped_data, axis=1)
    augmented_data = np.hstack((cumulative_data, np.ones((cumulative_data.shape[0], 1))))
    return augmented_data

def generate_pairwise_products(data):
    # Function to generate pairwise products
    result = []
    for row in data:
        pairwise_product = np.kron(row, row)
        n = len(row)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1).flatten()
        pairwise_product = pairwise_product[mask]
        result.append(pairwise_product)
    return np.array(result)

def add_bias_feature(data):
    # Function to add bias feature
    return np.concatenate((data, np.ones((data.shape[0], 1))), axis=1)

def my_map(X):
    # Main function for feature mapping
    engineered_features = custom_feature_engineering(X)
    pairwise_products = generate_pairwise_products(engineered_features)
    biased_features = add_bias_feature(pairwise_products)
    return biased_features

