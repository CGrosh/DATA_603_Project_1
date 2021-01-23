# Import necessary libraries 
import numpy as np 
import pandas as pd 
import scipy.io as io 
import scipy.stats 
import math
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def euclidean(x1, x2):
    dist = math.sqrt(sum([(a-b)**2 for a, b in zip(x1, x2)]))
    return dist

def knn_fit(x_train, y_train, x_test, k):

    train_x = np.array(x_train)
    train_y = np.array(y_train)

    pred_arr = []
    dist_arrs = []
    for test_val in range(len(x_test)):
        test_arr = np.array(x_test.iloc[test_val]).reshape(1, 7)

        dists_x = np.linalg.norm(train_x- test_arr, axis=1, ord=2)
        joined_dists = np.column_stack((train_y, dists_x))

        sorts = joined_dists[joined_dists[:,1].argsort()]
        sort_k = sorts[:k]

        pull_labels = sort_k[:,0]
        pred = scipy.stats.mode(pull_labels)[0][0]
        pred_arr.append(int(pred))

    return np.array(pred_arr)


# Load in the matlab data file 
data = io.loadmat('data.mat')['face']

# Array for the labels of the Neutral and Emotional Faces
labels = []
for i in range(200):
    labels.append(0)
    labels.append(1)
    labels.append(0)
illum_arr = np.array([i*3-1 for i in range(1,201)])
other_arr = np.array([i for i in range(600)])
neut_emote_indx = other_arr[np.isin(other_arr, illum_arr, invert=True)]

full_data = np.array([data[:,:,i] for i in range(600)])
neut_emote = np.take(full_data, neut_emote_indx, axis=0)

# PCA Preprocessing
pca_coms = 7
cols = ['PCA'+str(i+1) for i in range(pca_coms)]

# Standardize and center data
scale = StandardScaler()
# neut_emote = scale.fit_transform(neut_emote.reshape(400,504))
full_data = scale.fit_transform(full_data.reshape(600,504))
# Determine Covariance Matrix and eigenvalues and vectors 
cov_mats = np.cov(full_data.T)
e_vals, e_vecs = np.linalg.eig(cov_mats)

# Create PCA Arrays based on the pca_coms value 
pca_arrs = []
for com in range(pca_coms):
    pca_arrs.append(full_data.dot(e_vecs.T[com]).astype('float64').reshape(600,1))
pca_arrs = tuple(pca_arrs)
pca_full = np.concatenate(pca_arrs, axis=1)

pca_df = pd.DataFrame(data=pca_full, columns=cols)
pca_df['labels'] = pd.Series(labels)
# pca_df = pca_df.sample(frac=1).reset_index(drop=True)

x_train, x_test, y_train, y_test = train_test_split(pca_df[cols], 
                                    pca_df['labels'], test_size=0.3, 
                                    random_state=5)

preds = knn_fit(x_train, y_train, x_test, 65)
print(accuracy_score(y_test, preds))
# print(knn_fit(x_train, y_train, x_test[0], euclidean))