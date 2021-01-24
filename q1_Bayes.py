# Import necessary libraries 
import numpy as np 
import pandas as pd 
import scipy.io as io 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load in the matlab data file 
data = io.loadmat('data.mat')['face']

# Array for the labels of the Neutral and Emotional Faces
labels = []
for i in range(200):
    labels.append(0)
    labels.append(1)
    labels.append(0)

# Reformat the images to a nicer (600,48,42) format 
full_data = np.array([data[:,:,i] for i in range(600)])

# PCA Preprocessing
pca_coms = 20
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

# Split the Data into training and testing Features and Labels 
x_train, x_test, y_train, y_test = train_test_split(pca_df[cols], 
                                    pca_df['labels'], test_size=0.3, 
                                    random_state=5)

# Data of the Neutral and the Emotional Projected Instances
pca_emote = x_train.loc[np.array(y_train[y_train==1].index)]
pca_neutral = x_train.loc[np.array(y_train[y_train==0].index)]

# Compute Column Means and Variances
theta_mu_emote, theta_var_emote = np.mean(pca_emote[cols]), np.var(pca_emote[cols])
theta_mu_neut, theta_var_neut = np.mean(pca_neutral[cols]), np.var(pca_neutral[cols])

mu_emote, mu_neut = np.array(theta_mu_emote), np.array(theta_mu_neut)

# Center the Data 
pca_emote_stand = scale.fit_transform(pca_emote[cols])
pca_neut_stand = scale.fit_transform(pca_neutral[cols])

# Compute Covariance Matricies
pca_emote_cov = pca_emote_stand.T.dot(pca_emote_stand)/pca_emote_stand.shape[0]
pca_neutral_cov = pca_neut_stand.T.dot(pca_neut_stand)/pca_neut_stand.shape[0]
# Compute Inverses 
inv_emote_cov = np.linalg.inv(pca_emote_cov)
inv_neutral_cov = np.linalg.inv(pca_neutral_cov)

# Discriminant Functions for the 2 class 
# Neutral Faces
g1 = lambda x: (inv_neutral_cov.dot(mu_neut).T).dot(x) - \
    (0.5 * mu_neut.T.dot(inv_neutral_cov.dot(mu_neut)))

# Emotional Faces 
g2 = lambda x: (inv_emote_cov.dot(np.array(theta_mu_emote)).T).dot(x) - \
    (0.5 * mu_emote.T.dot(inv_emote_cov.dot(mu_emote)))

# Run the testing data through the discriminant Functions and 
# Classify based on the larger output value 
preds = []
for val in range(len(x_test)):
    test_arr = np.array(x_test.iloc[val])

    g1_pred = g1(test_arr)
    g2_pred = g2(test_arr)

    if g1_pred > g2_pred:
        preds.append(0)
    else:
        preds.append(1)

# Check the Accuracy of the Predicted values vs the Labels 
print(accuracy_score(y_test, preds))


