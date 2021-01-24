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

def show_pose(person, pose, data):
    plt.imshow(data[:,:,pose, person], cmap='gray')
    plt.show()

def get_pose(person, pose, data):
    return data[:,:,pose, person]

# Load in the data 
pose_data = io.loadmat('pose.mat')['pose']

# Reformat the data to a cleaner format 
full_data = []
for i in range(pose_data.shape[3]):
    person_imgs = pose_data[:,:,:,i]
    person_list = [person_imgs[:,:,img] for img in range(13)]
    full_data.append(person_list)

full_data = np.array(full_data)
labels = [[img for i in range(13)] for img in range(68)]

train_data = np.array([full_data[i][:10] for i in range(len(full_data))])
train_labels = np.array([labels[i][:10] for i in range(len(labels))])

train_data = train_data.reshape(680, 48, 40)
train_labels = train_labels.reshape(680,)

test_data = np.array([full_data[i][-3:] for i in range(len(full_data))])
test_labels = np.array([labels[i][-3:] for i in range(len(labels))])

test_data = test_data.reshape(204, 48, 40)
test_labels = test_labels.reshape(204,)

# PCA Preprocessing
pca_coms = 7
cols = ['PCA'+str(i+1) for i in range(pca_coms)]

# Standardize and center data
scale = StandardScaler()
# neut_emote = scale.fit_transform(neut_emote.reshape(400,504))
full_data = scale.fit_transform(train_data.reshape(680,1920))
# Determine Covariance Matrix and eigenvalues and vectors 
cov_mats = np.cov(full_data.T)
e_vals, e_vecs = np.linalg.eig(cov_mats)

# Create PCA Arrays based on the pca_coms value 
pca_arrs = []
for com in range(pca_coms):
    pca_arrs.append(full_data.dot(e_vecs.T[com]).astype('float64').reshape(680,1))
pca_arrs = tuple(pca_arrs)
pca_full = np.concatenate(pca_arrs, axis=1)

pca_df = pd.DataFrame(data=pca_full, columns=cols)
pca_df['labels'] = pd.Series(train_labels)