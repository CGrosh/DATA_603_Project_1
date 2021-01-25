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

# Load in the data 
pose_data = io.loadmat('pose.mat')['pose']

# Reformat the data to a cleaner format 
full_data = []
for i in range(pose_data.shape[3]):
    person_imgs = pose_data[:,:,:,i]
    person_list = [person_imgs[:,:,img] for img in range(13)]
    full_data.append(person_list)

full_data = np.array(full_data)
labels = np.array([[img for i in range(13)] for img in range(5)])
labels = labels.reshape(65,)

cleaned_data  = np.array([full_data[i][:] for i in range(len(full_data)-63)])
cleaned_data = cleaned_data.reshape(65, 1920)

# Mark the indices for the testing data 
test_ind = []
for i in range(len(full_data)-63):
    person_lst = []
    for img in range(len(full_data[i])):
        if img <= 9:
            person_lst.append(0)
        else:
            person_lst.append(1)
    test_ind.append(person_lst)
test_ind = np.array(test_ind).reshape(65,)

# train_data = np.array([full_data[i][:10] for i in range(len(full_data))])
# train_labels = np.array([labels[i][:10] for i in range(len(labels))])

# train_data = train_data.reshape(680, 48, 40)
# train_labels = train_labels.reshape(680,)

# test_data = np.array([full_data[i][-3:] for i in range(len(full_data))])
# test_labels = np.array([labels[i][-3:] for i in range(len(labels))])

# test_data = test_data.reshape(204, 48, 40)
# test_labels = test_labels.reshape(204,)

# PCA Preprocessing: PCA Before Train/Test Split
pca_coms = 70
cols = ['PCA'+str(i+1) for i in range(pca_coms)]

# Standardize and center data
scale = StandardScaler()
full_stand_data = scale.fit_transform(cleaned_data)

# Determine Covariance Matrix and eigenvalues and vectors 
cov_mats = np.cov(full_stand_data.T)
e_vals, e_vecs = np.linalg.eig(cov_mats)

# Create PCA Arrays based on the pca_coms value 
pca_arrs = []
for com in range(pca_coms):
    pca_arrs.append(full_stand_data.dot(e_vecs.T[com]).astype('float64').reshape(65,1))
pca_arrs = tuple(pca_arrs)
pca_full = np.concatenate(pca_arrs, axis=1)

pca_df = pd.DataFrame(data=pca_full, columns=cols)
pca_df['labels'] = pd.Series(labels)
pca_df['Test_Ind'] = pd.Series(test_ind)

x_train = pca_df[pca_df['Test_Ind'] == 0]
y_train = x_train['labels']
x_train = x_train[cols]

x_test = pca_df[pca_df['Test_Ind'] == 1]
y_test = x_test['labels']
x_test = x_test[cols]

# Breaking out the data for each class 
class_means = []
classes = []
center_data = []
for i in range(5):
    arr = np.array(x_train.loc[np.array(y_train[y_train==i].index)])
    classes.append(np.array(x_train.loc[np.array(y_train[y_train==i].index)]))
    class_means.append(np.array(np.mean(x_train.loc[np.array(y_train[y_train==i].index)])))
    center_data.append(scale.fit_transform(np.array(x_train.loc[np.array(y_train[y_train==i].index)])))
class_means = np.array(class_means)
classes = np.array(classes)
center_data = np.array(center_data)

# Build an Array containing the Covaraince Matricies for each class 
# Also build and array containing the inverses of each of those classes 
cov_matricies = []
inv_matricies = []
for i in range(5):
    cov_matricies.append(center_data[i].T.dot(center_data[i])/center_data[i].shape[0])
for i in range(len(cov_matricies)):
    inv_matricies.append(np.linalg.inv(cov_matricies[i]))
cov_matricies, inv_matricies = np.array(cov_matricies), np.array(inv_matricies)

# Array containing the Discriminant function for each classes 
# according to the index of the arry 
disc_funcs = []
for i in range(len(cov_matricies)):
    gn = lambda x: (inv_matricies[i].dot(class_means[i]).T).dot(x) - \
        (0.2 * class_means[i].T.dot(inv_matricies[i].dot(class_means[i])))
    disc_funcs.append(gn)

# Run the testing data through each Discriminant function and check for the largest value 
preds = []
for test_val in range(len(x_test)):
    test_arr = np.array(x_test.iloc[test_val])
    func_outs = []
    for func in disc_funcs:
        func_outs.append(func(test_arr))
    preds.append(np.argmax(func_outs))

# Print out the accuracy for the Bayes Model 
print(accuracy_score(y_test, preds))
    
    