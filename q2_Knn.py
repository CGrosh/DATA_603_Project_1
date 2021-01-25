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

# Similar function for computing the KNN
def knn_fit(x_train, y_train, x_test, k):

    # Format the training data into arrays 
    train_x = np.array(x_train)
    train_y = np.array(y_train)

    # Loop through the Testing data and preform calculations and predictions
    pred_arr = []
    dist_arrs = []
    for test_val in range(len(x_test)):
        # Format the testing data into a cleaner array 
        test_arr = np.array(x_test.iloc[test_val])
        # Reshape to be easier compared to the training data
        test_arr = test_arr.reshape(1, test_arr.shape[0])
        # Compute the p-norm distances from the whole training data to the test instance 
        dists_x = np.linalg.norm(train_x- test_arr, axis=1, ord=2)
        # Stack the training labels with the distances of each training sample 
        # from the the testing instance 
        joined_dists = np.column_stack((train_y, dists_x))

        # Sort the distances and labels based on the distance 
        sorts = joined_dists[joined_dists[:,1].argsort()]
        # Take the top k samples 
        sort_k = sorts[:k]

        # Grab the labels from the top k samples 
        pull_labels = sort_k[:,0]
        # Take the mode from the top k sample labels 
        pred = scipy.stats.mode(pull_labels)[0][0]
        # Append the mode (prediction) to the list above 
        pred_arr.append(int(pred))

    return np.array(pred_arr)

# Load in the data 
pose_data = io.loadmat('pose.mat')['pose']

# Reformat the data to a cleaner format 
full_data = []
for i in range(pose_data.shape[3]):
    person_imgs = pose_data[:,:,:,i]
    person_list = [person_imgs[:,:,img] for img in range(13)]
    full_data.append(person_list)

full_data = np.array(full_data)
labels = np.array([[img for i in range(13)] for img in range(68)])
labels = labels.reshape(884,)

cleaned_data  = np.array([full_data[i][:] for i in range(len(full_data))])
cleaned_data = cleaned_data.reshape(884, 1920)

# Identify the last 3 pose images of each subject as the test data 
test_ind = []
for i in range(len(full_data)):
    person_lst = []
    for img in range(len(full_data[i])):
        if img <= 9:
            person_lst.append(0)
        else:
            person_lst.append(1)
    test_ind.append(person_lst)
test_ind = np.array(test_ind).reshape(884,)

# This commented code was used when testing the train/test split before the PCA Preprocessing 
# train_data = np.array([full_data[i][:10] for i in range(len(full_data))])
# train_labels = np.array([labels[i][:10] for i in range(len(labels))])

# train_data = train_data.reshape(680, 48, 40)
# train_labels = train_labels.reshape(680,)

# test_data = np.array([full_data[i][-3:] for i in range(len(full_data))])
# test_labels = np.array([labels[i][-3:] for i in range(len(labels))])

# test_data = test_data.reshape(204, 48, 40)
# test_labels = test_labels.reshape(204,)

# PCA Preprocessing: PCA Before Train/Test Split
pca_coms = 700
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
    pca_arrs.append(full_stand_data.dot(e_vecs.T[com]).astype('float64').reshape(884,1))
pca_arrs = tuple(pca_arrs)
pca_full = np.concatenate(pca_arrs, axis=1)

# Attach the PCA Arrays to an organized Dataframe 
pca_df = pd.DataFrame(data=pca_full, columns=cols)
pca_df['labels'] = pd.Series(labels)
pca_df['Test_Ind'] = pd.Series(test_ind)

# Split out the train and test sets 
x_train = pca_df[pca_df['Test_Ind'] == 0]
y_train = x_train['labels']
x_train = x_train[cols]

x_test = pca_df[pca_df['Test_Ind'] == 1]
y_test = x_test['labels']
x_test = x_test[cols]


# Loop for testing out the different parameters of the KNN
Ns = [i for i in range(1,200,2)]
accs = []
for n in Ns:
    preds = knn_fit(x_train, y_train, x_test, n)
    accs.append(accuracy_score(y_test, preds))

n_df = pd.DataFrame({'N': Ns, 'Accuracy': accs})
print(n_df)
ax = n_df.plot(x='N', y='Accuracy')
ax.set_ylabel('Accuracy')
ax.set_title('PCA Dim: ' + str(pca_coms))
plt.show()

# # PCA Preprocessing: PCA After Train/Test Split
# pca_coms = 150
# cols = ['PCA'+str(i+1) for i in range(pca_coms)]

# # Standardize and center data
# scale = StandardScaler()
# train_stand_data = scale.fit_transform(train_data.reshape(680,1920))

# # Determine Covariance Matrix and eigenvalues and vectors 
# cov_mats = np.cov(train_stand_data.T)
# e_vals, e_vecs = np.linalg.eig(cov_mats)

# # Create PCA Arrays based on the pca_coms value 
# pca_arrs = []
# for com in range(pca_coms):
#     pca_arrs.append(train_stand_data.dot(e_vecs.T[com]).astype('float64').reshape(680,1))
# pca_arrs = tuple(pca_arrs)
# pca_full = np.concatenate(pca_arrs, axis=1)

# pca_df_train = pd.DataFrame(data=pca_full, columns=cols)
# pca_df_train['labels'] = pd.Series(train_labels)

# # Run the testing data through the PCA Pipeline 
# test_stand_data = scale.transform(test_data.reshape(204,1920))

# # Determine Covariance Matrix and eigenvalues and vectors 
# cov_mats_test = np.cov(test_stand_data.T)
# e_vals_test, e_vecs_test = np.linalg.eig(cov_mats_test)

# # Create PCA Arrays based on the pca_coms value 
# pca_arrs_test = []
# for com in range(pca_coms):
#     pca_arrs_test.append(test_stand_data.dot(e_vecs_test.T[com]).astype('float64').reshape(204,1))
# pca_arrs_test = tuple(pca_arrs_test)
# pca_full_test = np.concatenate(pca_arrs_test, axis=1)

# pca_df_test = pd.DataFrame(data=pca_full_test, columns=cols)
# pca_df_test['labels'] = pd.Series(test_labels)

# preds = knn_fit(pca_df_train[cols], pca_df_train['labels'], pca_df_test[cols], 50)
# print(accuracy_score(pca_df_test['labels'], preds))