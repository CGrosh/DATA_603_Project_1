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
    # if len(train_x.shape) == 1:
    train_x = train_x.reshape(1, train_x.shape[0])
    pred_arr = []
    dist_arrs = []
    for test_val in range(len(x_test)):
        test_arr = np.array(x_test.iloc[test_val])
        # if len(test_arr.shape) > 0:
        # test_arr = test_arr.reshape(1, test_arr.shape[0])
        dists_x = np.linalg.norm(train_x- test_arr, axis=0, ord=2)
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
pca_coms = 10
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

total_mu = np.array(np.mean(pca_df[cols]))

# Fishers Linear Discriminant 
pca_emote = pca_df[cols].loc[np.array(pca_df[pca_df['labels']==1].index)]
pca_neutral = pca_df[cols].loc[np.array(pca_df[pca_df['labels']==0].index)]

# Compute Column means and Center data 
mu_emote, mu_neut = np.array(np.mean(pca_emote)), np.array(np.mean(pca_neutral))
pca_emote_stand = scale.fit_transform(pca_emote)
pca_neut_stand = scale.fit_transform(pca_neutral)

# Scatter Matrices
emote_scat = pca_emote_stand.T.dot(pca_emote_stand)
netural_scat = pca_neut_stand.T.dot(pca_neut_stand)

# Within Class Scatter Matrix 
sw = emote_scat + netural_scat

w = np.linalg.inv(sw).dot(mu_neut-mu_emote)

# eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(sw).dot(sb))

# pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
# pairs = sorted(pairs, key=lambda x: x[0], reverse=True)

# eigen_value_sums = sum(eig_vals)

# w_matrix = np.hstack((pairs[0][1].reshape(7,1), pairs[1][1].reshape(7,1))).real
x_lda = np.array(pca_df[cols]).dot(w) 
pca_df['Fishers_LD'] = pd.Series(x_lda)

# Train Test Split for the Classifier using PCA Features
# x_train, x_test, y_train, y_test = train_test_split(pca_df[cols], 
#                                     pca_df['labels'], test_size=0.3, 
#                                     random_state=4)

# Train Test Split using the Fisher LDA Column
x_train, x_test, y_train, y_test = train_test_split(pca_df['Fishers_LD'], 
                                    pca_df['labels'], test_size=0.3, 
                                    random_state=4)

Ns = [i for i in range(1,200,2)]
accs = []
pca_com_lst = [pca_coms for i in range(len(Ns))]
for n in Ns:
    preds = knn_fit(x_train, y_train, x_test, n)
    accs.append(accuracy_score(y_test, preds))

n_df = pd.DataFrame({'Nearest Neighbor Value': Ns, 'Accuracy': accs, 'PCA_N': pca_com_lst})
print(n_df)
ax = n_df.plot(x='Nearest Neighbor Value', y='Accuracy')
ax.set_ylabel('Accuracy')
ax.set_title('Fishers LD on PCA Dim: ' + str(pca_coms))
plt.show()

# n_df.to_csv('../knn_1_mda10.csv')

# preds = knn_fit(x_train, y_train, x_test, 75)
# print(accuracy_score(y_test, preds))
