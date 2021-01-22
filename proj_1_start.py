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

# Currently not using the Illumination Arrays, will be using after the first model fit 

# Load in the matlab data file 
data = io.loadmat('data.mat')['face']

# Array for the labels of the Neutral and Emotional Faces
labels = []
for i in range(200):
    labels.append(0)
    labels.append(1)
    labels.append(0)

# Seperate the indicies of the illumination images 
illum_arr = np.array([i*3-1 for i in range(1,201)])
other_arr = np.array([i for i in range(600)])
neut_emote_indx = other_arr[np.isin(other_arr, illum_arr, invert=True)]

# Reformat the images to a nicer (600,48,42) format 
full_data = np.array([data[:,:,i] for i in range(600)])
# Select only the indicies from the data that are the neutral and emotional  
# neut_emote = np.take(full_data, neut_emote_indx, axis=0)


# Split data between Neutral, Expression, and Illumination images
# neut_faces = data[:,:,0::3]
# exp_faces = data[:,:,1::3]
# illum_faces = data[:,:,2::3]

# Reformat the shape of the data to easier be passed through PCA
# neut_shape = np.array([neut_faces[:,:,i] for i in range(200)])
# exp_face = np.array([exp_faces[:,:,i] for i in range(200)])
# illum_face = np.array([illum_faces[:,:,i] for i in range(200)])


# Define PCA pipeline
pca_coms = 4
pca = PCA(n_components=pca_coms)
cols = ['PCA'+str(i+1) for i in range(pca_coms)]

pca_neut = pca.fit_transform(full_data.reshape(600, 504))

pca_df = pd.DataFrame(data=pca_neut, columns=cols)
pca_df['labels'] = pd.Series(labels)
pca_df = pca_df.sample(frac=1).reset_index(drop=True)

x_train, x_test, y_train, y_test = train_test_split(pca_df[cols], 
                                    pca_df['labels'], test_size=0.3, random_state=4)

knn_mod = KNeighborsClassifier(n_neighbors=35)
knn_mod.fit(x_train, y_train)
y_hat = knn_mod.predict(x_test)
print(accuracy_score(y_test, y_hat))

# Ns = [i for i in range(1,60,3)]
# accs = []
# for n in Ns:
#     knn_mod = KNeighborsClassifier(n_neighbors=n)
#     knn_mod.fit(x_train, y_train)
#     y_hat = knn_mod.predict(x_test)
#     accs.append(accuracy_score(y_test, y_hat))

# n_df = pd.DataFrame({'N': Ns, 'Acc': accs})
# print(n_df)
# n_df.plot(x='N', y='Acc')
# plt.show()
# Plot and view the PCAs between the classes 
# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(1,1,1)
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)
# colors = ['r' ,'b']

# neutral_indices = pca_df['labels'] == 0
# emote_indicies = pca_df['labels'] == 1
# illum_indicies = pca_df['labels'] == 2
# ax.scatter(pca_df.loc[neutral_indices, 'PCA1']
#         ,pca_df.loc[neutral_indices, 'PCA2']
#         ,c = 'r'
#         ,s = 50)
# ax.scatter(pca_df.loc[emote_indicies, 'PCA1']
#         ,pca_df.loc[emote_indicies, 'PCA2']
#         ,c = 'b'
#         ,s = 50)
# ax.scatter(pca_df.loc[illum_indicies, 'PCA1']
#         ,pca_df.loc[illum_indicies, 'PCA2']
#         ,c = 'g'
#         ,s = 50)
# ax.legend([0,1,2])
# ax.grid()
# plt.show()


# Check and plot images from each set 
# print(neut_faces[:,:,0].shape)
# plt.imshow(illum_faces[:,:,0], cmap='gray')
# plt.show()


# first_n = neut_faces[:,:,2]
# second_n = 
# third_n = 

# print(first_n.shape)
# p1_p1 = data[:,:,3*1-3]
# p1_p2 = data[:,:,3*1-2]
# p1_p3 = data[:,:,3*1-1]

# p2_p1 = data[:,:,3*2-3]
# p2_p2 = data[:,:,3*2-2]
# p2_p3 = data[:,:,3*2-1]

# print(p1_p1.shape)
# plt.imshow(first_n, cmap='gray')
# plt.show()