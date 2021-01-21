# Import necessary libraries 
import numpy as np 
import pandas as pd 
import scipy.io as io 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Currently not using the Illumination Arrays, will be using after the first model fit 

# Load in the matlab data file 
data = io.loadmat('data.mat')['face']

# Array for the labels of the Neutral and Emotional Faces
labels = []
for i in range(400):
    labels.append(0)
    labels.append(1)

# Seperate the indicies of the illumination images 
illum_arr = np.array([i*3-1 for i in range(1,201)])
other_arr = np.array([i for i in range(600)])
neut_emote_indx = other_arr[np.isin(other_arr, illum_arr, invert=True)]

full_data = np.array([data[:,:,i] for i in range(600)])
neut_emote = np.take(full_data, neut_emote_indx, axis=0)


# Split data between Neutral, Expression, and Illumination images
# neut_faces = data[:,:,0::3]
# exp_faces = data[:,:,1::3]
# illum_faces = data[:,:,2::3]
# print(neut_faces.shape)
# Reformat the shape of the data to easier be passed through PCA
# neut_shape = np.array([neut_faces[:,:,i] for i in range(200)])
# exp_face = np.array([exp_faces[:,:,i] for i in range(200)])
# illum_face = np.array([illum_faces[:,:,i] for i in range(200)])

# Define PCA pipeline
pca = PCA(n_components=2)

pca_neut = pca.fit_transform(neut_emote.reshape(400, 504))

pca_df = pd.DataFrame(data=pca_neut, columns=['PCA1', 'PCA2'])
pca_fd = pca_df.sample(frac=1).reset_index(drop=True)
pca_df['labels'] = pd.Series(labels)
print(pca_df.head())

# plt.imshow(illum_face[0], cmap='gray')
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