# Import necessary libraries 
import numpy as np 
import pandas as pd 
import scipy.io as io 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load in the matlab data file 
data = io.loadmat('data.mat')['face']

# Array for the labels of the Neutral and Emotional Faces
labels = []
for i in range(400):
    labels.append(0)
    labels.append(1)

# Breaking out 3 different face types 
illum_arr = np.array([i*3-1 for i in range(1,201)])
other_arr = np.array([i for i in range(600)])
neut_emote_indx = other_arr[np.isin(other_arr, illum_arr, invert=True)]
# Use isin to segent the indicies

illum_arrs = np.array([i*3 for i in range(200)])
full_data = np.array([data[:,:,i] for i in range(600)])
neut_emote = np.take(full_data, neut_emote_indx, axis=0)
# print(neut_emote.shape)
# plt.imshow(neut_emote[-1], cmap='gray')
# plt.show()

print(full_data.shape)
print(neut_emote.shape)
print(illum_arrs.shape)

# Split data between Neutral, Expression, and Illumination images
# neut_faces = data[:,:,0::3]
# exp_faces = data[:,:,1::3]
# illum_faces = data[:,:,2::3]
# print(neut_faces.shape)
# Reformat the shape of the data to easier be passed through PCA
# neut_shape = np.array([neut_faces[:,:,i] for i in range(200)])
# exp_face = np.array([exp_faces[:,:,i] for i in range(200)])
# illum_face = np.array([illum_faces[:,:,i] for i in range(200)])

# Concatenate the Neutral and Expression Images 


# Define PCA pipeline
# pca = PCA(n_components=2)

# pca_neut = pca.fit_transform(neut_faces.reshape(200, 504))

# pca_df = pd.DataFrame(data=pca_neut, columns=['PCA1', 'PCA2'])
# print(pca_df)

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