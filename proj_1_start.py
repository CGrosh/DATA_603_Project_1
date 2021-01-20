# Import necessary libraries 
import numpy as np 
import scipy.io as io 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Load in the matlab data file 
data = io.loadmat('data.mat')['face']

# Split data between Neutral, Expression, and Illumination images
neut_faces = data[:,:,0::3]
exp_faces = data[:,:,1::3]
illum_faces = data[:,:,2::3]

# Reformat the shape of the data to easier be passed through PCA
neut_shape = np.array([neut_faces[:,:,i] for i in range(200)])
exp_face = np.array([exp_faces[:,:,i] for i in range(200)])
illum_face = np.array([illum_faces[:,:,i] for i in range(200)])

# Define PCA pipeline
pca = PCA(n_components=2)

pca_neut = pca.fit_transform(neut_faces.reshape(200, 504))
print(pca_neut.shape)

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