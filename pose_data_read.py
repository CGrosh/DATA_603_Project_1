import numpy as np 
import scipy.io as io 
import matplotlib.pyplot as plt 
import pandas as pd 

def show_pose(person, pose, data):
    plt.imshow(data[:,:,pose, person], cmap='gray')
    plt.show()

def get_pose(person, pose, data):
    return data[:,:,pose, person]

# Load in the data 
pose_data = io.loadmat('pose.mat')['pose']

# Reformat the data to a cleaner format 
