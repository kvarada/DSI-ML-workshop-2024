
import mglearn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, colorConverter, LinearSegmentedColormap
from scipy.spatial import distance
from sklearn.metrics import euclidean_distances
from sklearn.manifold import MDS
from scipy.spatial import distance
from sklearn.datasets import make_blobs
from matplotlib.patches import Circle
from sklearn.linear_model import Lasso
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import scipy
from scipy.cluster.hierarchy import (
    average,
    complete,
    dendrogram,
    fcluster,
    single,
    ward,
)
from scipy.spatial.distance import cdist

import torch
import torchvision
from torchvision import datasets, models, transforms, utils
from PIL import Image
import matplotlib.pyplot as plt
import random

torch.manual_seed(42)

import matplotlib.pyplot as plt

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed(seed=42)    

import glob
IMAGE_SIZE = 224
def read_img_dataset(data_dir, BATCH_SIZE):     
    data_transforms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),     
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),            
        ])
               
    image_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    dataloader = torch.utils.data.DataLoader(
         image_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    dataset_size = len(image_dataset)
    class_names = image_dataset.classes
    inputs, classes = next(iter(dataloader))
    return inputs, classes

def plot_sample_imgs(inputs):
    plt.figure(figsize=(10, 70)); plt.axis("off"); plt.title("Sample Training Images")
    plt.imshow(np.transpose(utils.make_grid(inputs, padding=1, normalize=True),(1, 2, 0)));


def get_flattened_representations(data_dir, BATCH_SIZE):

    flatten_transforms = transforms.Compose([    
                        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),    
                        transforms.Lambda(torch.flatten)])
    flatten_images = datasets.ImageFolder(root=data_dir, transform=flatten_transforms)
    flatten_dataloader = torch.utils.data.DataLoader(
        flatten_images, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    flatten_train, y_train = next(iter(flatten_dataloader))
    flatten_images = flatten_train.numpy()
    return flatten_images

def get_cluster_images(model, Z, inputs, cluster=0, n_img=5):
    fig, axes = plt.subplots(1, n_img + 1, subplot_kw={'xticks': (), 'yticks': ()},
                             figsize=(10, 10), gridspec_kw={"hspace": .3})
    img_shape = [3,224,224]
    transpose_axes = (1,2,0)      
    
    if type(model).__name__ == 'KMeans': 
        center = model.cluster_centers_[cluster]
        dists = np.linalg.norm(Z - center, axis=1)
        # mask = model.labels_ == cluster
        # dists = np.sum((Z - center) ** 2, axis=1)
        #dists[~mask] = np.inf        
        closest_index = np.argmin(dists)
        inds = np.argsort(dists)[:n_img]
        print(closest_index)
        if Z.shape[1] == 1024: 
            axes[0].imshow(np.transpose(inputs[closest_index].reshape(img_shape) / 2 + 0.5, transpose_axes))
            #axes[0].imshow(center.reshape((32,32)))
        else:
            axes[0].imshow(np.transpose(center.reshape(img_shape) / 2 + 0.5, transpose_axes))
        axes[0].set_title('Cluster center %d'%(cluster))       
    if type(model).__name__ == 'GaussianMixture':
        center = model.means_[cluster]        
        cluster_probs = model.predict_proba(Z)[:,cluster]        
        inds = np.argsort(cluster_probs)[-n_img:]
        dists = np.linalg.norm(Z - center, axis=1)
        # Find the index of the closest feature vector to the mean
        closest_index = np.argmin(dists)
        if Z.shape[1] == 1024: 
            axes[0].imshow(np.transpose(inputs[closest_index].reshape(img_shape) / 2 + 0.5, transpose_axes))
        else:
            axes[0].imshow(np.transpose(center.reshape(img_shape) / 2 + 0.5, transpose_axes))
        #axes[0].imshow(np.transpose(inputs[inds[0]].reshape(img_shape) / 2 + 0.5, transpose_axes))
        axes[0].set_title('Cluster %d'%(cluster))   
        
    i = 1
    print('Image indices: ', inds)
    for image in inputs[inds]:
        axes[i].imshow(np.transpose(image/2 + 0.5 , transpose_axes))
        i+=1
    plt.show()


def get_features(model, inputs):
    """Extract output of densenet model"""
    model.eval()
    with torch.no_grad():  # turn off computational graph stuff        
        Z = model(inputs).detach().numpy()         
    return Z
    