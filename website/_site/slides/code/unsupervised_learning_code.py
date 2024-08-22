import mglearn
import json
import numpy as np
import pandas as pd
import os, sys
from collections import OrderedDict
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms, models, datasets
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torchvision import datasets, models, transforms, utils
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg16

torch.manual_seed(42)

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

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


