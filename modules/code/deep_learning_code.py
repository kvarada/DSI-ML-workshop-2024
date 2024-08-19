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
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg16
torch.manual_seed(42)

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def classify_image(img, topn = 4):
    clf = vgg16(weights='VGG16_Weights.DEFAULT') # initialize the classifier with VGG16 weights
    preprocess = transforms.Compose([
                 transforms.Resize(299),
                 transforms.CenterCrop(299),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225]),])

    with open('data/imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    clf.eval()
    output = clf(batch_t)
    _, indices = torch.sort(output, descending=True)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    d = {'Class': [classes[idx] for idx in indices[0][:topn]], 
         'Probability score': [np.round(probabilities[0, idx].item(),3) for idx in indices[0][:topn]]}
    df = pd.DataFrame(d, columns = ['Class','Probability score'])
    return df


# Attribution: [Code from PyTorch docs](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html?highlight=transfer%20learning)


IMAGE_SIZE = 200
BATCH_SIZE = 64

def read_data(data_dir): 
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),     
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),            
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),                        
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),                        
            ]
        ),
    }

    import os 
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "valid"]
    }
    
    dataloaders = {}
    
    dataloaders["train"] = torch.utils.data.DataLoader(
            image_datasets["train"], batch_size=BATCH_SIZE, shuffle=True
        )
    
    dataloaders["valid"] = torch.utils.data.DataLoader(
            image_datasets["valid"], batch_size=BATCH_SIZE, shuffle=False
        )
    
    return image_datasets, dataloaders
