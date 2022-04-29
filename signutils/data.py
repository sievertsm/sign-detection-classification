import os
from glob import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import seaborn as sns

from PIL import Image
from skimage.filters import gaussian
import cv2

import time
import copy
import ntpath


import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.functional as F


def create_image_dataframe(parent_folder, extension='.jpg'):
    folder_images = os.listdir(parent_folder)
    
    df=[]
    
    for fldr in folder_images:
        temp = pd.DataFrame({'path': glob(os.path.join(parent_folder, fldr, f'*{extension}'))})
        temp['labelname'] = fldr
        df.append(temp)
        
    df = pd.concat(df, axis=0).reset_index(drop=True)
    df['filename'] = df['path'].apply(os.path.basename)
    
    return df


class SyntheticSignData(Dataset):
    """Synthetic Sign dataset."""

    def __init__(self, df, transform=None):
        """
        Args:
            df: dataframe containing columns for image path ("path") and image label ("label")
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.df = df.reset_index()
        self.paths = df['path'].values
        self.labels = df['label'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        image = Image.open(self.paths[idx])
        label = self.labels[idx]

        sample = (image, label)

        if self.transform:
            sample = self.transform(sample)

        return sample


class MakeSquare(object):
    """Convert images to be square."""

    def __init__(self):
        pass
        
    def __call__(self, sample):
        
        image, label = sample
        W, H = image.size
        # returns image if already square
        if W==H:
            return sample 

        # converts image to square
        else:
            maxDim = max(H, W)
            dH = (H - maxDim)//2
            dW = (W - maxDim)//2
            image = image.crop((dW, dH, dW+maxDim, dH+maxDim))

            return (image, label)
        
        
class RotateImage(object):
    """Randomly rotate images."""
    
    def __init__(self, p=0.5, degrees=30):
        self.p = p
        self.degrees = degrees

    def __call__(self, sample):

        image, label = sample
        prob = np.random.random()
        if prob < self.p:
            r = np.random.randint(low=-self.degrees, high=self.degrees+1)
            image = image.rotate(r)

        return (image, label)
        
    
class ResizeImage(object):
    """Resize image to specified dimension."""
    
    def __init__(self, size):
        self.size = size
        
    def __call__(self, sample):

        image, label = sample
        image = image.resize(self.size)

        return (image, label)
    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        image, label = sample
        
        image = np.array(image)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        image = torch.tensor(image)
        
        label = torch.tensor(label)
        
        sample = (image, label)
        
        return sample
    
    
class NormalizeImage(object):
    """Normalize images based on given means and standard deviations."""
    
    def __init__(self, mean, std, image_max=255):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        self.image_max = image_max
        
    def __call__(self, sample):  

        image, label = sample

        # ensure image values are between 0 and 1
        image = image / self.image_max
        
        # perform normalization
        image = (image - self.mean) / self.std
        sample = (image, label)
        
        return sample


class UnNormalizeImage(object):
    """Reverse the normalization based on mean and standard deviation."""
    
    def __init__(self, mean, std, image_max=255):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        self.image_max = image_max
        
    def __call__(self, image):

        image = image * self.std + self.mean
        image = np.uint8(image*self.image_max)
        
        return image


# was in the dataConverter.py file
def get_image_information(image_directory):
    '''
    Gets information about images in a directory
    
    INPUT:
        image_directory -- directory where the images are contained
        
    OUTPUT:
        df_image -- dataframe with the image name, height and width
    '''
    
    img_files = glob(f"{image_directory}/*")
    
    image_name, image_width, image_height = [], [], []
    for fl in img_files:

        im = Image.open(fl)
        im_w, im_h = im.size

        # image_name.append(fl.split('\\')[-1])
        image_name.append(ntpath.basename(fl))
        image_width.append(im_w)
        image_height.append(im_h)

    df_image = pd.DataFrame({'filename': image_name, 'image_width': image_width, 'image_height': image_height})
    df_image['image_id'] = df_image.index+1
    return df_image


def convert_df_to_json(df, image_directory, category_encoder=None):
    '''
    Takes the dataframe containing labeling information and converts it to json for COCO
    
    INPUT:
        df               -- dataframe containing labeling information
        image_directory  -- directory containing the images
        category_encoder -- dictionary that has the given category as the key and the desired label as the value
        
    OUTPUT:
        json_info -- dictionary containing the json information for COCO
    '''
    
    df = df[df['class']!='skip']
    
    # create a category encoder if none is given
    if not category_encoder:
        # get all categories in the data
        categories = sorted(df['class'].unique())
        # create an encoder going from class label to integer
        categories.remove('none')
        category_encoder = {c:int(i+1) for i, c in enumerate(categories)}
        
    # encode the class into class_labels integers
    df['class_label'] = df['class'].map(category_encoder)
    
    # get image information
    df_image = get_image_information(image_directory)
    
    # make a dataframe giving all of the images a unique image ID
    temp = pd.DataFrame(df['filename'].unique(), columns=['filename'])
    temp = pd.merge(left=temp, right=df_image, on='filename')
    
    # add the image ID into the dataframe
    df_ann = pd.merge(left=df, right=temp, on='filename', how='outer')
    # give each annotation a unique id
    df_ann = df_ann.reset_index(drop=True)
    df_ann['ann_id'] = df_ann.index
    df_ann['class_label'] = df_ann['class_label'].fillna(0)
    
    # get annotation info
    annotations_info=[]
    for i, row in df_ann.iterrows():
        if row['class_label'] != 0:
            a = {
                'id': row['ann_id'], 
                'image_id': row['image_id'], 
                'category_id': int(row['class_label']), 
                'bbox': [row['x0'], row['y0'], row['w'], row['h']], 
                'score': 0.0, 
                'area': (row['area']), 
                'dimensions': [0.0, 0.0, 0.0], 
                'rotation_y': 0.0, 
                'occluded': 0, 
                'truncated': 0.0, 
                'location': [0.0, 0.0, 0.0], 
                'iscrowd': 0.0 
            }
            annotations_info.append(a)

    # reduce dataframe to only image information
    df_img = df_ann[['image_id', 'image_width', 'image_height', 'filename']].drop_duplicates()
    # get image information
    images_info=[]
    for i, row in df_img.iterrows():
        f={
            "id": row['image_id'], 
            "width": row['image_width'], 
            "height": row['image_height'], 
            "file_name": row['filename']
        }
        images_info.append(f)

    # get categories information
    category_info = [{"id": category_encoder[k],"name": f"{k}", "supercategory": None} for k in category_encoder.keys()]
    
    # combine into one dictionary
    json_info = {'categories': category_info,'images': images_info, 'annotations': annotations_info}
    
    return json_info