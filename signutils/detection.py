# Initialization
import sys
import os

# To be able to reference packages/modules in this repository, this relative path must be added to the python path.
# Your notebook may be in a different folder, so modify this variable to point to the src folder.
src_rel_path = './'
curr_dir = os.getcwd()
proj_src_root = "/home/default/workspace/cs6945share/sign_project/SignDetection/DetectionTraining"
proj_root_path = "/home/default/workspace/cs6945share/sign_project/SignDetection"
data_path = "/home/default/workspace/cs6945share/blyncsy_signs/udot"
if proj_src_root not in sys.path:
    sys.path.insert(0, proj_src_root)
    print("Updated Python Path")


# Notebook imports
import math
import sys
import time
import torch
import json

import numpy as np
import torch
import torchvision
from train import transforms, coco_utils, coco_eval, utils
from train.coco_eval import CocoEvaluator

from PIL import Image

from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms import functional as F
import torchvision.models.detection.mask_rcnn

import geopandas as gpd
import pandas as pd

from glob import glob

#the following are to do with this interactive notebook code


# Define a function that can be used to generate image tags
def create_im_tag(row, folder_path = "/home/default/workspace/cs6945share/archived_workspaces/obata-0/home/default/workspace/cs6945/data/nmdot/bboxes", ws_prefix = None):
    """Function for creating an html image tag

    This function was designed to be used with a Pandas apply method.

    Parameters
    ----------
    row : dict
        a dict containing the field "sourceId" which is the filename of an image.
    folder_path : str
        a string representing the path to the folder (in the workspace) where 
        images are located. Must end in a forward slash
    ws_prefix : str, optional
        a string representing the workspace prefix (usually the owner's last name).
        Must **not** have a trailing foward slash, e.g. /henderson. If this arg
        is not provided then the function will try to determine it from the 
        environment.
        
    Returns
    -------
    str
        a string representing an HTML element that can show an image
    """
    
    if ws_prefix is None:
        ws_prefix = os.environ["WS_PROXY"]
    s = ws_prefix + "/mini-browser" + folder_path +row["sourceId"]
    html = """ \
        <div style='width:300; height:200'> \
            <a href='{0}' target='_blank' rel='noopener noreferrer'> \
                <img width=100% src='{0}'/> \
            </a> \
        </div>
    """.format(s)
    return html

# functions that are helpful during the applying step
def check_box_props(bboxes, params):
    length_threshold = params[0]
    AR_min = params[1]
    checks = []
    for box in bboxes:
        dim1 = box[2] - box[0]
        dim2 = box[3] - box[1]
        if dim1 < length_threshold or dim2 < length_threshold:
            checks.append(False)
            continue
        elif (dim1/dim2 < AR_min) or (dim2/dim1 < AR_min):
            checks.append(False)
            continue
        else:
            checks.append(True)      
    return checks

def write_signs_gdf(data_dict, file_path):
    # Create the output GeoDataFrame and write to file
    print(f"Writing progress to file: {file_path}")
    detect_df = gpd.GeoDataFrame(data_dict, geometry="geometry", crs="EPSG:4326")
    detect_df.to_file(file_path)

def sep_series_on_comma(sourceIds, just_first=True):
    all_ids = []
    for sourceId in sourceIds:
        if just_first:
            ids = [sourceId.split(",")[0]] # just getting the first one in the list since there are indexing issues
        else:
            sourceId.split(",")
        all_ids += ids
    return all_ids