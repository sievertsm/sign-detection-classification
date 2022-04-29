import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from glob import glob
from scipy.ndimage import median_filter as median

import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms
from PIL import Image
import torchvision

src_rel_path = './'
proj_src_root = os.path.abspath(src_rel_path)
proj_root_path = os.path.abspath(src_rel_path + "../")
data_path = "/cs6945share/blyncsy_signs/udot"

def handle_bboxstr(string):
    box_strs = string[1:-1].split("\n ")
    box_list = [box_str[1:-1].split() for box_str in box_strs]
    final_list = []
    for item in box_list:
        box_int = [int(float(num)) for num in item]
        final_list += [box_int]
    return final_list

def handle_labandscore(string):
    string = string.translate(str.maketrans('', '', ' \n\t\r')) # take out all forms of white space
    label_str = string[8:-18].split(",") # indicies found manually for torch tensor junk
    return [float(lab) for lab in label_str]

def df_str2num(df):
    boxes = list(df.boxes)
    labels = list(df.labels)
    scores = list(df.scores)
    
    for i in range(len(df)):
        boxes[i] = handle_bboxstr(boxes[i])
        labels[i] = handle_labandscore(labels[i])
        scores[i] = handle_labandscore(scores[i])
        
    df.boxes = boxes
    df.labels = labels
    df.scores = scores
    
    return df

def unique_img(name):
    base_name = name[:-4] + "00" + '.jpg'
    if os.path.exists(base_name):
        files = glob(base_name[:-6] + '*')
        nums = [int(base_name[-6:-4]) for names in files]
        next_num = max(nums) + 1
        new_name = files[0][:-6] + str(next_num).zfill(2) + '.jpg'
    else:
        new_name = name[:-4] + '00' + '.jpg'
    return new_name

def extract_box(img_path, boxes, data_path, savedir, save=True):
    img = plt.imread(os.path.join(data_path, img_path))
    for box in boxes:
        crop = img[box[1]:box[3], box[0]:box[2], :]
        name = os.path.join(savedir, img_path.split("/")[1])
        new_name = unique_img(name)
        if save:
            plt.imsave(new_name, crop)

def extract_box_indetect(img, boxes):
    crops = []
    for box in boxes:
        box = [int(elem) for elem in box]
        # crop = img[box[1]:box[3], box[0]:box[2], :]
        crop = img.crop((box[0], box[1], box[2], box[3]))
        crops.append(crop)       
    return crops




def vec_translate(a, my_dict):
    '''https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key'''
    return np.vectorize(my_dict.__getitem__)(a)


def update_geojson(df, i, out_path, detections):
    im_data = in_map_df.iloc[i,:].to_dict()
    im_data["img"] = create_im_tag(im_data, folder_path=out_path)
    detections.append(im_data)
    return detections

def update_geojson2(df, i, out_path, detections, col='classImages'):
    im_data = in_map_df.iloc[i,:].to_dict()
    im_data[col] = create_im_tag(im_data, folder_path=out_path)
    detections.append(im_data)
    return detections

# Define a function that can be used to generate image tags
def create_im_tag(row, folder_path="/home/default/workspace/cs6945share/sign_project/product/scratch", ws_prefix = None):
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
    s = ws_prefix + "/mini-browser" + folder_path + row["sourceId"]
    html = """ \
        <div style='width:300; height:200'> \
            <a href='{0}' target='_blank' rel='noopener noreferrer'> \
                <img width=100% src='{0}'/> \
            </a> \
        </div>
    """.format(s)
    return html

def write_signs_gdf(data_dict, file_path):
    # Create the output GeoDataFrame and write to file
    print(f"Writing progress to file: {file_path}")
    detect_df = gpd.GeoDataFrame(data_dict, geometry="geometry", crs="EPSG:4326")
    detect_df.to_file(file_path)


def show(img):
    img = img.detach()
    img = F.to_pil_image(img)
    fig, ax = plt.subplots(figsize=(15,15))
    plt.imshow(img)
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])    
    plt.show()   


def img_with_bboxes(r, saveim=False, showim=True, out_path=None):  

    if len(r['classification']['labels'])==0:
        if showim:
            img_name = r['detection']["sourceId"]
            img = plt.imread(img_name)
            fig, ax = plt.subplots(figsize=(18,18))
            plt.imshow(img)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            plt.show()
        return

    view_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    img_name = r["detection"]["sourceId"]
    img = Image.open(img_name)
    img = torchvision.transforms.ToTensor()(img)*255
    img = img.type(torch.uint8)
    boxes = r["detection"]["boxes"]
    boxes = torchvision.transforms.ToTensor()(boxes)[0]
    labels = r["classification"]["labels"]
    imboxes = draw_bounding_boxes(img, boxes, width=5, colors="magenta", labels=labels)
    if showim:
        show(imboxes)
    if saveim:
        imboxes = F.to_pil_image(imboxes)
        img_name = r["detection"]["sourceId"].split("/")[-1]
        out_path = os.path.join(out_path, img_name)
        imboxes.save(out_path)


def display_classification(r, showim=True, saveclass=False, out_path=None):
    
    img_with_bboxes(r, showim=showim)

    if len(r['classification']['labels'])==0: # condition for if there are no signs (length is just the sign detection dict in list)
        print("No signs detected.")
        return

    source_signs = glob(os.path.join('../Synthetic', 'sign_source', '*'))
    source_signs = {fl.split('_')[-1][:-4].lower():fl for fl in source_signs}

    images = r['images']

    n_images = len(images)

    font_size=18
    fig, ax = plt.subplots(nrows=n_images, ncols=len(r['classification']['labels_5'][0])+1, figsize=(18, 3*n_images))

    for k, im in enumerate(images):

        conf = float(r["detection"]["scores"][k])*100

        t_labs = r['classification']['labels_5'][k]
        t_score = r['classification']['scores_5'][k]

        # plt.suptitle('Classification Results', y=1.05, fontsize=20)

        if n_images == 1:
            ax[0].imshow(np.array(im))
            ax[0].set_title(f"{conf:.2f}% sign", fontsize=font_size)

            for i, lab in enumerate(t_labs):
                ax[i+1].imshow(plt.imread(source_signs[t_labs[i]]))
                ax[i+1].set_title(f"{t_labs[i]} | {100*t_score[i]:.2f}%", fontsize=font_size)

            for a in ax:
                a.axis('off')

        else:
            ax[k, 0].imshow(np.array(im))
            ax[k, 0].set_title(f"{conf:.2f}% sign", fontsize=font_size)
            ax[k, 0].axis('off')

            for i, lab in enumerate(t_labs):
                ax[k, i+1].imshow(plt.imread(source_signs[t_labs[i]]))
                ax[k, i+1].set_title(f"{t_labs[i]} | {100*t_score[i]:.2f}%", fontsize=font_size)
                ax[k, i+1].axis('off')
        
        fig.tight_layout()

        if saveclass:
            plt.savefig(out_path, bbox_inches='tight')


def sep_series_on_comma(sourceIds, just_first=True):
    all_ids = []
    for sourceId in sourceIds:
        if just_first:
            ids = [os.path.join(data_path, sourceId.split(",")[0])] # just getting the first one in the list since there are indexing issues
        else:
            sourceId.split(",")
        all_ids += ids
    return all_ids