import torch
import numpy as np
import pandas as pd 
import os
import glob
from PIL import Image,ImageFile 
from collections import defaultdict 
from tqdm import tqdm 
from torchvision import transforms 
from albumentations import (
    Compose,
    OneOf,
    RandomBrightnessContrast,
    RandomGamma,
    ShiftScaleRotate,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self,image_ids,transform=True,preprocessing_img_fxn=None):
        self.data = defaultdict(dict)
        self.transform = transform
        self.preprocessing_img_fxn =preprocessing_img_fxn
        self.aug = Compose(
            [
                ShiftScaleRotate
                (
                shift_limit =0.0625,
                scale_limit =0.1,
                rotate_limit =10,
                p=0.8
                ),
            OneOf(
                [
                    RandomGamma(
                        gamma_limit = (90,110)
                        ),
                        RandomBrightnessContrast(
                            brightness_limit =0.1,
                            contrast_limit =0.1
                        ),
                   
                ],
                p=0.5,
            ),
            ]
        )
        for img_id in image_ids:
            files = glob.glob(os.Path.join(TRAIN_PATH,img_id,"*.png"))
            self.data[counter] = {
                "img_path":os.path.join(
                    TRAIN_PATH,img_id +".png"
                ),
                "mask_path":os.path.join(
                    TRAIN_PATH,img_id + "_mask.png"
                ),
            }
    def __len__(self):
        return len(self.data)

    def __getitem__(self,item):
        img_path = self.data[item]["img_path"]
        mask_path = self.data[item]["mask_path"]
        
        #read_image and convert it to RGB
        img = Image.open(img_path)
        img = img.convert("RGB")
        #img to numpy array
        img = np.array(img)

        #read mask
        mask = Image.open(mask_path)

        #convrting to binary float matrix
        mask = (mask >=1).astype("float32")
        #apply transform to train data only

        if self.transform is True:
            augmented =self.aug(image =img,mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = self.preprocessing_img_fxn(img)
        return {
            "image":transforms.ToTensor()(img),
            "mask" :transforms.ToTensor()(mask).float()
        }
