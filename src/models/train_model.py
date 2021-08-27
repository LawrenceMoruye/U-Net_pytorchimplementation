import torch 
import os 
import sys 
import pandas as pd 
import numpy as np 
import torch.nn as nn 
import torch.optim as optim 
from torch.optim import lr_scheduler
from data.make_dataset import SIIMDataset 
from collections import OrderedDict 
from sklearn import model_selection
import albumentations as A
from tqdm import tqdm 
from utils import MixedLoss
from model import UNet

transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5),
    A.OneOf([A.RandomGamma(gamma_limit=(90,110)),
             A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1)], p=0.5),
    A.Resize(width=224, height=224)
])
TRAINING_CSV = "../"
TRAINING_BS = 8
TEST_BS = 4
EPOCHS = 5
DEVICE ="cpu"

def train(data_loader, model, criterion, optimizer):
    model.train()
    train_loss = 0
    for data in tqdm(data_loader):
        inputs = data['image']
        labels = data['mask']

        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    return train_loss/len(data_loader)

def evaluate(data_loader, model, criterion):
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for data in tqdm(data_loader):
            inputs = data['image']
            labels = data['mask']

            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            eval_loss += loss.item()

    return eval_loss/len(data_loader)


if __name__=="__main__":

    df = pd.read_csv(TRAINING_CSV)
    train_df,valid_df = model_selection.train_test_split(
        df,
        random_state=0,
        test_size=0.2
    )
    #train and validation img arrays
    train_images = train_df.image_id.values
    valid_images = valid_df.image_ids.values

    #get our model
    model =UNet()
    model.to(DEVICE)
    train_dataset = SIIMDataset(train_images,transform=transform,preprocessing_fun=preprocess_input)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=TRAINING_BS,shuffle=True)

    valid_dataset = SIIMDataset(train_images,transform=transform,preprocessing_fun=preprocess_input)
    valid_loader = torch.utils.data.DataLoader(valid_images,batch_size=TEST_BS,shuffle=True)

    criterion = nn.MixedLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

    for epoch in range(EPOCHS):
        print(f"Train epoch:{epoch}")
        train(train_dataset,train_loader,model,criterion,optimizer)
        print("Validation")
        valid_details=evaluate(valid_dataset,valid_loader,model)
        scheduler.step(valid_details['loss'])
        print("\n")
    




