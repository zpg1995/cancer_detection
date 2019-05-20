
# coding: utf-8

# Libraries
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from torchvision.models import inception_v3

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
## Parameters for model

# Hyper parameters
num_epochs = 8
num_classes = 2
batch_size = 64
learning_rate = 0.002
n_aug = 5 
size = 299
out_features = 2
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


labels = pd.read_csv('../../data/train_labels.csv')
sub = pd.read_csv('../../data/sample_submission.csv')
train_path = '../../data/train/'
test_path = '../../data/test/'

#Splitting data into train and val
# train, val = train_test_split(labels, stratify=labels.label, test_size=0.1)
train =[]
valid = []

val1=labels[:44000]
train1 = labels[44000:]
train.append(train1)
valid.append(val1)

val2=labels[44000:88000]
tem1=labels[:44000]
tem2 = labels[88000:]
train2 = pd.concat([tem1,tem2],axis=0,ignore_index=True)
train.append(train2)
valid.append(val2)

val3 = labels[88000:132000]
tem1 = labels[:88000]
tem2 = labels[132000:]
train3 = pd.concat([tem1,tem2],axis=0,ignore_index=True)
train.append(train3)
valid.append(val3)

val4 = labels[132000:176000]
tem1 = labels[:132000]
tem2 = labels[176000:]
train4 = pd.concat([tem1,tem2],axis=0,ignore_index=True)
train.append(train4)
valid.append(val4)

val5 =labels[176000:]
train5 = labels[:176000]
train.append(train5)
valid.append(val5)

# **Simple custom generator**


class MyDataset(Dataset):
    def __init__(self, df_data, data_dir = './', transform=None):
        super().__init__()
        self.df = df_data.values
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_name,label = self.df[index]
        img_path = os.path.join(self.data_dir, img_name+'.tif')
        image = cv2.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# train_tfms = transforms.Compose([transforms.ToPILImage(),
#                                   transforms.Pad(64, padding_mode='reflect'),
#                                   transforms.RandomHorizontalFlip(), 
#                                   transforms.RandomVerticalFlip(),
#                                   transforms.RandomRotation(20), 
#                                   transforms.ToTensor(),
#                                   transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

# trans_valid = transforms.Compose([transforms.ToPILImage(),
#                                   transforms.Pad(64, padding_mode='reflect'),
#                                   transforms.ToTensor(),
#                                   transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

trans_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((size,size)),
    # transforms.Pad(64, padding_mode='reflect'),
    transforms.RandomChoice([
        transforms.ColorJitter(brightness=0.5),
        transforms.ColorJitter(contrast=0.5), 
        transforms.ColorJitter(saturation=0.5),
        transforms.ColorJitter(hue=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), 
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3), 
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), 
    ]),
    transforms.RandomChoice([
        transforms.RandomRotation((0,0)),
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomVerticalFlip(p=1),
        transforms.RandomRotation((90,90)),
        transforms.RandomRotation((180,180)),
        transforms.RandomRotation((270,270)),
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomRotation((90,90)),
        ]),
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomRotation((270,270)),
        ]) 
    ]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


trans_valid = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((size,size)),
    # transforms.Pad(64, padding_mode='reflect'),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

loader_train = []
loader_valid =[]
for tra in train:
	dataset_train = MyDataset(df_data=tra, data_dir=train_path, transform=trans_train)
	loader_tra = DataLoader(dataset = dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
	loader_train.append(loader_tra)

for val in valid:	
	dataset_valid = MyDataset(df_data=val, data_dir=train_path, transform=trans_valid)
	loader_val = DataLoader(dataset = dataset_valid, batch_size=batch_size//2, shuffle=False, num_workers=0)
	loader_valid.append(loader_val)

# **Model**



model = inception_v3(pretrained =False)
model.load_state_dict(torch.load('./inception_v3.pth'))
model.aux_logits = False
in_features = model.fc.in_features
model.fc = nn.Linear(in_features,out_features)
# model = SimpleCNN().to(device)
# model = SimpleCNN()
# model.load_state_dict(torch.load('./model_trick.ckpt'))
model= model.to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)


scheduler_train = StepLR(optimizer,step_size=1,gamma=0.5)


# Train the model
# total_step = len(loader_train)
for epoch in range(num_epochs):
	for n_split in range(5):
	    for i, (images, labels) in enumerate(loader_train[n_split]):
	        images = images.to(device)
	        labels = labels.to(device)
	        
	        # Forward pass
	        outputs = model(images)

	        loss = criterion(outputs, labels)
	        
	        # Backward and optimize
	        optimizer.zero_grad()
	        loss.backward()
	        optimizer.step()
        
        # if (i+1) % 100 == 0:
        #     print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
        #            .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # scheduler_train.step()
# **Accuracy Check**

# Test the model
				# model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
				# with torch.no_grad():
				#     correct = 0
				#     total = 0
				#     for images, labels in loader_valid:
				#         images = images.to(device)
				#         labels = labels.to(device)
				#         outputs = model(images)
				#         _, predicted = torch.max(outputs.data, 1)
				#         total += labels.size(0)
				#         correct += (predicted == labels).sum().item()
          
    # print('Test Accuracy of the model on the 22003 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'incept.ckpt')


# **CSV submission**

model.eval()

preds = []
for i in range(n_aug):

    dataset_valid = MyDataset(df_data=sub, data_dir=test_path, transform=trans_train)
    loader_test = DataLoader(dataset = dataset_valid, batch_size=32, shuffle=False, num_workers=0)

    for batch_i, (data, target) in enumerate(loader_test):
        data, target = data.cuda(), target.cuda()
        output = model(data)

        pr = output[:,1].detach().cpu().numpy()
        for i in pr:
            preds.append(i)
    # sub.shape, len(preds)
    sub['label'] += preds
    preds = []

sub.label /= n_aug
sub.to_csv('inception_v3.csv', index=False)

