from torch.utils.data import Dataset
import os
import numpy as np
import torch
import cv2 as cv

class VocDataset(Dataset):
  def __init__(self,dir,color_map,mode):
    if mode == "train":
      self.root=os.path.join(dir,'VOCdevkit/VOC2012')
      self.target_dir=os.path.join(self.root,'SegmentationClass')
      self.images_dir=os.path.join(self.root,'JPEGImages')
      file_list=os.path.join(self.root,'ImageSets/Segmentation/train.txt')
      self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]
      self.color_map=color_map
      self.depth_dir = f'{os.path.split(os.path.dirname(os.getcwd()))[0]}/Detph_image_2012/'
    elif mode == "test":
      self.root=os.path.join(dir,'VOCdevkit/VOC2012')
      self.target_dir=os.path.join(self.root,'SegmentationClass')
      self.images_dir=os.path.join(self.root,'JPEGImages')
      file_list=os.path.join(self.root,'ImageSets/Segmentation/val.txt')
      self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]
      self.color_map=color_map
      self.depth_dir = f'{os.path.split(os.path.dirname(os.getcwd()))[0]}/Detph_image_2012/'


  def convert_to_segmentation_mask(self,mask):
  # This function converts color channels of semgentation masks to number of classes (21 in this case)
  # Semantic Segmentation requires a segmentation mask to be a NumPy array with the shape [height, width, num_classes].
  # Each channel in this mask should encode values for a single class. Pixel in a mask channel should have
  # a value of 1.0 if the pixel of the image belongs to this class and 0.0 otherwise.
    height, width = mask.shape[:2]
    segmentation_mask = np.zeros((height, width, len(self.color_map)), dtype=np.float32)
    for label_index, label in enumerate(self.color_map):
          segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
    return segmentation_mask

  def __getitem__(self,index):
    image_id=self.files[index]
    image_path=os.path.join(self.images_dir,f"{image_id}.jpg")
    label_path=os.path.join(self.target_dir,f"{image_id}.png")
    depth_path=os.path.join(self.depth_dir,f"Depth_{image_id}.jpg")

    image=cv.imread(image_path)
    image=cv.cvtColor(image,cv.COLOR_BGR2RGB)
    image=cv.resize(image,(256,256))
    image=torch.tensor(image).float()
    image= image.permute(2,0,1)

    label=cv.imread(label_path)
    label=cv.cvtColor(label,cv.COLOR_BGR2RGB)
    label=cv.resize(label,(256,256))
    label = self.convert_to_segmentation_mask(label)
    label=torch.tensor(label).float()
    label= label.permute(2,0,1)

    depth=cv.imread(depth_path)
    depth=cv.cvtColor(depth,cv.COLOR_BGR2GRAY)
    depth=cv.resize(depth, (256,256))
    depth=torch.tensor(depth).float()
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth=depth.unsqueeze(0)
    # depth=depth.permute(2,0,1)
    
    return image,label, depth


  
  def __len__(self):
    return len(self.files)