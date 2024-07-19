# https://pytorch.org/vision/main/generated/torchvision.datasets.VOCSegmentation.html
from torchvision.datasets import VOCSegmentation
import torchvision.transforms as transforms

# Train Dataset
train_data = VOCSegmentation(root='./PASCAL_VOC_2012', year='2012', image_set='train', download=True, transform = transforms.Compose([transforms.ToTensor()]))

# Validation Dataset
val_data = VOCSegmentation(root='./PASCAL_VOC_2012', year='2012', image_set='val', download=True, transform = transforms.Compose([transforms.ToTensor()]))