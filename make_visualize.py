from VocClasses import VOC_COLORMAP
import numpy as np
import cv2 as cv
import torch
import matplotlib 
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn

def render_depth(values, colormap_name="magma_r") -> Image:
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
    colors = colors[:, :, :3] # Discard alpha component
    return Image.fromarray(colors)

def onehot_to_single_channel(segmentation_mask, color_map):
    """Converts a multi-channel one-hot segmentation mask to a 1-channel mask.

    Args:
        segmentation_mask: A NumPy array of shape (height, width, num_classes)
                           representing the one-hot encoded mask.
        color_map: A list of colors, where each color corresponds to a class.

    Returns:
        A NumPy array of shape (height, width) representing the 1-channel mask.
    """
    c, width, height = segmentation_mask.shape  # Get height and width of original image

    single_channel_mask = np.zeros(( width, height), dtype=np.uint8)  # Use uint8 for consistent data type

    for label_index, label in enumerate(color_map):
        class_mask = segmentation_mask[label_index, :, :] == 1
        single_channel_mask[class_mask] = label_index  # Assign class indices directly

    return single_channel_mask

def convert_to_segmentation_mask(mask):
    height, width = mask.shape[:2]
    segmentation_mask = np.zeros((height, width, len(VOC_COLORMAP)), dtype=np.float32)
    for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
    return segmentation_mask

def make_visual(model, dirname, num="2009_005148"):
    image=cv.imread(f"{dirname}/PASCAL_VOC_2012/VOCdevkit/VOC2012/JPEGImages/{num}.jpg")
    image=cv.cvtColor(image,cv.COLOR_BGR2RGB)
    image=cv.resize(image,(256,256))
    image=torch.tensor(image).float()
    image= image.permute(2,0,1)

    label=cv.imread(f"{dirname}/PASCAL_VOC_2012/VOCdevkit/VOC2012/SegmentationClass/{num}.png")
    label=cv.cvtColor(label,cv.COLOR_BGR2RGB)
    label=cv.resize(label,(256,256))
    label = convert_to_segmentation_mask(label)
    label=torch.tensor(label).float()
    label= label.permute(2,0,1)

    depth=cv.imread(f"{dirname}/Detph_image_2012/Depth_{num}.jpg")
    depth=cv.cvtColor(depth,cv.COLOR_BGR2GRAY)
    depth=cv.resize(depth, (256,256))
    depth=torch.tensor(depth).float()
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth=depth.unsqueeze(0)

    colorss = np.array(VOC_COLORMAP)
    plt.figure(figsize=(26, 6))
    plt.subplot(1,5,1)
    plt.title("Image")
    plt.imshow(image.permute(1, 2, 0).detach().cpu().numpy()/255)
    plt.axis('off')

    plt.subplot(1,5,2)
    plt.title("GT")
    label = onehot_to_single_channel(label.detach().cpu().numpy(), VOC_COLORMAP)
    colored_image = colorss[label]
    plt.imshow(colored_image) 
    plt.axis('off')

    plt.subplot(1,5,3)
    plt.title("Semantic Predicted")
    model.eval()
    device = torch.device('cuda:0')
    s_out, d_out = model(image.unsqueeze(0).to(device))
    _, targets_u = torch.max(s_out.squeeze(), dim=0, keepdim = True)
    targets_u = targets_u.squeeze().cpu().detach().numpy()
    colored_image = colorss[targets_u]
    plt.imshow(colored_image)
    plt.axis('off')

    plt.subplot(1,5,4)
    plt.title("Depth Predicted")
    depth_image = d_out.squeeze().detach().cpu().numpy() 
    depth_image2 =  render_depth(depth_image)
    plt.imshow(depth_image2)
    plt.axis('off')
    
    plt.subplot(1,5,5)
    depth_image3 = render_depth(depth.squeeze().detach().cpu().numpy())
    plt.imshow(depth_image3)
    plt.axis('off')
    plt.title("Depth Label")