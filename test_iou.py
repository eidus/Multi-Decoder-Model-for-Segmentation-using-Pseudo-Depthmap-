from VocClasses import VOC_COLORMAP
import torch
import numpy as np
import segmentation_models_pytorch as smp

def convert_to_segmentation_mask2(mask):
    # batch = mask.shape[0]
    height, width = mask.shape[1:3]
    segmentation_mask = np.zeros((len(VOC_COLORMAP), height, width), dtype=np.float32)
    for label_index, label in enumerate(VOC_COLORMAP):
          segmentation_mask[label_index, :, :] = (mask == label_index).float()
    return segmentation_mask

def Test_iou(model, test_loader):
    
    device = torch.device('cuda:0')
    test_iou = []
    test_iIou = []
    test_acc = []

    for img, target, d in test_loader:
        img, target = img.to(device), target.to(device) # target : [1, 21, 256, 256]
        output, _ = model(img) # output : [1, 21, 256, 256]
        _, targets_u = torch.max(output, dim=1, keepdim = True)
        output_onehot = convert_to_segmentation_mask2(targets_u.squeeze(0).detach().cpu())
        tp, fp, fn, tn = smp.metrics.get_stats(torch.from_numpy(output_onehot).to(device), target.squeeze().int(), mode='multilabel', threshold=0.5)
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        
        tp, fp, fn, tn = smp.metrics.get_stats(torch.from_numpy(output_onehot[1:]).to(device), target.squeeze()[1:].int(), mode='multilabel', threshold=0.5)
        iiou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        test_iou.append(iou)
        test_acc.append(accuracy)
        test_iIou.append(iiou)

    return print(f"accuracy : {torch.tensor(test_acc).mean():.4f}, iou : {torch.tensor(test_iou).mean():.4f}, iIou : {torch.tensor(test_iIou).mean():.4f}")