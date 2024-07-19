import torchvision.utils as vutils
import os
import torch
from PIL import Image
import math
import itertools
from functools import partial
import torch.nn.functional as F
import urllib
import mmcv
from mmcv.runner import load_checkpoint
from dinov2.eval.depth.models import build_depther
from torchvision import transforms
import sys
# sys.path.append('/home/dinov2') # if you need

# set directory
os.makedirs(name = os.path.split(os.path.dirname(os.getcwd()))[0] + '/Detph_image_2012', exist_ok = False)

# dinov2 from https://github.com/facebookresearch/dinov2/blob/main/notebooks/depth_estimation.ipynb
class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


def create_depther(cfg, backbone_model, backbone_size, head_type):
    train_cfg = cfg.get("train_cfg")
    test_cfg = cfg.get("test_cfg")
    depther = build_depther(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)

    depther.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
        return_class_token=cfg.model.backbone.output_cls_token,
        norm=cfg.model.backbone.final_norm,
    )

    if hasattr(backbone_model, "patch_size"):
        depther.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))

    return depther

BACKBONE_SIZE = "large" # in ("small", "base", "large" or "giant")


backbone_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}
backbone_arch = backbone_archs[BACKBONE_SIZE]
backbone_name = f"dinov2_{backbone_arch}"

backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
backbone_model.eval()
backbone_model.cuda()

def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


HEAD_DATASET = "nyu" # in ("nyu", "kitti")
HEAD_TYPE = "dpt" # in ("linear", "linear4", "dpt")


DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

cfg_str = load_config_from_url(head_config_url)
cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

dino_v2 = create_depther(
    cfg,
    backbone_model=backbone_model,
    backbone_size=BACKBONE_SIZE,
    head_type=HEAD_TYPE,
)

load_checkpoint(dino_v2, head_checkpoint_url, map_location="cpu")
dino_v2.eval()
dino_v2.cuda()

# transform
def make_depth_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
        transforms.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        ),
    ])

transform = make_depth_transform()
scale_factor = 1

# image directory
dir = f'{os.path.split(os.path.dirname(os.getcwd()))[0]}/PASCAL_VOC_2012/'
root=os.path.join(dir,'VOCdevkit/VOC2012')
images_dir=os.path.join(root,'JPEGImages')
file_list=os.path.join(root,'ImageSets/Segmentation/trainval.txt')
files = [line.rstrip() for line in tuple(open(file_list, "r"))]

# save depth image
for i, file in enumerate(files):

    image_id=files[i]
    image_path=os.path.join(images_dir,f"{image_id}.jpg")
    image = Image.open(image_path)
    
    rescaled_image = image.resize((scale_factor * image.width, scale_factor * image.height))
    transformed_image = transform(rescaled_image)
    batch = transformed_image.unsqueeze(0).cuda() # Make a batch of one image

    with torch.inference_mode():
        result = dino_v2.whole_inference(batch, img_meta=None, rescale=True)

    result2 = result.squeeze(0)
    
    save_dir = f'{os.path.split(os.path.dirname(os.getcwd()))[0]}/Detph_image_2012'
    name = save_dir+ '/Depth_'+file+'.jpg'
    name2 = save_dir+'/Depth_'+file+'.pth'
    torch.save(result2, name2)

    result_normalized = (result2 - result2.min()) / (result2.max() - result2.min())
    vutils.save_image(result_normalized,name)