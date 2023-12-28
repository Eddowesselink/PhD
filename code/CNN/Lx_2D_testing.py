import os
import shutil
import tempfile
import glob
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference,SliceInferer
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from ignite.engine import Engine
from monai.transforms import (
    AsDiscreted,
    AddChanneld,
    Compose,
    Invertd,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    BatchInverseTransform,
    EnsureTyped,
    EnsureType,
    MapTransform,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    SaveImaged,
    FillHolesd,
    KeepLargestConnectedComponentd,
)
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR
from monai.networks.layers import Norm
from monai.utils import first, set_determinism
from monai.transforms.utils import allow_missing_keys_mode
from monai.transforms import Activations, AsChannelFirstd, AsDiscrete, Compose, LoadImaged, SaveImage, ScaleIntensityd, EnsureTyped, EnsureType
from monai.data import (
    Dataset,
    DataLoader,
    CacheDataset,
    decollate_batch,
)
from monai.data import NiftiSaver
from monai.transforms import InvertibleTransform
import torch

#setup data directory
directory = os.environ.get("Monai_Directory")
root_dir = tempfile.mkdtemp() if directory is None else directory
save_dir = 'D:/PS_Muscle_Segmentation/Monai/Lumbar_spine/CNN'

# load dataset 
data_dir = 'D:/PS_Muscle_Segmentation/Monai/Lumbar_spine'
images = sorted(glob.glob(os.path.join(data_dir, "testing", "*F.nii.gz")))
test_files = [{"image": img} for img in zip(images)]

spatial_window_size = (256,256,1)
spatial_window_batch_size = 4
batch_size_training = 50
amount_of_labels = 7
batch_size_validation = 1
starting_iteration = 30000
dice_val_best = 0
global_step_best = 0
pix_dim  = (1,1,1)
max_iterations = 30010
eval_num = 500
model_save_best = f'model_UNET_Lumbar_spine_iteration_{starting_iteration}.pth'
model_continue_training = f'model_UNET_Lumbar_spine_iteration_{starting_iteration}.pth'
learning_rate = 1e-4
weight_decay= 1e-5

#set seed for reproducibility (identical to training part)
set_determinism(seed=0)

#create transforms identical to training part, but here we don't specifiy the label key
validation_original = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(
            keys=["image"],
            pixdim=pix_dim,
            mode=("bilinear")),
        Orientationd(keys=["image"], axcodes="RAS"),
        CropForegroundd(keys=["image"], source_key="image"),
        NormalizeIntensityd(keys=["image"], nonzero = True),
        EnsureTyped(keys=["image"])
    ])

#create iterable dataset and dataloader, identical to training part
validation_original_dataset = Dataset(
    data=test_files, transform=validation_original,
)

validation_original_loader = DataLoader(
    validation_original_dataset , batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

#device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

post_transforms = Compose([
    Invertd(
        keys="pred",
        transform=validation_original,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=False,
        to_tensor=True,
        device = device
    ),
    AsDiscreted(keys="pred", argmax=True),
    FillHolesd(keys="pred", applied_labels=[1,2,3,4,5,6,7]),
    KeepLargestConnectedComponentd(keys="pred", applied_labels=[1,2,3,4,5,6,7]),
    SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=save_dir, output_dtype=('int16'), resample=False, separate_folder=False)
])

model= UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=amount_of_labels,
    channels=(16, 32, 64, 128, 256),
    act= 'LeakyRelu',
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.INSTANCE,
).to(device)

#load pre-excisting model 
model.load_state_dict(torch.load(
    os.path.join(root_dir,'Models',"lumbar_spine_UNET_per_iteration",model_save_best)))
model.eval()

with torch.no_grad():
    for i, test_data in enumerate(validation_original_loader):
        val_inputs = test_data["image"].to(device)
        axial_inferer = SliceInferer(roi_size=(256,256), sw_batch_size=spatial_window_batch_size, spatial_dim=2)
        test_data["pred"] = axial_inferer(val_inputs, model)
        val_data = [post_transforms(i) for i in decollate_batch(test_data)]