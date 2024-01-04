import os
import tempfile
import glob
from monai.inferers import SliceInferer
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscreted,
    Compose,
    Invertd,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    Spacingd,
    EnsureTyped,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    SaveImaged,
    FillHolesd,
    KeepLargestConnectedComponentd,
)
from monai.networks.layers import Norm
from monai.utils import  set_determinism
from monai.data import (
    Dataset,
    DataLoader,
    decollate_batch,
)
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
roi_size = spatial_window_size[0:2]
spatial_window_batch_size = 4
batch_size_training = 50
amount_of_labels = 7
inference_iteration = 30000
pix_dim  = (1,1,1)
model_save_best = f'model_UNET_Lumbar_spine_iteration_{inference_iteration}.pth'
model_continue_training = f'model_UNET_Lumbar_spine_iteration_{inference_iteration}.pth'
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
        axial_inferer = SliceInferer(roi_size=roi_size, sw_batch_size=spatial_window_batch_size, spatial_dim=2)
        test_data["pred"] = axial_inferer(val_inputs, model)
        val_data = [post_transforms(i) for i in decollate_batch(test_data)]