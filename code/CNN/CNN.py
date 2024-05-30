import os
import tempfile
import glob
import argparse
import torch
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
from monai.utils import set_determinism
from monai.data import Dataset, DataLoader, decollate_batch

def main(data_dir, model_dir):
    save_dir = os.path.join(args.data_dir, "CNN")
    images = sorted(glob.glob(os.path.join(args.data_dir, "*img.nii.gz")))
    # Check if any image files are found
    if not images:
        raise ValueError("No MRI images ending with 'img.nii.gz' found in the specified directory.")
    test_files = [{"image": img} for img in zip(images)]

    # Set important parameters
    roi_size = (112,112)
    spatial_window_batch_size = 1
    amount_of_labels = 9
    pix_dim  = (1,1,-1)

    # Set seed for reproducibility
    set_determinism(seed=0)

    # Setup transforms
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

    # Create dataset and dataloader
    validation_original_dataset = Dataset(data=test_files, transform=validation_original)
    validation_original_loader = DataLoader(validation_original_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # Device config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup post transforms
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
            device=device
        ),
        AsDiscreted(keys="pred", argmax=True),
        FillHolesd(keys="pred", applied_labels=[1,2,3,4,5,6,7,8,9]),
        KeepLargestConnectedComponentd(keys="pred", applied_labels=[1,2,3,4,5,6,7,8,9]),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=save_dir, output_dtype=('int16'), separate_folder = False, resample=False)
    ])

    # Create model
    model= UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=amount_of_labels,
        channels=(64, 128, 256, 512, 1024),
        act= 'LeakyRelu',
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.INSTANCE,
    ).to(device)

    # Load pre-existing model if continuing training
    model.load_state_dict(torch.load(os.path.join(args.model_dir)))
    model.eval()
    
    # Inference
    with torch.no_grad():
        for i, test_data in enumerate(validation_original_loader):
            val_inputs = test_data["image"].to(device)
            axial_inferer = SliceInferer(roi_size=roi_size, sw_batch_size=spatial_window_batch_size, spatial_dim=2)
            test_data["pred"] = axial_inferer(val_inputs, model)
            val_data = [post_transforms(i) for i in decollate_batch(test_data)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment MRI images using a pre-trained model.")
    parser.add_argument("--data_dir", type=str, help="Path to the data directory")
    parser.add_argument("--model_dir", type=str, help="Path to the directory containing the model parameters")
    args = parser.parse_args()
    main(args.data_dir, args.model_dir)
    print('There you go!')
    print('Happy to help!')