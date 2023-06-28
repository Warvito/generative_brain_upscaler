""" Script to upscale the test set """
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler
from monai import transforms
from monai.config import print_config
from monai.data import Dataset
from monai.utils import set_determinism
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from util import get_test_dataloader


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--output_dir", help="Location to save the output.")
    parser.add_argument("--test_ids", help="Location of file with test ids.")
    parser.add_argument("--config_file", help="Location of config file.")
    parser.add_argument("--stage1_path", help="Location of stage1 model.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")

    args = parser.parse_args()
    return args


def main(args):
    set_determinism(seed=args.seed)
    print_config()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Getting data...")
    test_loader = get_test_dataloader(
        batch_size=args.batch_size,
        test_ids=args.test_ids,
        num_workers=args.num_workers,
        upper_limit=1000,
    )

    print("Creating model...")
    device = torch.device("cuda")
    config = OmegaConf.load(args.config_file)
    stage1 = AutoencoderKL(**config["stage1"]["params"])
    stage1.load_state_dict(torch.load(args.stage1_path))
    stage1 = stage1.to(device)
    stage1.eval()

    config = OmegaConf.load(args.diffusion_config_file_path)
    diffusion = DiffusionModelUNet(**config["ldm"].get("params", dict()))
    diffusion.load_state_dict(torch.load(args.diffusion_path))
    diffusion = diffusion.to(device)
    diffusion.eval()

    scheduler = DDIMScheduler(
        num_train_timesteps=config["ldm"]["scheduler"]["num_train_timesteps"],
        beta_start=config["ldm"]["scheduler"]["beta_start"],
        beta_end=config["ldm"]["scheduler"]["beta_end"],
        schedule=config["ldm"]["scheduler"]["schedule"],
        prediction_type=config["ldm"]["scheduler"]["prediction_type"],
        clip_sample=False,
    )
    scheduler.set_timesteps(args.num_inference_steps)

    tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder")

    prompt = ["T1-weighted image of a brain."]
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    prompt_embeds = text_encoder(text_input_ids.squeeze(1))
    prompt_embeds = prompt_embeds[0].to(device)

    df = pd.read_csv(args.test_ids, sep="\t")
    df = df[args.start_seed : args.stop_seed]

    data_dicts = []
    for index, row in df.iterrows():
        data_dicts.append(
            {
                "image": str(row["image"]),
            }
        )

    eval_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            transforms.SpatialCropd(keys=["image"], roi_start=[16, 16, 96], roi_end=[176, 240, 256]),
            transforms.SpatialPadd(
                keys=["image"],
                spatial_size=[160, 224, 160],
            ),
            transforms.CopyItemsd(keys=["image"], times=1, names=["low_res_image"]),
            transforms.Resized(
                keys=["low_res_image"],
                spatial_size=[80, 112, 80],
            ),
            transforms.ToTensord(keys=["image", "low_res_image"]),
        ]
    )

    eval_ds = Dataset(
        data=data_dicts,
        transform=eval_transforms,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
    )

    reference_image = nib.load(args.reference_path)

    for batch in tqdm(eval_loader):
        low_res_image = batch["low_res_image"].to(device)

        latents = torch.randn((1, config["ldm"]["params"]["out_channels"], args.x_size, args.y_size, args.z_size)).to(
            device
        )
        low_res_noise = torch.randn((1, 1, args.x_size, args.y_size, args.z_size)).to(device)

        noise_level = torch.Tensor((args.noise_level,)).long().to(device)
        noisy_low_res_image = scheduler.add_noise(
            original_samples=low_res_image,
            noise=low_res_noise,
            timesteps=torch.Tensor((noise_level,)).long().to(device),
        )
        scheduler.set_timesteps(num_inference_steps=args.num_inference_steps)
        for t in tqdm(scheduler.timesteps, ncols=110):
            with torch.no_grad():
                latent_model_input = torch.cat([latents, noisy_low_res_image], dim=1)
                noise_pred = diffusion(
                    x=latent_model_input,
                    timesteps=torch.Tensor((t,)).to(device),
                    context=prompt_embeds,
                    class_labels=noise_level,
                )
                latents, _ = scheduler.step(noise_pred, t, latents)

        with torch.no_grad():
            sample = stage1.decode_stage_2_outputs(latents / args.scale_factor)

        sample = np.clip(sample.cpu().numpy(), 0, 1)

        sampled_nii = nib.Nifti1Image(sample[0, 0], reference_image.affine, reference_image.header)
        img_path = Path(batch["low_res_image_meta_dict"]["filename_or_obj"][0])
        nib.save(sampled_nii, output_dir / f"{img_path.stem}_upscaled.nii.gz")


if __name__ == "__main__":
    args = parse_args()
    main(args)
