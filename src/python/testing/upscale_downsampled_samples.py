""" Script to upscale the samples of the downsampled LDM."""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, help="Path to the MLFlow artifact of the stage1.")
    parser.add_argument("--output_dir", help="Location to save the output.")
    parser.add_argument("--downsampled_dir", help="Location to save the output.")
    parser.add_argument("--stage1_path", help="Path to the .pth model from the stage1.")
    parser.add_argument("--diffusion_path", help="Path to the .pth model from the diffusion model.")
    parser.add_argument("--stage1_config_file_path", help="Path to the .pth model from the stage1.")
    parser.add_argument("--diffusion_config_file_path", help="Path to the .pth model from the diffusion model.")
    parser.add_argument("--reference_path", help="Path to the reference image.")
    parser.add_argument("--start_index", type=int, help="Path to the MLFlow artifact of the stage1.")
    parser.add_argument("--stop_index", type=int, help="Path to the MLFlow artifact of the stage1.")
    parser.add_argument("--x_size", type=int, default=64, help="Latent space x size.")
    parser.add_argument("--y_size", type=int, default=64, help="Latent space y size.")
    parser.add_argument("--z_size", type=int, default=64, help="Latent space z size.")
    parser.add_argument("--scale_factor", type=float, help="Latent space y size.")
    parser.add_argument("--num_inference_steps", type=int, help="")
    parser.add_argument("--noise_level", type=int, help="")

    args = parser.parse_args()
    return args


def main(args):
    set_determinism(seed=args.seed)
    print_config()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Creating model...")
    device = torch.device("cuda")
    config = OmegaConf.load(args.stage1_config_file_path)
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

    # Samples
    samples_dir = Path(args.downsampled_dir)
    samples_datalist = []
    for sample_path in sorted(list(samples_dir.glob("*.nii.gz"))):
        samples_datalist.append(
            {
                "low_res_image": str(sample_path),
            }
        )
    print(f"{len(samples_datalist)} images found in {str(samples_dir)}")
    samples_datalist = samples_datalist[args.start_index : args.stop_index]

    sample_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["low_res_image"]),
            transforms.EnsureChannelFirstd(keys=["low_res_image"]),
            transforms.ToTensord(keys=["low_res_image"]),
        ]
    )

    samples_ds = Dataset(
        data=samples_datalist,
        transform=sample_transforms,
    )
    samples_loader = DataLoader(
        samples_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
    )

    reference_image = nib.load(args.reference_path)

    for batch in tqdm(samples_loader):
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
