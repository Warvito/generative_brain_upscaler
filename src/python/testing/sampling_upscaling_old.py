import matplotlib.pyplot as plt
import numpy as np
import torch
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler
from monai import transforms
from monai.utils import set_determinism
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# from generative.networks.schedulers import DDIMScheduler, DDPMScheduler

datalist = [{"image": "/media/walter/Storage/Downloads/sub-1000126_ses-1_T1w.nii.gz"}]

val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        transforms.SpatialCropd(keys=["image"], roi_start=[16, 16, 96], roi_end=[176, 240, 256]),
        transforms.SpatialPadd(
            keys=["image"],
            spatial_size=[160, 224, 160],
        ),
        transforms.Resized(
            keys=["image"],
            spatial_size=[80, 112, 80],
        ),
        transforms.ToTensord(keys=["image"]),
    ]
)
device = torch.device("cuda")


config = OmegaConf.load(
    "/media/walter/Storage/Projects/generative_brain_upscaler/configs/upsampler_stage1/aekl_v0.yaml"
)
upscaler_stage1_model = AutoencoderKL(**config["stage1"]["params"])
upscaler_stage1_model.load_state_dict(
    torch.load(
        "/media/walter/Storage/Projects/generative_brain_upscaler/outputs/trained_models_3d_old/autoencoder.pth",
    )
)
upscaler_stage1_model = upscaler_stage1_model.eval()
upscaler_stage1_model = upscaler_stage1_model.to(device)

config = OmegaConf.load("/media/walter/Storage/Projects/generative_brain_upscaler/configs/upsampler_ldm/ldm_v0.yaml")
upscaler_ldm_model = DiffusionModelUNet(**config["ldm"].get("params", dict()))
upscaler_ldm_model.load_state_dict(
    torch.load(
        "/media/walter/Storage/Projects/generative_brain_upscaler/outputs/trained_models_3d_old/diffusion_model.pth",
    )
)
upscaler_ldm_model = upscaler_ldm_model.eval()
upscaler_ldm_model = upscaler_ldm_model.to(device)

scheduler = DDIMScheduler(
    num_train_timesteps=config["ldm"]["scheduler"]["num_train_timesteps"],
    beta_start=config["ldm"]["scheduler"]["beta_start"],
    beta_end=config["ldm"]["scheduler"]["beta_end"],
    schedule=config["ldm"]["scheduler"]["schedule"],
    prediction_type=config["ldm"]["scheduler"]["prediction_type"],
    clip_sample=False,
)

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


set_determinism(seed=42)

batch = val_transforms(datalist)


sampling_image = batch[0]["image"].unsqueeze(0).to(device)
# sampling_image = sampling_image[:,:,:40,:56,:40]

latents = torch.randn((1, 3, 40, 56, 40)).to(device)
low_res_noise = torch.randn((1, 1, 40, 56, 40)).to(device)

latents = torch.randn((1, 3, 80, 112, 80)).to(device)
low_res_noise = torch.randn((1, 1, 80, 112, 80)).to(device)

noise_level = 1
noise_level = torch.Tensor((noise_level,)).long().to(device)
noisy_low_res_image = scheduler.add_noise(
    original_samples=sampling_image, noise=low_res_noise, timesteps=torch.Tensor((noise_level,)).long().to(device)
)
scheduler.set_timesteps(num_inference_steps=200)
for t in tqdm(scheduler.timesteps, ncols=110):
    with torch.no_grad():
        latent_model_input = torch.cat([latents, noisy_low_res_image], dim=1)
        noise_pred = upscaler_ldm_model(
            x=latent_model_input,
            timesteps=torch.Tensor((t,)).to(device),
            context=prompt_embeds,
            class_labels=noise_level,
        )
        latents, _ = scheduler.step(noise_pred, t, latents)

scale_factor = 0.3
with torch.no_grad():
    decoded = upscaler_stage1_model.decode_stage_2_outputs(latents / scale_factor)


# plt.imshow(batch[0]["image"][0, :, :, 20].cpu().numpy())
# plt.show()

plt.imshow(sampling_image[0, 0, :, :, 40].cpu().numpy())
plt.show()

plt.imshow(decoded[0, 0, :, :, 80].cpu().numpy())
plt.show()


img_row_0 = np.concatenate(
    (
        np.flipud(decoded[0, 0, 90, :, :].T),
        np.flipud(decoded[0, 0, :, 60, :].T),
    ),
    axis=1,
)

img_row_1 = np.concatenate(
    (
        decoded[0, 0, :, :, 70],
        np.zeros((160, 160)),
    ),
    axis=1,
)

img = np.concatenate(
    (
        img_row_0,
        img_row_1,
    ),
    axis=0,
)

# save image using pillow

img = Image.fromarray((img * 255).astype(np.uint8))
img.save("/media/walter/Storage/Projects/generative_brain_upscaler/outputs/figures/upsampled.png")


img_row_0 = np.concatenate(
    (
        np.flipud(sampling_image[0, 0, 45, :, :].T),
        np.flipud(sampling_image[0, 0, :, 30, :].T),
    ),
    axis=1,
)

img_row_1 = np.concatenate(
    (
        sampling_image[0, 0, :, :, 35],
        np.zeros((80, 80)),
    ),
    axis=1,
)

img = np.concatenate(
    (
        img_row_0,
        img_row_1,
    ),
    axis=0,
)

# save image using pillow
img = Image.fromarray((img * 255).astype(np.uint8))
img.save("/media/walter/Storage/Projects/generative_brain_upscaler/outputs/figures/downsampled.png")
