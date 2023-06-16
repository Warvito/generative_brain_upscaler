""" Script to generate downsampled images from the LDM and upscale it. """


import mlflow.pytorch
import torch
from generative.networks.schedulers import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from omegaconf import OmegaConf
from monai.utils import set_determinism
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda")

down_stage1_model = mlflow.pytorch.load_model(
    "/media/walter/Storage/Projects/generative-brain/mlruns/344968604149660181/00181802989a4a64b590acd78bb62ef7/artifacts/final_model"
)
down_stage1_model = down_stage1_model.to(device)
down_stage1_model = down_stage1_model.eval()

down_ldm_model = mlflow.pytorch.load_model(
    "/media/walter/Storage/Projects/generative-brain/mlruns/275410112256848408/b58a8d0934904405b397b249c90cdbb5/artifacts/final_model"
)
down_ldm_model = down_ldm_model.to(device)
down_ldm_model = down_ldm_model.eval()

config = OmegaConf.load("/media/walter/Storage/Projects/generative-brain/configs/downsampled_ldm/ldm_v0.yaml")
scheduler = DDIMScheduler(
    num_train_timesteps=config["ldm"]["scheduler"]["num_train_timesteps"],
    beta_start=config["ldm"]["scheduler"]["beta_start"],
    beta_end=config["ldm"]["scheduler"]["beta_end"],
    schedule=config["ldm"]["scheduler"]["schedule"],
    prediction_type=config["ldm"]["scheduler"]["prediction_type"],
    clip_sample=False,
)
scheduler.set_timesteps(200)

tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder")

prompt = ["", ""]
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

x_size=10
y_size=14
z_size=10
scale_factor=0.3

noise = torch.randn((1, config["ldm"]["params"]["in_channels"], x_size, y_size, z_size)).to(
    device
)

with torch.no_grad():
    progress_bar = tqdm(scheduler.timesteps)
    for t in progress_bar:
        noise_input = torch.cat([noise] * 2)
        model_output = down_ldm_model(
            noise_input, timesteps=torch.Tensor((t,)).to(noise.device).long(), context=prompt_embeds
        )
        noise_pred_uncond, noise_pred_text = model_output.chunk(2)
        noise_pred = noise_pred_uncond + 0 * (noise_pred_text - noise_pred_uncond)

        noise, _ = scheduler.step(noise_pred, t, noise)

with torch.no_grad():
    sample = down_stage1_model.decode_stage_2_outputs(noise / scale_factor)

sample = np.clip(sample.cpu().numpy(), 0, 1)
sample = (sample * 255).astype(np.uint8)

plt.imshow(sample[0, 0, :, :, 40])
plt.show()

# UPSCALER PART
upscaler_stage1_model = mlflow.pytorch.load_model(
    "/media/walter/Storage/Projects/generative-brain/mlruns/751866889003045521/a2cb93e5a920445e932ce29e09bd5a82/artifacts/final_model"
)
upscaler_stage1_model = upscaler_stage1_model.to(device)
upscaler_stage1_model = upscaler_stage1_model.eval()

upscaler_ldm_model = mlflow.pytorch.load_model(
    "/media/walter/Storage/Projects/generative-brain/mlruns/275410112256848408/62010f5537c54faf9caef1dfdf3d48ac/artifacts/final_model"
)
upscaler_ldm_model = upscaler_ldm_model.to(device)
upscaler_ldm_model = upscaler_ldm_model.eval()

config = OmegaConf.load("/media/walter/Storage/Projects/generative-brain/configs/upsampler_ldm/ldm_v0.yaml")
scheduler = DDIMScheduler(
    num_train_timesteps=config["ldm"]["scheduler"]["num_train_timesteps"],
    beta_start=config["ldm"]["scheduler"]["beta_start"],
    beta_end=config["ldm"]["scheduler"]["beta_end"],
    schedule=config["ldm"]["scheduler"]["schedule"],
    prediction_type=config["ldm"]["scheduler"]["prediction_type"],
    clip_sample=False,
)
scheduler.set_timesteps(200)

tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder")

prompt = [""]
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
x_size=80
y_size=112
z_size=80
scale_factor=0.3

sampling_image = torch.Tensor(sample[:,:,0:40, 0:56, 0:40]).to(device)

latents = torch.randn((1, 3, 40, 56, 40)).to(device)
low_res_noise = torch.randn((1, 1, 40, 56, 40)).to(device)
noise_level = 20
noise_level = torch.Tensor((noise_level,)).long().to(device)
noisy_low_res_image = scheduler.add_noise(
    original_samples=sampling_image, noise=low_res_noise, timesteps=torch.Tensor((noise_level,)).long().to(device)
)
scheduler.set_timesteps(num_inference_steps=1000)
for t in tqdm(scheduler.timesteps, ncols=110):
    with torch.no_grad():
        latent_model_input = torch.cat([latents, noisy_low_res_image], dim=1)
        noise_pred = upscaler_ldm_model(x=latent_model_input, timesteps=torch.Tensor((t,)).to(device), context=prompt_embeds, class_labels=noise_level)

        # 2. compute previous image: x_t -> x_t-1
        latents, _ = scheduler.step(noise_pred, t, latents)

with torch.no_grad():
    decoded = upscaler_stage1_model.decode_stage_2_outputs(latents / scale_factor)

plt.imshow(decoded[0, 0, :, :, 20].cpu().numpy())
plt.show()