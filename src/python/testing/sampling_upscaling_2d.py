import matplotlib.pyplot as plt
import torch
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler
from monai import transforms
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

datalist = [{"t1w": "/media/walter/Storage/Downloads/sub-3144273/ses-1/sub-3144273_ses-1_slice-1_T1w.png"}]

val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["t1w"]),
        transforms.EnsureChannelFirstd(keys=["t1w"]),
        transforms.Rotate90d(keys=["t1w"], k=-1, spatial_axes=(0, 1)),  # Fix flipped image read
        transforms.Flipd(keys=["t1w"], spatial_axis=1),  # Fix flipped image read
        transforms.ScaleIntensityRanged(keys=["t1w"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        transforms.CenterSpatialCropd(keys=["t1w"], roi_size=[64, 96]),
        transforms.CopyItemsd(keys=["t1w"], times=1, names=["low_res_t1w"]),
        transforms.Resized(
            keys=["low_res_t1w"],
            spatial_size=[32, 48],
        ),
        transforms.ToTensord(keys=["t1w", "low_res_t1w"]),
    ]
)


device = torch.device("cuda")


config = OmegaConf.load(
    "/media/walter/Storage/Projects/generative_brain_upscaler/configs/upsampler_stage1_2d/aekl_v0.yaml"
)
upscaler_stage1_model = AutoencoderKL(**config["stage1"]["params"])
upscaler_stage1_model.load_state_dict(
    torch.load(
        "/media/walter/Storage/Projects/generative_brain_upscaler/outputs/trained_models_2d/autoencoder.pth",
    )
)
upscaler_stage1_model = upscaler_stage1_model.eval()
upscaler_stage1_model = upscaler_stage1_model.to(device)

config = OmegaConf.load("/media/walter/Storage/Projects/generative_brain_upscaler/configs/upsampler_ldm_2d/ldm_v0.yaml")
upscaler_ldm_model = DiffusionModelUNet(**config["ldm"].get("params", dict()))
upscaler_ldm_model.load_state_dict(
    torch.load(
        "/media/walter/Storage/Projects/generative_brain_upscaler/outputs/trained_models_2d/diffusion_model.pth",
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
scheduler.set_timesteps(1000)

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


sampling_image = batch[0]["low_res_t1w"].unsqueeze(0).to(device)

latents = torch.randn((1, 3, 32, 48)).to(device)
low_res_noise = torch.randn((1, 1, 32, 48)).to(device)
noise_level = 1
noise_level = torch.Tensor((noise_level,)).long().to(device)
noisy_low_res_image = scheduler.add_noise(
    original_samples=sampling_image, noise=low_res_noise, timesteps=torch.Tensor((noise_level,)).long().to(device)
)
scheduler.set_timesteps(num_inference_steps=1000)
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


plt.imshow(batch[0]["t1w"][0, :, :].cpu().numpy())
plt.show()

plt.imshow(batch[0]["low_res_t1w"][0, :, :].cpu().numpy())
plt.show()

plt.imshow(decoded[0, 0, :, :].cpu().numpy())
plt.show()
