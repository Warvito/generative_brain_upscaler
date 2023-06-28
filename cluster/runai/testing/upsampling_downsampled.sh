downsampled_dir="/project/outputs/downsampled_samples_unconditioned/"
output_dir="/project/outputs/samples_unconditioned_upsampled/"
stage1_path="/project/outputs/trained_models_3d_upscaler/autoencoder.pth"
diffusion_path="/project/outputs/trained_models_3d_upscaler/diffusion_model.pth"
stage1_config_file_path="/project/configs/upsampler_stage1/aekl_v0.yaml"
diffusion_config_file_path="/project/configs/upsampler_ldm/ldm_v0.yaml"
reference_path="/project/outputs/reference_image/super-res.nii.gz"
prompt="uncondtioned"
guidance_scale=0.0
x_size=80
y_size=112
z_size=80
scale_factor=0.3
noise_level=1
num_inference_steps=200

for i in {0..3}; do
  start_seed=$((i*250))
  stop_seed=$(((i+1)*250))
  runai submit \
    --name  downsampled-sampling-${start_seed}-${stop_seed} \
    --image aicregistry:5000/wds20:ldm_brain_upscaler \
    --backoff-limit 0 \
    --gpu 1 \
    --cpu 4 \
    --node-type "A100" \
    --large-shm \
    --run-as-user \
    --host-ipc \
    --project wds20 \
    --volume /nfs/home/wds20/projects/generative_brain_upscaler/:/project/ \
    --command -- bash /project/src/bash/start_script.sh \
      python3 /project/src/python/testing/sample_images.py \
        output_dir=${output_dir} \
        stage1_path=${stage1_path} \
        diffusion_path=${diffusion_path} \
        stage1_config_file_path=${stage1_config_file_path} \
        diffusion_config_file_path=${diffusion_config_file_path} \
        reference_path=${reference_path} \
        start_seed=${start_seed} \
        stop_seed=${stop_seed} \
        prompt=${prompt} \
        guidance_scale=${guidance_scale} \
        x_size=${x_size} \
        y_size=${y_size} \
        z_size=${z_size} \
        scale_factor=${scale_factor} \
        num_inference_steps=${num_inference_steps}
done


runai submit \
  --name  downsampled \
  --image aicregistry:5000/wds20:ldm_brain_upscaler \
  --backoff-limit 0 \
  --gpu 1 \
  --cpu 4 \
  --node-type "A100" \
  --large-shm \
  --run-as-user \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_brain_upscaler/:/project/ \
  --command -- sleep infinity
