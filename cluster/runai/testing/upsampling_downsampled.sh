downsampled_dir="/project/outputs/downsampled_samples_unconditioned/"
output_dir="/project/outputs/samples_unconditioned_upsampled/"
stage1_path="/project/outputs/trained_models_3d_upscaler/autoencoder.pth"
diffusion_path="/project/outputs/trained_models_3d_upscaler/diffusion_model.pth"
stage1_config_file_path="/project/configs/upsampler_stage1/aekl_v0.yaml"
diffusion_config_file_path="/project/configs/upsampler_ldm/ldm_v0.yaml"
reference_path="/project/outputs/reference_image/super-res.nii.gz"
x_size=80
y_size=112
z_size=80
scale_factor=0.3
noise_level=1
num_inference_steps=200

for i in {0..9}; do
  start_index=$((i*100))
  stop_index=$(((i+1)*100))
  runai submit \
    --name  upscaler-sampling-${start_index}-${stop_index} \
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
      python3 /project/src/python/testing/upscale_downsampled_samples.py \
        downsampled_dir=${downsampled_dir} \
        output_dir=${output_dir} \
        stage1_path=${stage1_path} \
        diffusion_path=${diffusion_path} \
        stage1_config_file_path=${stage1_config_file_path} \
        diffusion_config_file_path=${diffusion_config_file_path} \
        reference_path=${reference_path} \
        start_index=${start_index} \
        stop_index=${stop_index} \
        noise_level=${noise_level} \
        x_size=${x_size} \
        y_size=${y_size} \
        z_size=${z_size} \
        scale_factor=${scale_factor} \
        num_inference_steps=${num_inference_steps}
done



seed=42
output_dir="/project/outputs/testset_unconditioned_upsampled/"
stage1_path="/project/outputs/trained_models_3d_upscaler/autoencoder.pth"
diffusion_path="/project/outputs/trained_models_3d_upscaler/diffusion_model.pth"
stage1_config_file_path="/project/configs/upsampler_stage1/aekl_v0.yaml"
diffusion_config_file_path="/project/configs/upsampler_ldm/ldm_v0.yaml"
reference_path="/project/outputs/reference_image/super-res.nii.gz"
x_size=80
y_size=112
z_size=80
scale_factor=0.3
noise_level=1
num_inference_steps=200
test_ids="/project/outputs/ids/test.tsv"

for i in 0; do
  start_index=$((i*100))
  stop_index=$(((i+1)*100))
  runai submit \
    --name  upscaler-testset-${start_index}-${stop_index}-1 \
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
    --volume /nfs/project/AMIGO/Biobank/derivatives/super-res/:/data/ \
    --command -- sleep infinity
    --command -- bash /project/src/bash/start_script.sh \
      python3 /project/src/python/testing/upscale_downsampled_test_set.py \
        seed=${seed} \
        output_dir=${output_dir} \
        stage1_path=${stage1_path} \
        diffusion_path=${diffusion_path} \
        stage1_config_file_path=${stage1_config_file_path} \
        diffusion_config_file_path=${diffusion_config_file_path} \
        reference_path=${reference_path} \
        start_index=${start_index} \
        stop_index=${stop_index} \
        noise_level=${noise_level} \
        x_size=${x_size} \
        y_size=${y_size} \
        z_size=${z_size} \
        scale_factor=${scale_factor} \
        test_ids=${test_ids} \
        num_inference_steps=${num_inference_steps}
done
