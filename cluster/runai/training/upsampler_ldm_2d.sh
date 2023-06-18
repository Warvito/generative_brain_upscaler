seed=42
run_dir="aekl_v0_ldm_v0_upsampler_2d"
training_ids="/project/outputs/ids/train_2d.tsv"
validation_ids="/project/outputs/ids/validation_2d.tsv"
stage1_uri="/project/mlruns/648461630920358299/3ddd8b39ecef4c2cbfa699b3f8e0337a/artifacts/final_model"
config_file="/project/configs/upsampler_ldm_2d/ldm_v0.yaml"
scale_factor=0.3
batch_size=1024
n_epochs=100
eval_freq=10
num_workers=256
experiment="LDM-2D"

runai submit \
  --name upscaler-ldm-v0-2d \
  --image aicregistry:5000/wds20:ldm_brain_upscaler \
  --backoff-limit 0 \
  --gpu 8 \
  --cpu 160 \
  --node-type "A100" \
  --large-shm \
  --run-as-user \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_brain_upscaler/:/project/ \
  --volume /nfs/home/wds20/datasets/Biobank/derivatives/2d_controlnet/:/data/ \
  --command -- bash /project/src/bash/start_script.sh \
      python3 /project/src/python/training/train_upsampler_ldm_2d.py \
      seed=${seed} \
      run_dir=${run_dir} \
      training_ids=${training_ids} \
      validation_ids=${validation_ids} \
      stage1_uri=${stage1_uri} \
      config_file=${config_file} \
      scale_factor=${scale_factor} \
      batch_size=${batch_size} \
      n_epochs=${n_epochs} \
      eval_freq=${eval_freq} \
      num_workers=${num_workers} \
      experiment=${experiment}
