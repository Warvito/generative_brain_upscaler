seed=42
run_dir="upsampler_aekl_v0_2d"
training_ids="/project/outputs/ids/train_2d.tsv"
validation_ids="/project/outputs/ids/validation_2d.tsv"
config_file="/project/configs/upsampler_stage1_2d/aekl_v0.yaml"
batch_size=128
n_epochs=30
adv_start=10
eval_freq=5
num_workers=128
experiment="UPSAMPLER-AEKL-2D"

runai submit \
  --name upscaler-aekl-v0-2d \
  --image aicregistry:5000/wds20:ldm_brain_upscaler \
  --backoff-limit 0 \
  --gpu 4 \
  --cpu 96 \
  --node-type "A100" \
  --large-shm \
  --run-as-user \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_brain_upscaler/:/project/ \
  --volume /nfs/home/wds20/datasets/Biobank/derivatives/2d_controlnet/:/data/ \
  --command -- bash /project/src/bash/start_script.sh \
    python3 /project/src/python/training/train_aekl_2d.py \
      seed=${seed} \
      run_dir=${run_dir} \
      training_ids=${training_ids} \
      validation_ids=${validation_ids} \
      config_file=${config_file} \
      batch_size=${batch_size} \
      n_epochs=${n_epochs} \
      adv_start=${adv_start} \
      eval_freq=${eval_freq} \
      num_workers=${num_workers} \
      experiment=${experiment}
