seed=42
run_dir="downsampled_aekl_v0_ldm_v0"
training_ids="/project/outputs/ids/train.tsv"
validation_ids="/project/outputs/ids/validation.tsv"
stage1_uri="/project/mlruns/289718304593103846/8bb32928ebba416d9de589a20e1fae5f/artifacts/final_model"
config_file="/project/configs/downsampled_ldm/ldm_v0.yaml"
scale_factor=0.3
batch_size=8
n_epochs=25
eval_freq=1
num_workers=8
experiment="DOWNSAMPLED-LDM"

runai submit \
  --name downsampled-ldm-v0 \
  --image aicregistry:5000/wds20:ldm_brain_upscaler \
  --backoff-limit 0 \
  --gpu 4 \
  --cpu 128 \
  --memory-limit 320G \
  --large-shm \
  --run-as-user \
  --node-type "A100" \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_brain_upscaler/:/project/ \
  --volume /nfs/project/AMIGO/Biobank/derivatives/super-res/:/data/ \
  --command -- bash /project/src/bash/start_script.sh \
      python3 /project/src/python/training/train_downsampled_ldm.py \
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
