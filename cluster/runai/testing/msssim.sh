seed=42
sample_dir="/project/outputs/downsampled_samples_unconditioned/"
num_workers=8

runai submit \
  --name downsampled-ssim-sample \
  --image aicregistry:5000/wds20:ldm_brain_upscaler \
  --backoff-limit 0 \
  --gpu 1 \
  --cpu 4 \
  --memory-limit 256G \
  --large-shm \
  --run-as-user \
  --node-type "A100" \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_brain_upscaler/:/project/ \
  --command -- bash /project/src/bash/start_script.sh \
    python3 /project/src/python/testing/compute_msssim_sample.py \
      seed=${seed} \
      sample_dir=${sample_dir} \
      num_workers=${num_workers}
