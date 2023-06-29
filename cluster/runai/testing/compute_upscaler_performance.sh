seed=42
samples_dir="/project/outputs/testset_unconditioned_upsampled/"
test_ids="/project/outputs/ids/test.tsv"
num_workers=8

runai submit \
  --name upscaler-performance \
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
  --volume /nfs/project/AMIGO/Biobank/derivatives/super-res/:/data/ \
  --command -- sleep infinity
  --command -- bash /project/src/bash/start_script.sh \
    python3 /project/src/python/testing/compute_upscaler_performance.py \
      seed=${seed} \
      samples_dir=${samples_dir} \
      test_ids=${test_ids} \
      num_workers=${num_workers}
