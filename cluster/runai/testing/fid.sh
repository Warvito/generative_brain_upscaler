seed=42
sample_dir="/project/outputs/downsampled_samples_unconditioned/"
test_ids="/project/outputs/ids/test.tsv"
num_workers=8
batch_size=16

runai submit \
  --name downsampled-fid \
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
  --command -- bash /project/src/bash/start_script.sh \
      python3 /project/src/python/testing/compute_downsampled_fid.py \
      seed=${seed} \
      sample_dir=${sample_dir} \
      test_ids=${test_ids} \
      batch_size=${batch_size} \
      num_workers=${num_workers}


seed=42
sample_dir="/project/outputs/samples_unconditioned_upsampled/"
test_ids="/project/outputs/ids/test.tsv"
num_workers=8
batch_size=16

runai submit \
  --name upscaler-fid \
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
  --command -- bash /project/src/bash/start_script.sh \
      python3 /project/src/python/testing/compute_fid.py \
      seed=${seed} \
      sample_dir=${sample_dir} \
      test_ids=${test_ids} \
      batch_size=${batch_size} \
      num_workers=${num_workers}
