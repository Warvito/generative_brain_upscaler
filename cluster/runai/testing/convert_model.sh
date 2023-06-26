stage1_mlflow_path="/project/mlruns/289718304593103846/8bb32928ebba416d9de589a20e1fae5f/artifacts/final_model"
diffusion_mlflow_path="/project/mlruns/691775524573125461/67ea3b870975473387e434a77abd36a0/artifacts/final_model"
output_dir="/project/outputs/trained_models_3d_downsampled/"

runai submit \
  --name downsampled-convert-model \
  --image aicregistry:5000/wds20:ldm_brain_upscaler \
  --backoff-limit 0 \
  --gpu 1 \
  --cpu 4 \
  --large-shm \
  --run-as-user \
  --node-type "A100" \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_brain_upscaler/:/project/ \
  --command -- bash /project/src/bash/start_script.sh \
    python3 /project/src/python/testing/convert_mlflow_to_pytorch.py \
      stage1_mlflow_path=${stage1_mlflow_path} \
      diffusion_mlflow_path=${diffusion_mlflow_path} \
      output_dir=${output_dir}


stage1_mlflow_path="/project/mlruns/751866889003045521/a2cb93e5a920445e932ce29e09bd5a82/artifacts/final_model"
diffusion_mlflow_path="/project/mlruns/275410112256848408/62010f5537c54faf9caef1dfdf3d48ac/artifacts/final_model"
output_dir="/project/outputs/trained_models_3d_old/"

runai submit \
  --name upscaler-convert-model-old \
  --image aicregistry:5000/wds20:ldm_brain_upscaler \
  --backoff-limit 0 \
  --gpu 1 \
  --cpu 4 \
  --large-shm \
  --run-as-user \
  --node-type "A100" \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_brain_upscaler/:/project/ \
  --command -- bash /project/src/bash/start_script.sh \
    python3 /project/src/python/testing/convert_mlflow_to_pytorch.py \
      stage1_mlflow_path=${stage1_mlflow_path} \
      diffusion_mlflow_path=${diffusion_mlflow_path} \
      output_dir=${output_dir}
