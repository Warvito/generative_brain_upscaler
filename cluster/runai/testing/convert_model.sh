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


stage1_mlflow_path="/project/mlruns/265350922640342393/c9be892664404cd5a261eae91d941b6f/artifacts/final_model"
diffusion_mlflow_path="/project/mlruns/923773097242416920/f326b39adb80429ea19b3b124a422e0e/artifacts/final_model"
output_dir="/project/outputs/trained_models_3d_upscaler/"

runai submit \
  --name upscaler-convert-model \
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
