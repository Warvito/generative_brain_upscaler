stage1_mlflow_path="/project/mlruns/265350922640342393/9c6a6599815e4716a7b46b940f01900a/artifacts/final_model"
diffusion_mlflow_path="/project/mlruns/469170894764104566/30fb0822e22e4b46a0a7e4bf4f294635/artifacts/final_model"
output_dir="/project/outputs/trained_models_3d/"

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


stage1_mlflow_path="/project/mlruns/648461630920358299/3ddd8b39ecef4c2cbfa699b3f8e0337a/artifacts/final_model"
diffusion_mlflow_path="/project/mlruns/359900379465979962/42c00a52e03b43fdaabb2e3f642c42cb/artifacts/final_model"
output_dir="/project/outputs/trained_models_2d/"

runai submit \
  --name brain-convert-model \
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
