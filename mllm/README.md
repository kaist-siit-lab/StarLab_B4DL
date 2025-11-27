# Training Script


Before running, please download [this file](https://huggingface.co/lmsys/vicuna-7b-v1.5/tree/main) and place it under ./base_model/

```shell
bash runner.sh \
     --s1_data ./b4dl_dataset/stage1_lidarllm_mm.json \
     --s1_feat ./b4dl/stage1_features \
     --s2_data ./b4dl_dataset/stage2.json \
     --s2_feat ./b4dl/stage2_features \
     --model_name_or_path ./base_model/vicuna-v1-5-7b

### SIIT only
bash runner.sh \
     --s1_data /home/youngwoo.shin/ftp_home/Projects/SW스타랩/4d_datageneration/mm/2-step/lidarllm_only_dataset/stage1_lidarllm_mm.json \
     --s1_feat /home/youngwoo.shin/ftp_home/Projects/SW스타랩/4d_datageneration/mm/extracted_features/stage1_features \
     --s2_data /home/youngwoo.shin/ftp_home/Projects/SW스타랩/4d_datageneration/mm/2-step/final_dataset/mm_version/700_GT_token_dataset/train/stage2_combined.json \
     --s2_feat /home/youngwoo.shin/ftp_home/Projects/SW스타랩/4d_datageneration/mm/extracted_features/stage2_features \
     --model_name_or_path /home/youngwoo.shin/ftp_home/Projects/SW스타랩/4d_datageneration/mm/vtimellm_checkpoints/vicuna-7b-v1.5