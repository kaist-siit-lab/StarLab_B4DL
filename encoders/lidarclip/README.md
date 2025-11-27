
please download [ViT-L-14.pt](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) and place it under ./pretrained/

You need to train the model first, and with that trained weight, extract the features before running the B4DL model.

```sh
python train.py --name=lidarclip --checkpoint-save-dir=./ckpt --batch-size 128 --workers 4 --data-dir /path/to/dataset --clip-model ViT-L/14

python extract_pc_features.py --checkpoint=/path/to/trained/.ckpt --scene-json-path ./annotations/scene_metadata.json --frame-json-path ./annotations/sequence_metadata.json \
        --stage1-save-dir ./b4dl/stage1_features/ --stage2-save-dir ./b4dl/stage2_features/

#### SIIT only
python train.py --name=lidarclip --checkpoint-save-dir=./ckpt --batch-size 128 --workers 4 --data-dir /mnt/nfs_shared_data/dataset/cch/nuScenes/ --clip-model ViT-L/14

python extract_pc_features.py --checkpoint=./ckpt/epoch=19-step=26000.ckpt --scene-json-path ./annotations/scene_metadata.json \
    --frame-json-path ./annotations/sequence_metadata.json --stage1-save-dir ./extracted_features/stage1_features/
    --stage2-save-dir ./extracted_features/stage2_features/