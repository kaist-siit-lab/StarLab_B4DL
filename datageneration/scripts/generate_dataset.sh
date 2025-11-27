#!/bin/bash

python3 generate_dataset.py \
    --start_index 0 \
    --end_index 10 \
    --api_key {your openai api key} \
    --dataroot ./data \
    --task existence