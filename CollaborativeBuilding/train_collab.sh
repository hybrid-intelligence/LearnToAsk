#!/bin/bash

python3 train.py --json_data_dir builder_data_with_glove --saved_models_path saved_models_$1 --seed $1