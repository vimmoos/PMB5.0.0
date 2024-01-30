#!/bin/bash

# NOTE: Run this script from the same folder it is in (exp)

# set environment variables
export HF_HUB_CACHE="/scratch/$USER/.cache/huggingface/hubS"
export MODEL_SAVEPATH="/scratch/$USER/compsem_models/"


# load modules
echo "-> Purging Modules"
module purge

echo "-> Loading Python3.11"
module load Python/3.11.3-GCCcore-12.3.0
echo -n "\$python3 --version: "
python3 --version

# Command for (creating and) activating a venv and installing required Python packages
if [ ! -d "env" ]; then
    echo "-> Creating new venv"
    python3 -m venv env
fi

echo "-> Activating venv"
source env/bin/activate

echo "-> Installing/updating requirements.txt"
pip install -r requirements.txt | grep -v 'already satisfied'

# running the script
echo "-> Running __main__.py"
python3 __main__.py --language en --model_name t5-base --train-split gold --epochs 2 --print
# for more comand line args run `python3 __main__.py --help` or look in parser.py