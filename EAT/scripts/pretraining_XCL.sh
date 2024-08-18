#!/bin/bash
#SBATCH --job-name=EAT-XCL
#SBATCH --output=%j.log
#SBATCH --error=%j.err
#SBATCH --ntasks=1
#SBATCH --mem=100gb
#SBATCH --time=50:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=main
#SBATCH --gres=gpu:1

date;hostname;pwd
source /mnt/stud/work/python/mconda/39/bin/activate base
conda init zsh
conda activate birdset

export CUDA_VISIBLE_DEVICES=0,1,2,3

python fairseq_cli/hydra_train.py -m \
    --config-dir EAT/config \
    --config-name pretraining_BirdSet \
    common.user_dir=EAT \
    distributed_training.distributed_world_size=1 \
    dataset.batch_size=12 \
    task.data=/hpc_stor03/sjtu_home/wenxi.chen/mydata/audio/unbalanced_train \
    task.h5_format=True \
    checkpoint.restore_file=/hpc_stor03/sjtu_home/wenxi.chen/model_ckpt/test/checkpoint_last.pt \
    # checkpoint.save_dir=/hpc_stor03/sjtu_home/wenxi.chen/model_ckpt/test \