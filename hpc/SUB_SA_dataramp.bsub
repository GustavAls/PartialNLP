#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J data_ramp_sst2[1-5]
### -- ask for number of cores (default: 1) --
#BSUB -n 4
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 12:00
# request 10GB of system-memory
#BSUB -R "rusage[mem=6GB]"
#BSUB -R "select[gpu16gb]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
###BSUB -u your_email_address
### -- send notification at start --
##BSUB -B
### -- send notification at completion--
##BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o scratch/s184400/SA_results/out/gpu_%J.out
#BSUB -e scratch/s184400/SA_results/out/gpu_%J.err
# -- end of LSF options --

# Activate the python eviorment
module load python3/3.9.11
source ~/PartialNLP/scratch/s184400/partial_venv/bin/activate
# Set python path
export PYTHONPATH=$PYTHONPATH:~/PartialNLP/
export PYTHONPATH=$PYTHONPATH:~/PartialNLP/scratch/s184400/cloned_repos/

nvidia-smi
# Load the cuda module
module load cuda/11.7
/appl/cuda/11.7.0/samples/bin/x86_64/linux/release/deviceQuery

echo "Running script..."
# Send sample job to GPU

##### ADD INFO HERE ####
dataset="sst2"

runs=(0 1 2 3 4)
run="${runs[$LSB_JOBINDEX - 1]}"

# Create some folders
dir_outpath="scratch/s184400/SA_results/data_ramp/${dataset}/run_${run}"
dir_datapath="scratch/s184400/SA_results/Init/${dataset}_dataset/run_${run}"


# Make the directories if they don't exist
mkdir -p $dir_outpath


# Run the training
python3 SentimentAnalysis/SentimentClassifier.py \
    --output_path $dir_outpath \
    --data_path $dir_datapath \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --device_batch_size 32 \
    --run $run \
    --num_epochs 1 \
    --dataset_name $dataset \
    --learning_rate 5e-05 \
    --no_cuda False \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --test_size 1 \
    --val_size 10 \
    --save_total_limit 1 \
    --load_best_model_at_end True \
    --metric_for_best_model "loss" \
    --dataramping True \


if [ $? -eq 0 ]; then
    echo "Success"
    exit 0
else
    echo "Error"
    exit 1
fi