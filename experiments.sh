#!/bin/bash
#SBATCH --account=rrg-bengioy-ad            # Yoshua pays for your job
#SBATCH --cpus-per-task=8               # Ask for 6 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=08:00:00                   # The job will run for 48 hours
#SBATCH --mail-user=sharathraparthy@gmail.com
# SBATCH --array=1-6
#SBATCH --mail-type=ALL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH -o /scratch/raparths/dqn-%j.out  # Write the log in $SCRATCH


source $HOME/.bashrc
source $HOME/reward/bin/activate

# wandb login 1ec9f8d0c00287d5ea7ef9a83ed6a0783d0d0fe3
python examples.py

