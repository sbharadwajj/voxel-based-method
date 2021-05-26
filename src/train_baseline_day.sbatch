#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=TutorialJob

#resources:

#SBATCH --cpus-per-task=8
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=test
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=3G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=00:15:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=job_files/sanity-check.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=job_files/sanity-check.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=***@uni-tuebingen.de
# your mail address

####
#b) copy all needed data to the jobs scratch folder
####

####
#c) Execute your tensorflow code in a specific singularity container
#d) Write your checkpoints to your home directory, so that you still have them if your job fails
#cnn_minst.py <model save path> <mnist data path>
####

singularity exec /home/bharadwaj/envi/torch14-cuda10-ConvOccu.simg python train_baseline.py /home/bharadwaj/implementations/convONw-baseline2/baseline.yaml
# singularity exec /home/bharadwaj/envi/test2.simg make
# singularity exec /home/bharadwaj/envi/test2.simg python2.7 /home/bharadwaj/implementations/PointSetGeneration/depthestimate/train_nn.py [data=/home/bharadwaj/implementations/PointSetGeneration/data/] [dump=dump/] [num=3] predict

echo DONE!
