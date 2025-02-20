#!/bin/sh
#Set the job name (for your reference)
#PBS -N CV2
### Set the project name
#PBS -P col764.aib242289.course
### Request email when job begins and ends
### PBS -m bea
### Specify email address to use for notification.
###P BS -M aib242289@iitd.ac.in
### chunk specific resources ###(select=5:ncpus=4:mpiprocs=4:ngpus=2:mem=2GB::centos=skylake etc.)
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=01:00:00
#PBS -o outfile
#PBS -e errors

###PBS -l software=INTEL_PARALLEL_STUDIO

echo "==============================="
module load apps/pytorch/1.10.0/gpu/intelpython3.7
echo $PBS_JOBID
cd ~/AIL862/Assignment2
echo "in folder"
#job execution command
python assign2.py 
echo "compleeted"