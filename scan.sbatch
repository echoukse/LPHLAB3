#!/bin/bash
#SBATCH -J scan           # job name
#SBATCH -o scan.o%j       # output and error file name (%j expands to jobID)
#SBATCH -n 1              # total number of mpi tasks requested
#SBATCH -p normal-mic     # queue (partition) -- normal, development, etc.
#SBATCH -t 01:30:00        # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=eshachoukse.iitkgp@gmail.com
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH -A EE382N-20
export OFFLOAD_REPORT=2
export OMP_NUM_THREADS=240 
export KMP_PLACE_THREADS=60c,4t
./checker.pl scan             # run the MPI executable named a.out
#./checker.pl find_repeats             # run the MPI executable named a.out
