# tidal_heating
# Repository to analyses of GW events using waveform with tidal heating effects

Different Config files and instructions to run workflow on Sarathi Clusters


Environment:

Run the following commands after sourcing your virtual environment

export LD_LIBRARY_PATH=/soft/condor_mpi/lib:$LD_LIBRARY_PATH
export PATH=/soft/condor_mpi/bin:$PATH
export MKL_NUM_THREADS="1"
export MKL_DYNAMIC="FALSE"
export OMP_NUM_THREADS=1
export MPI_PER_NODE=16
