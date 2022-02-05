'''
This is an implementation of the split and merge approach to calculate svd. Proposed by Liang et al. 2016

Run with mpirun -np 4 python3 sam_svd.py
'''
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import numpy as np
import scipy.optimize
import time
import matplotlib.pyplot as plt
import sys
sys.path.append('./modules/')
from nek_snaps import dbCreator,dbReader,massMatrixReader
from plotters import contour2d,scatter

#Initialize mpi but only do stuff in rank zero
from mpi4py import MPI # equivalent to use MPI_init()

#==========================Main code==============================


#Get data from mpi
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD

#Control parameters
iSnap=1
iMode=15
nx=240
ny=160
nxy=nx*ny
n=nxy*2     #Number of rows in the snapshot matrix
m=95        #Number of columns in the snapshot matrix
nparts=size

#Define the send buffer as none to include in the scatter command
X = None
if rank == 0:
    #Generate the synthetic data only rank 0
    print('Reading/Generating data in rank 0')
    tic = time.perf_counter()
    X = np.loadtxt("data/U.txt").reshape(n, m)
    toc = time.perf_counter()
    print(f"Time to read the data at rank {rank}: {toc - tic:0.4f} seconds")
    #X=np.random.rand(n,m)
    tic = time.perf_counter()

# Create the local recive buffer in each MPI rank. 
Xi = np.empty((int(n/nparts),m))
#Scatter the data from rank 0 to the others scatter(snedbuf,recbuf,root)
comm.Scatter(X, Xi, root=0)

#In each rank flatten the data
#Xi=np.reshape(Xi,(1,int(n/nparts*m)))
offset = comm.Get_rank()*Xi.nbytes

#Open the file in parallel and write
amode = MPI.MODE_WRONLY|MPI.MODE_CREATE
fh = MPI.File.Open(comm, "data/Ubin.contig", amode)
fh.Write_at_all(offset, Xi)
fh.Close()

print('writen')
print(Xi)

# Create the local recive buffer in each MPI rank.
Xir = np.empty((int(n/nparts),m))
offset = comm.Get_rank()*Xir.nbytes

#Read the data
amode = MPI.MODE_RDONLY
fh = MPI.File.Open(comm, "data/Ubin.contig", amode)
fh.Read_at_all(offset, Xir)
fh.Close()

print('Read')
print(Xi-Xir)
