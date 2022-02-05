'''
This is an implementation of the streaming split and merge aproach to calculate svd. Proposed by Liang et al. 2016

This implementation IS NOT OPTIMAL. Each update step needs to reallocate and append snapshots. Fix this to speed up the algorithm.

Run with mpirun -np 4 python3 stream_sam_svd.py
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
from tqdm import tqdm

#Initialize mpi but only do stuff in rank zero
from mpi4py import MPI #equivalent to use MPI_init()

#=============================Functions==============================

def distributed_svd(Xi,n,m):
    #Take the partiotioned data Xi in each rank and do the SVD.
    #The outputs are:
        # A partitioned mode matrix U
        # The eigen values D
        # The right orthogonal matrix trasposed Vt


    #Perfrom Svd in all ranks
    tic_in = time.perf_counter()
    Ui,Di,Vti=np.linalg.svd(Xi, full_matrices=False)
    toc_in = time.perf_counter()
    Yi=np.diag(Di)@Vti
    #print(f"Time for SVD of Xi in rank {rank}: {toc_in - tic_in:0.4f} seconds")

    #Gather Yi into Y in rank 0
    #prepare the buffer for recieving
    Y = None
    if rank == 0:
        #Generate the buffer to gather in rank 0
        Y = np.empty((m*nparts,m))
    comm.Gather(Yi, Y, root=0)

    if rank == 0:
        #If tank is zero, calculate the svd of the combined eigen matrix
        #Perform the svd of the combined eigen matrix
        tic_in = time.perf_counter()
        Uy,Dy,Vty=np.linalg.svd(Y, full_matrices=False)
        toc_in = time.perf_counter()
        #print(f"Time for SVD of Y in rank {rank}: {toc_in - tic_in:0.4f} seconds")
    else:
        #If the rank is not zero, simply create a buffer to recieve the Uy Dy and Vty
        Uy  = np.empty((m*nparts,m))
        Dy  = np.empty((m))
        Vty = np.empty((m,m))
    comm.Bcast(Uy, root=0)
    comm.Bcast(Dy, root=0)
    comm.Bcast(Vty, root=0)
    #Now matrix multiply each Ui by the corresponding entries in Uy
    U_local=Ui@Uy[rank*m:(rank+1)*m,:]

    return U_local, Dy, Vty

def gathermodes(U_1t, n, m):
    U = None #prepare the buffer for recieving
    if rank == 0:
        #Generate the buffer to gather in rank 0
        U = np.empty((n,m))
    comm.Gather(U_1t, U, root=0)
    return U

#==========================Main code==============================


#Get data from mpi
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD

#Control parameters
n=512*1000
m=50
nparts=size


#Define the send buffer as none to include in the scatter command

if rank == 0:
    #Generate the synthetic data only rank 0
    X=np.random.rand(n,m)
    pbar= tqdm(total=m)
    tic = time.perf_counter()


# Create the local recive buffer in each MPI rank. 
Xi = np.empty((int(n/nparts),1))

#Start the streaming process
for j in range(0,m):
    if rank == 0:
        pbar.update(1)
    #Scatter the data from rank 0 to the others scatter(snedbuf,recbuf,root)
    snapshot = None #Send place holder
    if rank == 0:
        snapshot=np.zeros((n,1))
        snapshot[:,0]=np.copy(X[:,j]) #Assing send buffer in rank 0
    comm.Scatter(snapshot, Xi, root=0)

    if j==0:
        #Perform the distributed SVD and don't accumulate
        U_1t,D_1t,Vt_1t=distributed_svd(Xi,n,j+1)
    else:
        #Find the svd of the new snapshot
        U_tp1,D_tp1,Vt_tp1=distributed_svd(Xi,n,1)
        #2 contruct matrices to Do the updating
        V_tilde=scipy.linalg.block_diag(Vt_1t.T,Vt_tp1.T)
        W=np.append(U_1t@np.diag(D_1t),U_tp1@np.diag(D_tp1),axis=1)
        Uw,Dw,Vtw=distributed_svd(W,n,j+1)
        #3 Update
        U_1t=Uw
        D_1t=Dw
        Vt_1t=(V_tilde@Vtw.T).T

#============= Streaming is done, gather the modes and evaluate========#

U = gathermodes(U_1t,n,m)

####################

if rank == 0: 
    toc = time.perf_counter()
    print(f"Time for Full SVD: {toc - tic:0.4f} seconds")
    
    print("Reconstruc to test acuracy")
    Xr=U@np.diag(D_1t)@Vt_1t
    #print(Xr)
    A=X-Xr
    plt.figure(figsize=(7,5))
    ax = plt.subplot(1,1,1)
    p = ax.pcolormesh(A)
    plt.colorbar(p)
    plt.show()


    
