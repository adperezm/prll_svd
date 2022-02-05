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
    # Gather all the columns of the modes into rank 0 to process

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


#Perform distributed SVD
U_local,Dy,Vty=distributed_svd(Xi,n,m)

#Gather the local modes into rank 0 
U = gathermodes(U_local,n,m)

# Perform the rest of the process in rank 0
if rank == 0: 
    toc = time.perf_counter()
    print(f"Time for Full SVD: {toc - tic:0.4f} seconds")

    #Now I can post process in rank 0. Load mesh and massmat
    x = np.loadtxt("data/x.txt").reshape(ny, nx)
    y = np.loadtxt("data/y.txt").reshape(ny, nx)
    bmsq = np.loadtxt("data/bmsq.txt").reshape(nxy, 1)

    print(f"Reconstruc to see acuracy of SVD")
    #Reconstruc for comparison
    Xr=U@np.diag(Dy)@Vty
    #Find and plot the differences
    A=X-Xr
    plt.figure(figsize=(7,5))
    ax = plt.subplot(1,1,1)
    p = ax.pcolormesh(A)
    plt.colorbar(p)
    plt.show()
  

    #Plot one of the modes
    phix=np.empty((nxy,1))
    phix[:,0]=U[0:nxy,iMode]

    for i in range (0,nxy):
        phix[i,0]=phix[i,0]/bmsq[i]

    contour2d(phix,x,y,title='Mode %d' %iMode)


    #Plot the reconstructed field to see
    ux=np.empty((nxy,1))
    ux[:,0]=Xr[0:nxy,iSnap]

    for i in range (0,nxy):
        ux[i,0]=ux[i,0]/bmsq[i]

    contour2d(ux,x,y,title='Reconstructed data with all modes at snapshot %d' %iSnap)




    
