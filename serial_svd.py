'''
Serial Svd code. It multithreads by default. If you want to see the real serial performance, uncomment the os.x commands below.

Run with python3 serial_svd.py
'''
import os
#os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
#os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
#os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
#os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
#os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6
import numpy as np
import scipy.optimize
import time
import matplotlib.pyplot as plt


n=512*1000
m=100
X=np.random.rand(m,n)

tic = time.perf_counter()
Ur,Dr,Vtr=np.linalg.svd(X, full_matrices=False)
toc = time.perf_counter()
print(f"Time for full SVD {toc - tic:0.4f} seconds")

#print('Reconstruc to check accuracy')
##Reconstruct with the new svd
#Xr=Ur@np.diag(Dr)@Vtr
##Find and plot the differences
#A=X-Xr
#plt.figure(figsize=(7,5))
#ax = plt.subplot(1,1,1)
#p = ax.pcolormesh(A)
#plt.colorbar(p)
#plt.show()
