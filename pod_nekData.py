
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import matplotlib.pylab as pylab
params = {'legend.fontsize': 15,
          'legend.loc':'best',
          'figure.figsize': (15, 5),
         'lines.markerfacecolor':'none',
         'axes.labelsize': 17,
         'axes.titlesize': 17,
         'xtick.labelsize':15,
         'ytick.labelsize':15,
         'grid.alpha':0.6}
pylab.rcParams.update(params)
#
from pymech import dataset
import pickle
#local modules
sys.path.append('./modules/')
from nek_snaps import dbCreator,dbReader,massMatrixReader
from plotters import contour2d,scatter

iMod=0
iSnap=1
maxMode=94



#------- SETTINGS -------------------------------------
#Path to the *.f files created by Nek5000
#Change path_ to where the data are downloaded.
path_="/mnt/c/Users/adalb/Desktop/2simEx_wksh-master/data/mixlay_snapshots/"

caseName='mixlay'  #Nek case name

doPickle=False  #if True, a pickle database from Nek data is created
if doPickle:
   qoiName=['temperature','pressure','ux','uy']   #list of QoIs whose data to be read
   #range of snapshots to be included in the DB
   start_=1   #ID of the starting file, e.g. 5 -> caseName0.f00005
   end_=95     #ID of the end file
#-----------------------------------------------------

#Create a pickle database or read an already existing pickle database
if doPickle:
   info={'dataPath':path_,
         'caseName':caseName,
         'startID':start_,
         'endID':end_,
         'qoiName':qoiName}
   dbQoI=dbCreator(info)
else:    
   info={'pickleFile':'/mnt/c/Users/adalb/Desktop/simEx_wksh-master/data/mixlay_snapshots/mixlay_1to95'} 
   dbQoI=dbReader(info)


# Let's see what keys and values are available in the `dbQoI`:


print('Keys in QoI-db: ',dbQoI.keys())
print('Keys in db of a QoI',dbQoI['temperature'].keys())
print('Starting and end file included the db: %s to %s.' %(dbQoI['temperature']['startFile'],dbQoI['temperature']['endFile']))


# ## Extract the required data from the database

# Read the mass matrix $\mathbf{M}$ associated to the grid points of the Nek5000 simulation. Note that we assume the mass matrix reamined unchanged in time. 

#M: mass matrix of Nek5000 simulation
print('Keys in db of a QoI',dbQoI['massMat'].keys())
bm1=dbQoI['massMat']['val']
bmsq=np.sqrt(bm1).flatten('F')

print(bm1.shape)
print(bmsq.shape)


# Now we should grab the snapshot data of the QoI from the database. The QoI name is set via `qoiName`. If the QoI is a vector, like velocity, the snapshots of the associated components are read separately and then get concatenated. 

#QoI(s) for which POD is to be constructed
qoiName='velocity'   #'temperature', 'pressure', 'velocity'


if qoiName=='temperature' or qoiName=='pressure':   #scalar QoI
   #grab the snapshots 
   db=dbQoI[qoiName]
   U=db['snapDB']
   nx,ny,m=U.shape
   #rehshape snapshot matrix to (nx*ny,m)
   U=np.reshape(U,(nx*ny,m),order='F')   #snapshot data       
   #multiply snapshots by M^1/2
   for i in range(m): U[:,i]*=bmsq       
        
elif qoiName=='velocity':   #velocity vector
   #grab the snapshots 
   db =dbQoI['ux']       
   nx,ny,m=dbQoI['ux']['snapDB'].shape
   #rehshape snapshot matrix to (nx*ny,m)
   ux=np.reshape(dbQoI['ux']['snapDB'],(nx*ny,m),order='F')
   uy=np.reshape(dbQoI['uy']['snapDB'],(nx*ny,m),order='F')   
   #multiply snapshots by M^1/2
   for i in range(m): ux[:,i]*=bmsq
   for i in range(m): uy[:,i]*=bmsq
   #concatenate samples of velocity components: U=[ux|uy] 
   U=np.concatenate([ux,uy],axis=0)   
    
print(bmsq)
np.savetxt("data/bmsq.txt", bmsq)

print('shape of U: (nx*ny,nSnap)=',U.shape)    
a_file = open("data/U.txt", "w")
for row in U:
    np.savetxt(a_file, row)
a_file.close()



# Let's get the coordinates of the spatial points for contour plots in the post-processing step. 

x=db['x']        #x-ccordinate
y=db['y']        #y coordinate
n=nx*ny          #total number of spatial points    
print(x.shape)

a_file = open("data/x.txt", "w")
for row in x:
    np.savetxt(a_file, row)
a_file.close()
print(U.shape)


a_file = open("data/y.txt", "w")
for row in y:
    np.savetxt(a_file, row)
a_file.close()
print(U.shape)

#X, S, WT = np.linalg.svd(U[:,0:5],full_matrices=False)
X, S, WT = np.linalg.svd(U,full_matrices=False)


print(X.shape)
print(np.diag(S).shape)
print(WT.shape)

# 2. Plot/print the SVD spectral values and compare them to the eigenvalues in Method 1.

#The sigma values contain the energy and are also the square of the eigen values.
Seg=S
plt.figure(figsize=(15,4))
plt.semilogy((Seg)/(sum(Seg)),'-ob',label=r'$\lambda_k/\sum_{i=1}^m{\lambda_i}$')
#plt.semilogy((Lam1)/sum(Lam1),'-ob',label=r'$\lambda_k/\sum_{i=1}^m{\lambda_i}$')
#plt.semilogx(np.cumsum(Lam1)/sum(Lam1),'o-b',label='$\sum_{i=1}^k\lambda_i/\sum_{i=1}^m{\lambda_i}$')
plt.xlabel(r'$k$')
plt.legend(loc='best')
plt.grid()
plt.show()


# 3. Plot an arbitrary temporal coefficient and compute its norm. Also check orhogonality of two arbitrary coefficients.

#Get all the temporal coefficients. They are row vectors.
T=np.matmul(np.diag(S),WT)
print(T.shape)

#If you get the norm of each of them, then you should get the values of S, as what you get is the energy
tt=np.zeros([95,1])
for i in range (0,95):
    tt[i]=np.linalg.norm(T[i,:])


plt.figure(figsize=(15,4))
plt.semilogy((tt)/(sum(tt)),'-ob',label=r'$\lambda_k/\sum_{i=1}^m{\lambda_i}$')
plt.semilogy((Seg)/(sum(Seg)),'-xr',label=r'$\lambda_k/\sum_{i=1}^m{\lambda_i}$')
#plt.semilogy((Lam1)/sum(Lam1),'-ob',label=r'$\lambda_k/\sum_{i=1}^m{\lambda_i}$')
#plt.semilogx(np.cumsum(Lam1)/sum(Lam1),'o-b',label='$\sum_{i=1}^k\lambda_i/\sum_{i=1}^m{\lambda_i}$')
plt.xlabel(r'$k$')
plt.legend(loc='best')
plt.grid()
plt.show()


# 4. Plot contours of an arbitrary mode (basis) in the x-y plane  .

print(X[:n,iMod].shape)
print(bmsq.shape)
contour2d(X[:n,iMod]/bmsq,x,y,title=r'Contours of $\mathbf{\varphi}_%d (x,y)$ of ux' %(iMod))
contour2d(X[n:,iMod]/bmsq,x,y,title=r'Contours of $\mathbf{\varphi}_%d (x,y)$ of uy' %(iMod))   


# 5. Reconstruct the POD considering a user-defined number of modes is included in the expansion.


imodes=maxMode+1
print(X.shape)

uRec1=np.matmul(X[:,0:imodes],T[0:imodes,:])
print(uRec1.shape)

#divide by M^1/2
if qoiName=='temperature' or qoiName=='pressure':
   for i in range(m): uRec1[:,i]/=bmsq
elif qoiName=='velocity':
   for i in range(m): 
       uRec1[:n,i]/=bmsq    
       uRec1[n:,i]/=bmsq


#Observed snapshot 
if qoiName=='temperature' or qoiName=='pressure':
   Uex=U[:,iSnap]/bmsq
   uRec1_=uRec1[:,iSnap]
elif qoiName=='velocity':
   ##ux
   uRec1_=uRec1[:n,iSnap]
   Uex=U[:n,iSnap]/bmsq

    
#plot    
contour2d(Uex,x,y,title='Observed data at snapshot %d' %iSnap)

title_='POD reconstruction at snapshot %d, first %d modes (out of %d)' %(iSnap,maxMode,m)
contour2d(uRec1[:n,iSnap],x,y,title=title_)


# 6. Compare the POD reconstructed field to the associated snapshot data. How does the accuracy vary with increasing the number of POD modes used in reconstruction?

#scatter plot
scatter(uRec1[:n,iSnap],Uex,xlab='POD Reconstruction',ylab='Observation')


