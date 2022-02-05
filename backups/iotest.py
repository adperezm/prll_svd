from mpi4py import MPI
import numpy as np

amode = MPI.MODE_WRONLY|MPI.MODE_CREATE
comm = MPI.COMM_WORLD
fh = MPI.File.Open(comm, "./datafile.contig", amode)

buffer = np.empty((10))
buffer[:] = comm.Get_rank()

print('Files to write')
print(buffer)

offset = comm.Get_rank()*buffer.nbytes
fh.Write_at_all(offset, buffer)

fh.Close()


amode = MPI.MODE_RDONLY
fh = MPI.File.Open(comm, "./datafile.contig", amode)

buffer_r = np.empty((10))


offset = comm.Get_rank()*buffer.nbytes
fh.Read_at_all(offset, buffer_r)

fh.Close()


print('Files read')
print(buffer_r)
