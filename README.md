# prll_svd

# Before running
First run python3 pod_nekdata.py in order to write some data that is needed.

# How to run
All the algorithms that have "sam" in their name stand for SPLIT AND MERGE. They are MPI programs that should be run, for example, with:

mpi -np 4 python3 sam_svd.py
