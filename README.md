# cudascan
Scan an array both inc/exc with CUDA 

This code is able to scan an array of size n = 2 ^ M where M can be from 2 to 29!
both inclusive and exclusive scan have been implemented.
the default mode is inclusive.
Blelloch algorithm (work-efficient) scan has been used.

#### compile : nvcc scan2_to_29.cu scan2_main.cu -o scan2
#### run : ./scan2 M
