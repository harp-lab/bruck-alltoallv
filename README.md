# Optimizing the Bruck Algorithm for Non-uniform All-to-all Communication

MPI_Alltoallv is a generalization of MPI_Alltoall, supporting the exchange of non-uniform distributions of data. However, MPI_Alltoallv is typically implemented using only variants of the spread-out algorithm, and therefore misses out on the performance benefits that the log-time Bruck algorithm offers (especially for smaller data loads).

Therefore, we implement and empirically evaluate all existing variants of the Bruck algorithm for uniform and non-uniform data loads–this forms the basis for our own Bruck-based non-uniform all-to-all algorithms. In particular, we developed two implementations, padded Bruck and two-phase Bruck, that efficiently generalize Bruck algorithm to non-uniform all-to-all data exchange. 

## Padded Bruck 

# Citing 
Fan K, Gilray T, Pascucci V, Huang X, Micinski K, Kumar S. Optimizing the Bruck Algorithm for Non-uniform All-to-all Communication. InProceedings of the 31st International Symposium on High-Performance Parallel and Distributed Computing 2022 Jun 27 (pp. 172-184).

Building and installing
------------------------------------------

Building and installing bruck-alltoallv requires cmake 3.1+ and a current C++11-compatible Compiler. Clone bruck-alltoallv from github and proceed
as follows:

     $ git clone https://github.com/harp-lab/bruck-alltoallv.git
     $ cd bruck-alltoallv
     $ git checkout alltoallv
     $ mkdir build && cd build
     $ cmake ..
     $ make
     
     
Examples
------------------------------------------

The examples folder contains two examples: `uniform bruck algorithm example.cpp` and non `uniform bruck algorithm example.cpp`. The first is used to run uniform Bruck algorithm variants, and the second is used to run non-uniform Bruck algorithm variants.
