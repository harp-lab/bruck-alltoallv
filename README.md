# bruck-alltoallv

MPI_Alltoallv is a generalization of MPI_Alltoall, supporting the exchange of non-uniform distributions of data. However, MPI_Alltoallv is typically implemented using only variants of the spread-out algorithm, and therefore misses out on the performance benefits that the log-time Bruck algorithm offers (especially for smaller data loads).


Therefore, we implement and empirically evaluate all existing variants of the Bruck algorithm for uniform and non-uniform data loads–this forms the basis for our own Bruck-based non-uniform all-to-all algorithms. In particular, we developed two implementations, padded Bruck and two-phase Bruck, that efficiently generalize Bruck algorithm to non-uniform all-to-all data exchange. 

Algorithms include:

* Basic niform Bruck algorithm without MPI-derived datatypes
* Basic uniform Bruck algorithm with MPI-derived datatypes
* Modified uniform Bruck algorithm without MPI-derived datatypes (no final rotation step)
* Modified uniform Bruck algorithm with MPI-derived datatypes (no final rotation step)
* No Rotation uniform Bruck algorithm (no both rotation steps)
* Zero Copy uniform Bruck algorithm (no internal copy steps)
* No Rotation and Copy uniform Bruck algorithm (no internal copy and rotation steps)
* Padded non_uniform Bruck algorithm without MPI-derived datatypes
* Padded MPI_alltoall algorithm
* Basic non-uniform Bruck algorithm with MPI-derived datatypes
* Modified non-uniform Bruck algorithm with MPI-derived datatypes (no final rotation step)
* Zero Copy non-uniform Bruck algorithm (no internal copy steps)
* Two-phase non-uniform Bruck algorithm 
* Spead-out algorithm (using non-blocking point-to-point communication)

Building and installing
------------------------------------------

Building and installing bruck-alltoallv requires cmake 3.1+ and a current C++11-compatible Compiler. Clone bruck-alltoallv from github and proceed
as follows:

     $ git clone https://github.com/harp-lab/bruck-alltoallv.git
     $ cd bruck-alltoallv
     $ mkdir build && cd build
     $ cmake ..
     $ make
     
     
Examples
------------------------------------------

The examples folder contains two examples: `uniform bruck algorithm example.cpp` and non `uniform bruck algorithm example.cpp`. The first is used to run uniform Bruck algorithm variants, and the second is used to run non-uniform Bruck algorithm variants.
