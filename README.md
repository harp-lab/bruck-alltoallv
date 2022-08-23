# Optimizing the Bruck Algorithm for Non-uniform All-to-all Communication

MPI_Alltoallv is a generalization of MPI_Alltoall, supporting the exchange of non-uniform distributions of data. However, MPI_Alltoallv is typically implemented using only variants of the spread-out algorithm, and therefore misses out on the performance benefits that the log-time Bruck algorithm offers (especially for smaller data loads).

Therefore, we implement and empirically evaluate all existing variants of the Bruck algorithm for uniform and non-uniform data loads–this forms the basis for our own Bruck-based non-uniform all-to-all algorithms. In particular, we developed two implementations, padded Bruck and two-phase Bruck, that efficiently generalize Bruck algorithm to non-uniform all-to-all data exchange. 

## Padded Bruck 

Padded Bruck converts a non-uniform all-to-all problem into a uni- form all-to-all problem through padding—a natural extension. There are three main phases: (a) padding all non-uniform buffers to a fixed-sized buffer, (b) invoking Bruck-style communication for the uniform buffers, and (c) scanning the received buffers to extract the actual data.

<img src="https://github.com/harp-lab/bruck-alltoallv/blob/main/figs/padded_bruck.png" width="800"/>

## Two-phase Bruck

The Two-phase Bruck algorithm performs a coupled two-phase data exchange (for all log(𝑃) communication steps) and by using a large monolithic buffer. The two-phase communication involves a meta-data exchange followed by actual data transfer, where the meta-data prepares processes for the actual data exchange. The monolithic working buffer facilitates seamless intermediate data exchanges, pre-allocated to an upper bound on overflow data. The approach requires more space in the transfer phases to optimize communication time.

<img src="https://github.com/harp-lab/bruck-alltoallv/blob/main/figs/two_phase_alg.png" width="1000"/>

## Input Parameters 

The same with MPI_Alltoallv.

```
sendbuf: starting address of send buffer (char*)
sendcounts: integer array equal to the group size specifying the number of elements to send to each processor
sdispls: integer array (of length group size). Entry j specifies the displacement (relative to sendbuf from which to take the outgoing data destined for process j
sendtype: data type of send buffer elements (handle)
recvcounts: integer array equal to the group size specifying the maximum number of elements that can be received from each processor
rdispls: integer array (of length group size). Entry i specifies the displacement (relative to recvbuf at which to place the incoming data from process i
recvtype: data type of receive buffer elements (handle)
comm: communicator (handle)
```

## Building and installing

Building and installing bruck-alltoallv requires cmake 3.1+ and a current C++11-compatible Compiler. Clone bruck-alltoallv from github and proceed
as follows:

     $ git clone https://github.com/harp-lab/bruck-alltoallv.git
     $ cd bruck-alltoallv
     $ git checkout alltoallv
     $ mkdir build && cd build
     $ cmake ..
     $ make

## Examples

We conduct a thorough evaluation of our algorithms using synthetic microbenchmarks on the Theta Supercomputer [4] of Argonne National Lab (ANL). 

     $ cd examples
     $ mpirun -n <nprocs> ./nubruck

# Citing 
```
Fan K, Gilray T, Pascucci V, Huang X, Micinski K, Kumar S. Optimizing the Bruck Algorithm for Non-uniform All-to-all Communication. InProceedings of the 31st International Symposium on High-Performance Parallel and Distributed Computing 2022 Jun 27 (pp. 172-184).
```
     
     

