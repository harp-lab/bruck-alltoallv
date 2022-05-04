/*
 * non_uniform_bruck.h
 *
 *      Author: kokofan
 */

#ifndef SRC_NON_UNIFORM_BRUCK_H_
#define SRC_NON_UNIFORM_BRUCK_H_

#include "../brucks.h"

void padded_bruck_non_uniform_benchmark(int dist, int range, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);
void padded_alltoall_non_uniform_benchmark(int dist, int range, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);
void datatype_bruck_non_uniform_benchmark(int range, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);
void modified_dt_bruck_non_uniform_benchmark(int range, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);
void zeroCopy_bruck_non_uniform_benchmark(int range, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);
void twophase_non_uniform_benchmark(int dist, int range, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);
void ptp_non_uniform_benchmark(char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);
void new_twophase_non_uniform_benchmark(int dist, int range, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

#endif /* SRC_NON_UNIFORM_BRUCK_H_ */
