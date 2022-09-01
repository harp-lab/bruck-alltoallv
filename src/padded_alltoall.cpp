/*
 * padded_alltoall.cpp
 *
 *      Author: kokofan
 */

#include "non_uniform_bruck.h"

/// Padded MPI_alltoall algorithm
void padded_alltoall(char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm){

	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	int typesize;
	MPI_Type_size(sendtype, &typesize);

	// 1. Find max send count
	int local_max_count = 0;
	for (int i = 0; i < nprocs; i++) {
		if (sendcounts[i] > local_max_count)
			local_max_count = sendcounts[i];
	}
	int max_send_count = 0;
	MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);

	// 2. local padding
	char* temp_send_buffer = (char*)malloc(max_send_count*nprocs*typesize);
	for (int i = 0; i < nprocs; i++)
		memcpy(&temp_send_buffer[i*max_send_count*typesize], &sendbuf[sdispls[i]*typesize], sendcounts[i]*typesize);

	// 3. all-to-all communication
	char* temp_recv_buffer = (char*)malloc(max_send_count*nprocs*typesize);
	MPI_Alltoall(temp_send_buffer, max_send_count*typesize, MPI_CHAR, temp_recv_buffer, max_send_count*typesize, MPI_CHAR, comm);
	free(temp_send_buffer);

	// 4. filter
	int offset = 0;
	for (int i = 0; i < nprocs; i++)
		memcpy(&recvbuf[rdispls[i]*typesize], &temp_recv_buffer[i*max_send_count*typesize], recvcounts[i]*typesize);
	free(temp_recv_buffer);

}

