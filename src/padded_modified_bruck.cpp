/*
 * padded_modified_bruck.cpp
 *
 *      Author: kokofan
 */

#include "non_uniform_bruck.h"

/// padded modified bruck with MPI datatype
void padded_modified_dt_bruck(char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) {

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

	// 2. local rotation
	char* temp_send_buffer = (char*)malloc(max_send_count*nprocs*typesize);
	char* temp_recv_buffer = (char*)malloc(max_send_count*typesize*nprocs);
	memset(temp_send_buffer, 0, max_send_count*nprocs*typesize);
	int offset = 0;
	for (int i = 0; i < nprocs; i++) {
		int index = (2*rank-i+nprocs)%nprocs;
		memcpy(&temp_send_buffer[index*max_send_count*typesize], &sendbuf[offset], sendcounts[i]*typesize);
		offset += sendcounts[i]*typesize;
	}

    // 3. exchange data with log(P) steps

    long long unit_size = max_send_count * typesize;
 	for (int k = 1; k < nprocs; k <<= 1) {
 		// 1) create data type
		int displs[(nprocs+1)/2];
		int sendb_num = 0;
		for (int i = 1; i < nprocs; i++) {
			if (i & k)
				displs[sendb_num++] = ((rank+i)%nprocs)*unit_size;
		}
		MPI_Datatype send_type;
		MPI_Type_create_indexed_block(sendb_num, unit_size, displs, MPI_CHAR, &send_type);
		MPI_Type_commit(&send_type);

		// 2) exchange data
		int recv_proc = (rank + k) % nprocs; // receive data from rank + 2^k process
		int send_proc = (rank - k + nprocs) % nprocs; // send data from rank - 2^k process
		MPI_Sendrecv(temp_send_buffer, 1, send_type, send_proc, 0, temp_recv_buffer, 1, send_type, recv_proc, 0, comm, MPI_STATUS_IGNORE);
		MPI_Type_free(&send_type);

		// 3) copy time
		for (int i = 0; i < sendb_num; i++)
			memcpy(temp_send_buffer+displs[i], temp_recv_buffer+displs[i], unit_size);
 	}
 	free(temp_recv_buffer);

	// 4. remove padding
	for (int i=0; i < nprocs; i++)
		memcpy(&recvbuf[rdispls[i]*typesize], &temp_send_buffer[i*max_send_count*typesize], recvcounts[i]*typesize);
 	free(temp_send_buffer);
}





