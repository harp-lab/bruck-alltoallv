/*
 * padded_bruck.cpp
 *
 *      Author: kokofan
 */

#include "non_uniform_bruck.h"

// Padded bruck algorithm
void padded_bruck_alltoallv(char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) {

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
	memset(temp_send_buffer, 0, max_send_count*nprocs*typesize);
	int offset = 0;
	for (int i = 0; i < nprocs; i++) {
		int index = (i - rank + nprocs) % nprocs;
		memcpy(&temp_send_buffer[index*max_send_count*typesize], &sendbuf[offset], sendcounts[i]*typesize);
		offset += sendcounts[i]*typesize;
	}

	// 3. exchange data with log(P) steps
	long long unit_size = max_send_count * typesize;
	char* temp_buffer = (char*)malloc(max_send_count*typesize*((nprocs+1)/2));
	char* temp_recv_buffer = (char*)malloc(max_send_count*typesize*((nprocs+1)/2));
	for (int k = 1; k < nprocs; k <<= 1) {
		// 1) find which data blocks to send
		int send_indexes[(nprocs+1)/2];
		int sendb_num = 0;
		for (int i = k; i < nprocs; i++) {
			if (i & k)
				send_indexes[sendb_num++] = i;
		}

		// 2) copy blocks which need to be sent at this step
		for (int i = 0; i < sendb_num; i++) {
			long long offset = send_indexes[i] * unit_size;
			memcpy(temp_buffer+(i*unit_size), temp_send_buffer+offset, unit_size);
		}

		// 3) send and receive
		int recv_proc = (rank - k + nprocs) % nprocs; // receive data from rank - 2^step process
		int send_proc = (rank + k) % nprocs; // send data from rank + 2^k process
		long long comm_size = sendb_num * unit_size;
		MPI_Sendrecv(temp_buffer, comm_size, MPI_CHAR, send_proc, 0, temp_recv_buffer, comm_size, MPI_CHAR, recv_proc, 0, comm, MPI_STATUS_IGNORE);

		// 4) replace with received data
		for (int i = 0; i < sendb_num; i++) {
			long long offset = send_indexes[i] * unit_size;
			memcpy(temp_send_buffer+offset, temp_recv_buffer+(i*unit_size), unit_size);
		}
	}
	free(temp_buffer);
	free(temp_recv_buffer);

	// 4. second rotation
	offset = 0;
	for (int i = 0; i < nprocs; i++) {
		int index = (rank - i + nprocs) % nprocs;
		memcpy(&recvbuf[rdispls[index]*typesize], &temp_send_buffer[i*unit_size], recvcounts[index]*typesize);
	}
	free(temp_send_buffer);
}


/// padded bruck algorithm with MPI_datatype
void padded_bruck_alltoallv_dt(char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) {

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
		int index = (i - rank + nprocs) % nprocs;
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
				displs[sendb_num++] = i*unit_size;
		}
		MPI_Datatype send_type;
		MPI_Type_create_indexed_block(sendb_num, unit_size, displs, MPI_CHAR, &send_type);
		MPI_Type_commit(&send_type);

		// 2) exchange data
		int recv_proc = (rank - k + nprocs) % nprocs; // receive data from rank - 2^step process
		int send_proc = (rank + k) % nprocs; // send data from rank + 2^k process
		MPI_Sendrecv(temp_send_buffer, 1, send_type, send_proc, 0, temp_recv_buffer, 1, send_type, recv_proc, 0, comm, MPI_STATUS_IGNORE);
		MPI_Type_free(&send_type);

		// 3) replace time
		for (int i = 0; i < sendb_num; i++)
			memcpy(temp_send_buffer+displs[i], temp_recv_buffer+displs[i], unit_size);
	}
	free(temp_recv_buffer);

	// 4. second rotation
	double revs_rotation_start = MPI_Wtime();
	offset = 0;
	for (int i = 0; i < nprocs; i++) {
		int index = (rank - i + nprocs) % nprocs;
		memcpy(&recvbuf[rdispls[index]*typesize], &temp_send_buffer[i*unit_size], recvcounts[index]*typesize);
	}
	free(temp_send_buffer);
}

