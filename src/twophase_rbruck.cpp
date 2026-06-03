/*
 * twophase_bruck.cpp
 *
 *      Author: kokofan
 */

#include "non_uniform_bruck.h"

void twophase_bruck_alltoallv(char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) {

	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	int typesize;
	MPI_Type_size(sendtype, &typesize);

	// 1. Find max send count
	int local_max_count = 0;
	int max_send_count = 0;

	int csendcounts[nprocs];
	for (int i = 0; i < nprocs; i++) {
		csendcounts[i] = sendcounts[i];
		if (sendcounts[i] > local_max_count)
			local_max_count = sendcounts[i];
	}
	MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);

	// 2. create local index array after rotation
	int rotate_index_array[nprocs];
    for (int i = 0; i < nprocs; i++)
    	rotate_index_array[i] = (2*rank-i+nprocs)%nprocs;

	// 3. exchange data with log(P) steps
	int max_send_elements = (nprocs+1)/2;
	char* extra_buffer = (char*) malloc(max_send_count*typesize*nprocs);
	char* temp_send_buffer = (char*) malloc(max_send_count*typesize*max_send_elements);
	char* temp_recv_buffer = (char*) malloc(max_send_count*typesize*max_send_elements);
	int pos_status[nprocs];
	memset(pos_status, 0, nprocs*sizeof(int));

	int ind = 0;
	for (int k = 1; k < nprocs; k <<= 1) {
    	// 1) find which data blocks to send
    	int send_indexes[max_send_elements];
    	int sendb_num = 0;
		for (int i = 1; i < nprocs; i++) {
			if (i & k)
				send_indexes[sendb_num++] = (rank+i)%nprocs;
		}

		// 2) prepare metadata and send buffer
		int metadata_send[sendb_num+1];
		int offset = 0;
		for (int i = 0; i < sendb_num; i++) {
			int send_index = rotate_index_array[send_indexes[i]];
			metadata_send[i] = csendcounts[send_index];
			if (pos_status[send_index] == 0)
				memcpy(&temp_send_buffer[offset], &sendbuf[sdispls[send_index]*typesize], csendcounts[send_index]*typesize);
			else
				memcpy(&temp_send_buffer[offset], &extra_buffer[send_indexes[i]*max_send_count*typesize], csendcounts[send_index]*typesize);
			offset += csendcounts[send_index]*typesize;
		}

		// 3) exchange metadata
		int sendrank = (rank - k + nprocs) % nprocs;
		int recvrank = (rank + k) % nprocs;
		int metadata_recv[sendb_num];
		MPI_Sendrecv(metadata_send, sendb_num, MPI_INT, sendrank, 0, metadata_recv, sendb_num, MPI_INT, recvrank, 0, comm, MPI_STATUS_IGNORE);

		// 4) exchange data
		int sendCount = 0;
		for (int i = 0; i < sendb_num; i++)
			sendCount += metadata_recv[i];
		MPI_Sendrecv(temp_send_buffer, offset, MPI_CHAR, sendrank, 1, temp_recv_buffer, sendCount*typesize, MPI_CHAR, recvrank, 1, comm, MPI_STATUS_IGNORE);

		// 5) replace
		offset = 0;
		for (int i = 0; i < sendb_num; i++) {
			int send_index = rotate_index_array[send_indexes[i]];
			memcpy(&extra_buffer[send_indexes[i]*max_send_count*typesize], &temp_recv_buffer[offset], metadata_recv[i]*typesize);
			offset += metadata_recv[i]*typesize;
			pos_status[send_index] = 1;
			csendcounts[send_index] = metadata_recv[i];
		}
		ind++;
	}
	free(temp_send_buffer);
	free(temp_recv_buffer);

	for (int i = 0; i < nprocs; i++) {
		if (rank == i)
			memcpy(&recvbuf[rdispls[i]*typesize], &sendbuf[sdispls[i]*typesize], recvcounts[i]*typesize);
		else
			memcpy(&recvbuf[rdispls[i]*typesize], &extra_buffer[i*max_send_count*typesize], recvcounts[i]*typesize);
	}
	free(extra_buffer);
}


void twophase_bruck_alltoallv_new(char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) {

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

	// 2. create local index array after rotation
	int rotate_index_array[nprocs];
	for (int i = 0; i < nprocs; i++)
		rotate_index_array[i] = (2*rank-i+nprocs)%nprocs;

	// 3. exchange data with log(P) steps
	int max_send_elements = (nprocs+1)/2;
	char* extra_buffer = (char*) malloc(max_send_count*typesize*nprocs);
	char* temp_send_buffer = (char*) malloc(max_send_count*typesize*max_send_elements);
	char* temp_recv_buffer = (char*) malloc(max_send_count*typesize*max_send_elements);
	int pos_status[nprocs];
	memset(pos_status, 0, nprocs*sizeof(int));
	memcpy(&recvbuf[rdispls[rank]*typesize], &sendbuf[sdispls[rank]*typesize], recvcounts[rank]*typesize);

	for (int k = 1; k < nprocs; k <<= 1) {
		// 1) find which data blocks to send
		int send_indexes[max_send_elements];
		int sendb_num = 0;
		for (int i = k; i < nprocs; i++) {
			if (i & k)
				send_indexes[sendb_num++] = (rank+i)%nprocs;
		}

		// 2) prepare metadata and send buffer
		int metadata_send[sendb_num];
		int sendCount = 0;
		int offset = 0;
		for (int i = 0; i < sendb_num; i++) {
			int send_index = rotate_index_array[send_indexes[i]];
			metadata_send[i] = sendcounts[send_index];
			if (pos_status[send_index] == 0)
				memcpy(&temp_send_buffer[offset], &sendbuf[sdispls[send_index]*typesize], sendcounts[send_index]*typesize);
			else
				memcpy(&temp_send_buffer[offset], &extra_buffer[send_indexes[i]*max_send_count*typesize], sendcounts[send_index]*typesize);
			offset += sendcounts[send_index]*typesize;
		}

		// 3) exchange metadata
		int sendrank = (rank - k + nprocs) % nprocs;
		int recvrank = (rank + k) % nprocs;
		int metadata_recv[sendb_num];
		MPI_Sendrecv(metadata_send, sendb_num, MPI_INT, sendrank, 0, metadata_recv, sendb_num, MPI_INT, recvrank, 0, comm, MPI_STATUS_IGNORE);

		for(int i = 0; i < sendb_num; i++)
			sendCount += metadata_recv[i];

		// 4) exchange data
		MPI_Sendrecv(temp_send_buffer, offset, MPI_CHAR, sendrank, 1, temp_recv_buffer, sendCount*typesize, MPI_CHAR, recvrank, 1, comm, MPI_STATUS_IGNORE);

		// 5) replace
		offset = 0;
		for (int i = 0; i < sendb_num; i++) {
			int send_index = rotate_index_array[send_indexes[i]];

			if ((send_indexes[i] - rank + nprocs) % nprocs < (k << 1))
				memcpy(&recvbuf[rdispls[send_indexes[i]]*typesize], &temp_recv_buffer[offset], metadata_recv[i]*typesize);
			else
				memcpy(&extra_buffer[send_indexes[i]*max_send_count*typesize], &temp_recv_buffer[offset], metadata_recv[i]*typesize);

			offset += metadata_recv[i]*typesize;
			pos_status[send_index] = 1;
			sendcounts[send_index] = metadata_recv[i];
		}
	}
	free(temp_send_buffer);
	free(temp_recv_buffer);
	free(extra_buffer);
}



