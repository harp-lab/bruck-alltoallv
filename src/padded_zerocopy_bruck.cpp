/*
 * padded_zerocopy_bruck.cpp
 *
 *      Author: kokofan
 */

#include "non_uniform_bruck.h"

/// zerocopy bruck
void padded_zeroCopy_bruck(char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) {

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
	memcpy(temp_recv_buffer, temp_send_buffer, max_send_count*typesize*nprocs);

	// 3. exchange data with log(P) steps
	int bits[nprocs];
	bits[0] = 0; // the number of bits k' > k
	for (int j = 1; j < nprocs; j++) bits[j] = bits[j>>1]+(j&0x1);

	int commblocks[nprocs];
	MPI_Datatype commtypes[nprocs];
	MPI_Aint recvindex[nprocs];
	MPI_Aint sendindex[nprocs];

	char* inter_buffer = (char*)malloc(max_send_count*typesize*nprocs);
	long long unit_size = max_send_count * typesize;
	double total_create_dt_time = 0, total_comm_time = 0;
	unsigned int mask = 0xFFFFFFFF;
	int b, j;
	for (int k = 1; k < nprocs; k <<= 1) {
		// 1) create struct data-type that send and receive data from two buffers
		b = 0; j = k;
		while (j < nprocs) {
			int index = (rank+j)%nprocs;
			commblocks[b] = unit_size;
			commtypes[b] = MPI_CHAR;

			if ((bits[j&mask]&0x1)==0x1) { // send to recvbuf when the number of bits k' > k is odd
				recvindex[b] = (MPI_Aint)((char*)temp_recv_buffer+index*unit_size);

				if ((j & mask) == j) // recv from sendbuf when the number of bits k' > k is even
					sendindex[b] = (MPI_Aint)((char*)temp_send_buffer+index*unit_size);
				else // from intermediate buffer
					sendindex[b] = (MPI_Aint)((char*)inter_buffer+index*unit_size);
			}
			else { // send to intermediate buffer
				recvindex[b] = (MPI_Aint)((char*)inter_buffer+index*unit_size);

				if ((j & mask) == j) // recv from sendbuf when the number of bits k' > k is even
					sendindex[b] = (MPI_Aint)((char*)temp_send_buffer+index*unit_size);
				else
					sendindex[b] = (MPI_Aint)((char*)temp_recv_buffer+index*unit_size);
			}
			b++;
			j++; if ((j & k) != k) j += k; // data blocks whose kth bit is 1
		}

		MPI_Datatype sendblocktype;
		MPI_Type_create_struct(b, commblocks, sendindex, commtypes, &sendblocktype);
		MPI_Type_commit(&sendblocktype);
		MPI_Datatype recvblocktype;
		MPI_Type_create_struct(b,commblocks, recvindex, commtypes, &recvblocktype);
		MPI_Type_commit(&recvblocktype);

		// 2) exchange data
		int sendrank = (rank - k + nprocs) % nprocs;
		int recvrank = (rank + k) % nprocs;
		MPI_Sendrecv(MPI_BOTTOM, 1, sendblocktype, sendrank, 0, MPI_BOTTOM, 1, recvblocktype, recvrank, 0, comm, MPI_STATUS_IGNORE);

		MPI_Type_free(&recvblocktype);
		MPI_Type_free(&sendblocktype);

		mask <<= 1;
	}
	free(inter_buffer);
	free(temp_send_buffer);

	// 4. remove padding
	for (int i=0; i < nprocs; i++)
		memcpy(&recvbuf[rdispls[i]*typesize], &temp_recv_buffer[i*max_send_count*typesize], recvcounts[i]*typesize);
 	free(temp_recv_buffer);
}


