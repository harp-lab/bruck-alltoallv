/*
 * non_uniform_bruck.cpp
 *
 *      Author: kokofan
 */

#include "non_uniform_bruck.h"

// Padded bruck algorithm
void padded_bruck_non_uniform_benchmark(int dist, int range, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
	double u_start = MPI_Wtime();

	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	int typesize;
	MPI_Type_size(sendtype, &typesize);

	// 1. Find max send count
	double find_count_start = MPI_Wtime();
	int local_max_count = 0;
	for (int i = 0; i < nprocs; i++)
	{
		if (sendcounts[i] > local_max_count)
			local_max_count = sendcounts[i];
	}
	int max_send_count = 0;
	MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);
	double find_count_end = MPI_Wtime();
	double find_count_time = find_count_end - find_count_start;

	// 2. local rotation
	double rotation_start = MPI_Wtime();
	char* temp_send_buffer = (char*)malloc(max_send_count*nprocs*typesize);
	memset(temp_send_buffer, 0, max_send_count*nprocs*typesize);
	int offset = 0;
	for (int i = 0; i < nprocs; i++)
	{
		int index = (i - rank + nprocs) % nprocs;
		memcpy(&temp_send_buffer[index*max_send_count*typesize], &sendbuf[offset], sendcounts[i]*typesize);
		offset += sendcounts[i]*typesize;
	}
	double rotation_end = MPI_Wtime();
	double rotation_time = rotation_end - rotation_start;

	// 3. exchange data with log(P) steps
	double exchange_start =  MPI_Wtime();
	long long unit_size = max_send_count * typesize;
	double total_find_blocks_time = 0, total_copy_time = 0, total_comm_time = 0, total_replace_time = 0;
	char* temp_buffer = (char*)malloc(max_send_count*typesize*((nprocs+1)/2));
	char* temp_recv_buffer = (char*)malloc(max_send_count*typesize*((nprocs+1)/2));
	for (int k = 1; k < nprocs; k <<= 1)
	{
		// 1) find which data blocks to send
		double find_blocks_start = MPI_Wtime();
		int send_indexes[(nprocs+1)/2];
		int sendb_num = 0;
		for (int i = k; i < nprocs; i++)
		{
			if (i & k)
				send_indexes[sendb_num++] = i;
		}
		double find_blocks_end = MPI_Wtime();
		total_find_blocks_time += (find_blocks_end - find_blocks_start);

		// 2) copy blocks which need to be sent at this step
		double copy_start = MPI_Wtime();
		for (int i = 0; i < sendb_num; i++)
		{
			long long offset = send_indexes[i] * unit_size;
			memcpy(temp_buffer+(i*unit_size), temp_send_buffer+offset, unit_size);
		}
		double copy_end = MPI_Wtime();
		total_copy_time += (copy_end - copy_start);

		// 3) send and receive
		double comm_start = MPI_Wtime();
		int recv_proc = (rank - k + nprocs) % nprocs; // receive data from rank - 2^step process
		int send_proc = (rank + k) % nprocs; // send data from rank + 2^k process
		long long comm_size = sendb_num * unit_size;
		MPI_Sendrecv(temp_buffer, comm_size, MPI_CHAR, send_proc, 0, temp_recv_buffer, comm_size, MPI_CHAR, recv_proc, 0, comm, MPI_STATUS_IGNORE);
		double comm_end = MPI_Wtime();
		total_comm_time += (comm_end - comm_start);

		// 4) replace with received data
		double replace_start = MPI_Wtime();
		for (int i = 0; i < sendb_num; i++)
		{
			long long offset = send_indexes[i] * unit_size;
			memcpy(temp_send_buffer+offset, temp_recv_buffer+(i*unit_size), unit_size);
		}
		double replace_end = MPI_Wtime();
		total_replace_time += (replace_end - replace_start);
	}
	free(temp_buffer);
	free(temp_recv_buffer);
	double exchange_end = MPI_Wtime();
	double exchange_time = exchange_end - exchange_start;

	// 4. second rotation
	double revs_rotation_start = MPI_Wtime();
	offset = 0;
	for (int i = 0; i < nprocs; i++)
	{
		int index = (rank - i + nprocs) % nprocs;
		memcpy(&recvbuf[rdispls[index]*typesize], &temp_send_buffer[i*unit_size], recvcounts[index]*typesize);
	}
	free(temp_send_buffer);
	double revs_rotation_end = MPI_Wtime();
	double revs_rotation_time = revs_rotation_end - revs_rotation_start;

	double u_end = MPI_Wtime();
	double max_u_time = 0;
	double total_u_time = u_end - u_start;
	MPI_Allreduce(&total_u_time, &max_u_time, 1, MPI_DOUBLE, MPI_MAX, comm);
	if (total_u_time == max_u_time)
	{
		 std::cout << "[PaddedBruck] ["  << dist << " " << nprocs << " " << range << " " << max_send_count << "] " << total_u_time << " " << find_count_time << " " << rotation_time << " "
				 << exchange_time << " [" << total_find_blocks_time << " " << total_copy_time << " " << total_comm_time << " " << total_replace_time << "] "<< revs_rotation_time << std::endl;
	}
}


/// Padded MPI_alltoall algorithm
void padded_alltoall_non_uniform_benchmark(int dist, int range, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
	double u_start = MPI_Wtime();

	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	int typesize;
	MPI_Type_size(sendtype, &typesize);

	// 1. Find max send count
	double find_count_start = MPI_Wtime();
	int local_max_count = 0;
	for (int i = 0; i < nprocs; i++)
	{
		if (sendcounts[i] > local_max_count)
			local_max_count = sendcounts[i];
	}
	int max_send_count = 0;
	MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);
	double find_count_end = MPI_Wtime();
	double find_count_time = find_count_end - find_count_start;

	// 2. local padding
	double copy_start = MPI_Wtime();
	char* temp_send_buffer = (char*)malloc(max_send_count*nprocs*typesize);
	for (int i = 0; i < nprocs; i++)
		memcpy(&temp_send_buffer[i*max_send_count*typesize], &sendbuf[sdispls[i]*typesize], sendcounts[i]*typesize);
	double copy_end = MPI_Wtime();
	double copy_time = copy_end - copy_start;

	// 3. all-to-all communication
	double comm_start = MPI_Wtime();
	char* temp_recv_buffer = (char*)malloc(max_send_count*nprocs*typesize);
	MPI_Alltoall(temp_send_buffer, max_send_count*typesize, MPI_CHAR, temp_recv_buffer, max_send_count*typesize, MPI_CHAR, comm);
	free(temp_send_buffer);
	double comm_end = MPI_Wtime();
	double comm_time = (comm_end - comm_start);

	// 4. filter
	double filter_start = MPI_Wtime();
	int offset = 0;
	for (int i = 0; i < nprocs; i++)
		memcpy(&recvbuf[rdispls[i]*typesize], &temp_recv_buffer[i*max_send_count*typesize], recvcounts[i]*typesize);
	free(temp_recv_buffer);
	double filter_end = MPI_Wtime();
	double filter_time = filter_end - filter_start;

	double u_end = MPI_Wtime();
	double max_u_time = 0;
	double total_u_time = u_end - u_start;
	MPI_Allreduce(&total_u_time, &max_u_time, 1, MPI_DOUBLE, MPI_MAX, comm);
	if (total_u_time == max_u_time)
	{
		 std::cout << "[PaddedAlltoall] ["  << dist << " " << nprocs << " " << range << " " << max_send_count << "] " << total_u_time << " " << find_count_time << " " << copy_time << " "
				 << comm_time << " " << filter_time << std::endl;
	}
}


/// padded bruck algorithm with MPI_datatype
void datatype_bruck_non_uniform_benchmark(int range, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
	double u_start = MPI_Wtime();

	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	int typesize;
	MPI_Type_size(sendtype, &typesize);

	// 1. Find max send count
	double find_count_start = MPI_Wtime();
	int local_max_count = 0;
	for (int i = 0; i < nprocs; i++)
	{
		if (sendcounts[i] > local_max_count)
			local_max_count = sendcounts[i];
	}
	int max_send_count = 0;
	MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);
	double find_count_end = MPI_Wtime();
	double find_count_time = find_count_end - find_count_start;

	// 2. local rotation
	double rotation_start = MPI_Wtime();
	char* temp_send_buffer = (char*)malloc(max_send_count*nprocs*typesize);
	char* temp_recv_buffer = (char*)malloc(max_send_count*typesize*nprocs);
	memset(temp_send_buffer, 0, max_send_count*nprocs*typesize);
	int offset = 0;
	for (int i = 0; i < nprocs; i++)
	{
		int index = (i - rank + nprocs) % nprocs;
		memcpy(&temp_send_buffer[index*max_send_count*typesize], &sendbuf[offset], sendcounts[i]*typesize);
		offset += sendcounts[i]*typesize;
	}
	double rotation_end = MPI_Wtime();
	double rotation_time = rotation_end - rotation_start;

	// 3. exchange data with log(P) steps
	double exchange_start =  MPI_Wtime();
	long long unit_size = max_send_count * typesize;
	double total_create_dt_time=0, total_replace_time=0, total_comm_time=0;
	for (int k = 1; k < nprocs; k <<= 1)
	{
		// 1) create data type
		double create_datatype_start = MPI_Wtime();
		int displs[(nprocs+1)/2];
		int sendb_num = 0;
		for (int i = 1; i < nprocs; i++)
		{
			if (i & k)
				displs[sendb_num++] = i*unit_size;
		}
		MPI_Datatype send_type;
		MPI_Type_create_indexed_block(sendb_num, unit_size, displs, MPI_CHAR, &send_type);
		MPI_Type_commit(&send_type);
		double create_datatype_end = MPI_Wtime();
		total_create_dt_time += create_datatype_end - create_datatype_start;

		// 2) exchange data
		double comm_start = MPI_Wtime();
		int recv_proc = (rank - k + nprocs) % nprocs; // receive data from rank - 2^step process
		int send_proc = (rank + k) % nprocs; // send data from rank + 2^k process
		MPI_Sendrecv(temp_send_buffer, 1, send_type, send_proc, 0, temp_recv_buffer, 1, send_type, recv_proc, 0, comm, MPI_STATUS_IGNORE);
		MPI_Type_free(&send_type);
		double comm_end = MPI_Wtime();
		total_comm_time += (comm_end - comm_start);

		// 3) replace time
		double replace_start = MPI_Wtime();
		for (int i = 0; i < sendb_num; i++)
			memcpy(temp_send_buffer+displs[i], temp_recv_buffer+displs[i], unit_size);
		double replace_end = MPI_Wtime();
		total_replace_time += (replace_end - replace_start);
	}
	free(temp_recv_buffer);
	double exchange_end = MPI_Wtime();
	double exchange_time = exchange_end - exchange_start;

	// 4. second rotation
	double revs_rotation_start = MPI_Wtime();
	offset = 0;
	for (int i = 0; i < nprocs; i++)
	{
		int index = (rank - i + nprocs) % nprocs;
		memcpy(&recvbuf[rdispls[index]*typesize], &temp_send_buffer[i*unit_size], recvcounts[index]*typesize);
	}
	free(temp_send_buffer);
	double revs_rotation_end = MPI_Wtime();
	double revs_rotation_time = revs_rotation_end - revs_rotation_start;

	double u_end = MPI_Wtime();
	double max_u_time = 0;
	double total_u_time = u_end - u_start;
	MPI_Allreduce(&total_u_time, &max_u_time, 1, MPI_DOUBLE, MPI_MAX, comm);
	if (total_u_time == max_u_time)
	{
		 std::cout << "[DTBruckNoN] ["  << nprocs << " " << range << " " << max_send_count << "] " << total_u_time << " " << find_count_time << " " << rotation_time << " "
				 << exchange_time << " [" << total_create_dt_time << " " << total_comm_time << " " << total_replace_time << "] "<< revs_rotation_time << std::endl;
	}
}

/// modified bruck with MPI datatype
void modified_dt_bruck_non_uniform_benchmark(int range, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
	double u_start = MPI_Wtime();

	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	int typesize;
	MPI_Type_size(sendtype, &typesize);

	// 1. Find max send count
	double find_count_start = MPI_Wtime();
	int local_max_count = 0;
	for (int i = 0; i < nprocs; i++)
	{
		if (sendcounts[i] > local_max_count)
			local_max_count = sendcounts[i];
	}
	int max_send_count = 0;
	MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);
	double find_count_end = MPI_Wtime();
	double find_count_time = find_count_end - find_count_start;

	// 2. local rotation
	double rotation_start = MPI_Wtime();
	char* temp_send_buffer = (char*)malloc(max_send_count*nprocs*typesize);
	char* temp_recv_buffer = (char*)malloc(max_send_count*typesize*nprocs);
	memset(temp_send_buffer, 0, max_send_count*nprocs*typesize);
	int offset = 0;
	for (int i = 0; i < nprocs; i++)
	{
		int index = (2*rank-i+nprocs)%nprocs;
		memcpy(&temp_send_buffer[index*max_send_count*typesize], &sendbuf[offset], sendcounts[i]*typesize);
		offset += sendcounts[i]*typesize;
	}
	double rotation_end = MPI_Wtime();
	double rotation_time = rotation_end - rotation_start;

    // 3. exchange data with log(P) steps
    double exchange_start =  MPI_Wtime();
    long long unit_size = max_send_count * typesize;
 	double total_create_dt_time=0, total_comm_time=0, total_copy_time=0;
 	for (int k = 1; k < nprocs; k <<= 1)
 	{
 		// 1) create data type
		double create_datatype_start = MPI_Wtime();
		int displs[(nprocs+1)/2];
		int sendb_num = 0;
		for (int i = 1; i < nprocs; i++)
		{
			if (i & k)
				displs[sendb_num++] = ((rank+i)%nprocs)*unit_size;
		}
		MPI_Datatype send_type;
		MPI_Type_create_indexed_block(sendb_num, unit_size, displs, MPI_CHAR, &send_type);
		MPI_Type_commit(&send_type);
		double create_datatype_end = MPI_Wtime();
		total_create_dt_time += create_datatype_end - create_datatype_start;

		// 2) exchange data
		double comm_start = MPI_Wtime();
		int recv_proc = (rank + k) % nprocs; // receive data from rank + 2^k process
		int send_proc = (rank - k + nprocs) % nprocs; // send data from rank - 2^k process
		MPI_Sendrecv(temp_send_buffer, 1, send_type, send_proc, 0, temp_recv_buffer, 1, send_type, recv_proc, 0, comm, MPI_STATUS_IGNORE);
		MPI_Type_free(&send_type);
		double comm_end = MPI_Wtime();
		total_comm_time += (comm_end - comm_start);

		// 3) copy time
		double copy_start = MPI_Wtime();
		for (int i = 0; i < sendb_num; i++)
			memcpy(temp_send_buffer+displs[i], temp_recv_buffer+displs[i], unit_size);
		double copy_end = MPI_Wtime();
		total_copy_time += (copy_end - copy_start);
 	}
 	free(temp_recv_buffer);
 	double exchange_end = MPI_Wtime();
 	double exchange_time = exchange_end - exchange_start;

	// 4. remove padding
	double filter_start = MPI_Wtime();
	for (int i=0; i < nprocs; i++)
		memcpy(&recvbuf[rdispls[i]*typesize], &temp_send_buffer[i*max_send_count*typesize], recvcounts[i]*typesize);
 	free(temp_send_buffer);
	double filter_end = MPI_Wtime();
	double filter_time = filter_end - filter_start;

	double u_end = MPI_Wtime();
	double max_u_time = 0;
	double total_u_time = u_end - u_start;
	MPI_Allreduce(&total_u_time, &max_u_time, 1, MPI_DOUBLE, MPI_MAX, comm);
	if (total_u_time == max_u_time)
	{
		 std::cout << "[ModDtBruckNoN] ["  << nprocs << " " << range << " " << max_send_count << "] " << total_u_time << " " << find_count_time << " " << rotation_time << " "
				 << exchange_time << " [" << total_create_dt_time << " " << total_comm_time << " " << total_copy_time << "] "<< filter_time << std::endl;
	}
}


/// zerocopy bruck
void zeroCopy_bruck_non_uniform_benchmark(int range, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
	double u_start = MPI_Wtime();

	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	int typesize;
	MPI_Type_size(sendtype, &typesize);

	// 1. Find max send count
	double find_count_start = MPI_Wtime();
	int local_max_count = 0;
	for (int i = 0; i < nprocs; i++)
	{
		if (sendcounts[i] > local_max_count)
			local_max_count = sendcounts[i];
	}
	int max_send_count = 0;
	MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);
	double find_count_end = MPI_Wtime();
	double find_count_time = find_count_end - find_count_start;

	// 2. local rotation
	double rotation_start = MPI_Wtime();
	char* temp_send_buffer = (char*)malloc(max_send_count*nprocs*typesize);
	char* temp_recv_buffer = (char*)malloc(max_send_count*typesize*nprocs);
	memset(temp_send_buffer, 0, max_send_count*nprocs*typesize);
	int offset = 0;
	for (int i = 0; i < nprocs; i++)
	{
		int index = (2*rank-i+nprocs)%nprocs;
		memcpy(&temp_send_buffer[index*max_send_count*typesize], &sendbuf[offset], sendcounts[i]*typesize);
		offset += sendcounts[i]*typesize;
	}
	memcpy(temp_recv_buffer, temp_send_buffer, max_send_count*typesize*nprocs);
	double rotation_end = MPI_Wtime();
	double rotation_time = rotation_end - rotation_start;

	// 3. exchange data with log(P) steps
	double exchange_start =  MPI_Wtime();

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
	for (int k = 1; k < nprocs; k <<= 1)
	{
		// 1) create struct data-type that send and receive data from two buffers
		double create_datatype_start = MPI_Wtime();
		b = 0; j = k;
		while (j < nprocs)
		{
			int index = (rank+j)%nprocs;
			commblocks[b] = unit_size;
			commtypes[b] = MPI_CHAR;

			if ((bits[j&mask]&0x1)==0x1) // send to recvbuf when the number of bits k' > k is odd
			{
				recvindex[b] = (MPI_Aint)((char*)temp_recv_buffer+index*unit_size);

				if ((j & mask) == j) // recv from sendbuf when the number of bits k' > k is even
					sendindex[b] = (MPI_Aint)((char*)temp_send_buffer+index*unit_size);
				else // from intermediate buffer
					sendindex[b] = (MPI_Aint)((char*)inter_buffer+index*unit_size);
			}
			else // send to intermediate buffer
			{
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
		double create_datatype_end = MPI_Wtime();
		total_create_dt_time += (create_datatype_end - create_datatype_start);

		// 2) exchange data
		double comm_start = MPI_Wtime();
		int sendrank = (rank - k + nprocs) % nprocs;
		int recvrank = (rank + k) % nprocs;
		MPI_Sendrecv(MPI_BOTTOM, 1, sendblocktype, sendrank, 0, MPI_BOTTOM, 1, recvblocktype, recvrank, 0, comm, MPI_STATUS_IGNORE);

		MPI_Type_free(&recvblocktype);
		MPI_Type_free(&sendblocktype);
		double comm_end = MPI_Wtime();
		total_comm_time += (comm_end - comm_start);

		mask <<= 1;
	}
	free(inter_buffer);
	free(temp_send_buffer);
	double exchange_end =  MPI_Wtime();
	double exchange_time = exchange_end - exchange_start;

	// 4. remove padding
	double filter_start = MPI_Wtime();
	for (int i=0; i < nprocs; i++)
		memcpy(&recvbuf[rdispls[i]*typesize], &temp_recv_buffer[i*max_send_count*typesize], recvcounts[i]*typesize);
 	free(temp_recv_buffer);
	double filter_end = MPI_Wtime();
	double filter_time = filter_end - filter_start;

	double u_end = MPI_Wtime();
	double max_u_time = 0;
	double total_u_time = u_end - u_start;
	MPI_Allreduce(&total_u_time, &max_u_time, 1, MPI_DOUBLE, MPI_MAX, comm);
	if (total_u_time == max_u_time)
	{
		 std::cout << "[ZeroCopyBruckNoN] ["  << nprocs << " " << range << " " << max_send_count << "] " << total_u_time << " " << find_count_time << " " << rotation_time << " "
				 << exchange_time << " [" << total_create_dt_time << " " << total_comm_time << "] "<< filter_time << std::endl;
	}
}


void twophase_non_uniform_benchmark(int dist, int range, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	int typesize;
	MPI_Type_size(sendtype, &typesize);

	// 1. Find max send count
	double u_start = MPI_Wtime();
	double find_count_start = MPI_Wtime();

	int local_max_count = 0;
	int max_send_count = 0;

	for (int i = 0; i < nprocs; i++)
	{
		if (sendcounts[i] > local_max_count)
			local_max_count = sendcounts[i];
	}
	MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);

	double find_count_end = MPI_Wtime();
	double find_count_time = find_count_end - find_count_start;

	// 2. create local index array after rotation
	double create_rindex_start = MPI_Wtime();
	int rotate_index_array[nprocs];

    for (int i = 0; i < nprocs; i++)
    	rotate_index_array[i] = (2*rank-i+nprocs)%nprocs;

	double create_rindex_end = MPI_Wtime();
	double create_rindex_time = create_rindex_end - create_rindex_start;

	// 3. exchange data with log(P) steps
	double exchange_start = MPI_Wtime();
	int max_send_elements = (nprocs+1)/2;
	char* extra_buffer = (char*) malloc(max_send_count*typesize*nprocs);
	char* temp_send_buffer = (char*) malloc(max_send_count*typesize*max_send_elements);
	char* temp_recv_buffer = (char*) malloc(max_send_count*typesize*max_send_elements);
	double total_find_blocks_time = 0, total_pre_time = 0, total_send_meda_time = 0, total_comm_time = 0, total_replace_time = 0;
	int pos_status[nprocs];
	memset(pos_status, 0, nprocs*sizeof(int));

	int ind = 0;
	for (int k = 1; k < nprocs; k <<= 1)
	{
    	// 1) find which data blocks to send
    	double find_blocks_start = MPI_Wtime();
    	int send_indexes[max_send_elements];
    	int sendb_num = 0;
		for (int i = 1; i < nprocs; i++)
		{
			if (i & k)
				send_indexes[sendb_num++] = (rank+i)%nprocs;
		}
		double find_blocks_end = MPI_Wtime();
		total_find_blocks_time += (find_blocks_end - find_blocks_start);

		// 2) prepare metadata and send buffer
		double pre_send_start = MPI_Wtime();
		int metadata_send[sendb_num+1];

		int offset = 0;
		for (int i = 0; i < sendb_num; i++)
		{
			int send_index = rotate_index_array[send_indexes[i]];
			metadata_send[i] = sendcounts[send_index];
			if (pos_status[send_index] == 0)
				memcpy(&temp_send_buffer[offset], &sendbuf[sdispls[send_index]*typesize], sendcounts[send_index]*typesize);
			else
				memcpy(&temp_send_buffer[offset], &extra_buffer[send_indexes[i]*max_send_count*typesize], sendcounts[send_index]*typesize);
			offset += sendcounts[send_index]*typesize;
		}
		double pre_send_end = MPI_Wtime();
		total_pre_time += pre_send_end - pre_send_start;

		// 3) exchange metadata
		double send_meda_start = MPI_Wtime();
		int sendrank = (rank - k + nprocs) % nprocs;
		int recvrank = (rank + k) % nprocs;
		int metadata_recv[sendb_num];
		MPI_Sendrecv(metadata_send, sendb_num, MPI_INT, sendrank, 0, metadata_recv, sendb_num, MPI_INT, recvrank, 0, comm, MPI_STATUS_IGNORE);
		double send_meda_end = MPI_Wtime();
		total_send_meda_time += (send_meda_end - send_meda_start);

		// 4) exchange data
		double comm_start = MPI_Wtime();
		int sendCount = 0;
		for (int i = 0; i < sendb_num; i++)
			sendCount += metadata_recv[i];

		MPI_Sendrecv(temp_send_buffer, offset, MPI_CHAR, sendrank, 1, temp_recv_buffer, sendCount*typesize, MPI_CHAR, recvrank, 1, comm, MPI_STATUS_IGNORE);
		double comm_end = MPI_Wtime();
		total_comm_time = (comm_end - comm_start);

		// 5) replace
		double replace_start = MPI_Wtime();
		for (int i = 0; i < sendb_num; i++)
		{
			int send_index = rotate_index_array[send_indexes[i]];
			memcpy(&extra_buffer[send_indexes[i]*max_send_count*typesize], &temp_recv_buffer[offset], metadata_recv[i]*typesize);
			offset += metadata_recv[i]*typesize;
			pos_status[send_index] = 1;
			sendcounts[send_index] = metadata_recv[i];
		}
		double replace_end = MPI_Wtime();
		total_replace_time += (replace_end - replace_start);

		ind++;
	}
	free(temp_send_buffer);
	free(temp_recv_buffer);
	double exchange_end = MPI_Wtime();
	double exchange_time = (exchange_end - exchange_start);

	double filter_start = MPI_Wtime();
	for (int i = 0; i < nprocs; i++)
	{
		if (rank == i)
			memcpy(&recvbuf[rdispls[i]*typesize], &sendbuf[sdispls[i]*typesize], recvcounts[i]*typesize);
		else
			memcpy(&recvbuf[rdispls[i]*typesize], &extra_buffer[i*max_send_count*typesize], recvcounts[i]*typesize);
	}
	free(extra_buffer);

	double filter_end = MPI_Wtime();
	double filter_time = filter_end - filter_start;

    double u_end = MPI_Wtime();
	double max_u_time = 0;
	double total_u_time = u_end - u_start;
	MPI_Allreduce(&total_u_time, &max_u_time, 1, MPI_DOUBLE, MPI_MAX, comm);
	if (total_u_time == max_u_time)
	{
		 std::cout << "[TwoPhase] [" << dist << " " << nprocs << " " << range << " " << max_send_count << "] " << total_u_time << " " << find_count_time << " " << create_rindex_time << " " << exchange_time << " ["
				 << total_find_blocks_time << " " << total_pre_time << " " << total_send_meda_time << " " << total_comm_time << " " << total_replace_time << "] " << filter_time << std::endl;
	}
}


void ptp_non_uniform_benchmark(char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	int typesize;
	MPI_Type_size(sendtype, &typesize);

	MPI_Request* req = (MPI_Request*)malloc(2*nprocs*sizeof(MPI_Request));
	MPI_Status* stat = (MPI_Status*)malloc(2*nprocs*sizeof(MPI_Status));
	for (int i = 0; i < nprocs; i++)
	{
		int src = (rank + i) % nprocs; // avoid always to reach first master node
		MPI_Irecv(&recvbuf[rdispls[src]*typesize], recvcounts[src]*typesize, MPI_CHAR, src, 0, comm, &req[i]);
	}

	for (int i = 0; i < nprocs; i++)
	{
		int dst = (rank - i + nprocs) % nprocs;
		MPI_Isend(&sendbuf[sdispls[dst]*typesize], sendcounts[dst]*typesize, MPI_CHAR, dst, 0, comm, &req[i+nprocs]);
	}
	MPI_Waitall(2*nprocs, req, stat);
	free(req);
	free(stat);
}


void new_twophase_non_uniform_benchmark(int dist, int range, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	int typesize;
	MPI_Type_size(sendtype, &typesize);

	// 1. Find max send count
	double u_start = MPI_Wtime();
	double find_count_start = MPI_Wtime();
	int local_max_count = 0;
	for (int i = 0; i < nprocs; i++)
	{
		if (sendcounts[i] > local_max_count)
			local_max_count = sendcounts[i];
	}
	int max_send_count = 0;
	MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);
	double find_count_end = MPI_Wtime();
	double find_count_time = find_count_end - find_count_start;

	// 2. create local index array after rotation
	double create_rindex_start = MPI_Wtime();
	int rotate_index_array[nprocs];
	for (int i = 0; i < nprocs; i++)
		rotate_index_array[i] = (2*rank-i+nprocs)%nprocs;
	double create_rindex_end = MPI_Wtime();
	double create_rindex_time = create_rindex_end - create_rindex_start;

	// 3. exchange data with log(P) steps
	double exchange_start = MPI_Wtime();
	int max_send_elements = (nprocs+1)/2;
	char* extra_buffer = (char*) malloc(max_send_count*typesize*nprocs);
	char* temp_send_buffer = (char*) malloc(max_send_count*typesize*max_send_elements);
	char* temp_recv_buffer = (char*) malloc(max_send_count*typesize*max_send_elements);
	int pos_status[nprocs];
	memset(pos_status, 0, nprocs*sizeof(int));
	double total_find_blocks_time = 0, total_pre_time = 0, total_send_meda_time = 0, total_comm_time = 0, total_replace_time = 0;

	memcpy(&recvbuf[rdispls[rank]*typesize], &sendbuf[sdispls[rank]*typesize], recvcounts[rank]*typesize);

	for (int k = 1; k < nprocs; k <<= 1)
	{
		// 1) find which data blocks to send
		double find_blocks_start = MPI_Wtime();
		int send_indexes[max_send_elements];
		int sendb_num = 0;
		for (int i = k; i < nprocs; i++)
		{
			if (i & k)
				send_indexes[sendb_num++] = (rank+i)%nprocs;
		}

		double find_blocks_end = MPI_Wtime();
		total_find_blocks_time += (find_blocks_end - find_blocks_start);

		// 2) prepare metadata and send buffer
		double pre_send_start = MPI_Wtime();
		int metadata_send[sendb_num];
		int sendCount = 0;
		int offset = 0;
		for (int i = 0; i < sendb_num; i++)
		{
			int send_index = rotate_index_array[send_indexes[i]];
			metadata_send[i] = sendcounts[send_index];
			if (pos_status[send_index] == 0)
				memcpy(&temp_send_buffer[offset], &sendbuf[sdispls[send_index]*typesize], sendcounts[send_index]*typesize);
			else
				memcpy(&temp_send_buffer[offset], &extra_buffer[send_indexes[i]*max_send_count*typesize], sendcounts[send_index]*typesize);
			offset += sendcounts[send_index]*typesize;
		}
		double pre_send_end = MPI_Wtime();
		total_pre_time += pre_send_end - pre_send_start;


		// 3) exchange metadata
		double send_meda_start = MPI_Wtime();
		int sendrank = (rank - k + nprocs) % nprocs;
		int recvrank = (rank + k) % nprocs;
		int metadata_recv[sendb_num];
		MPI_Sendrecv(metadata_send, sendb_num, MPI_INT, sendrank, 0, metadata_recv, sendb_num, MPI_INT, recvrank, 0, comm, MPI_STATUS_IGNORE);
		double send_meda_end = MPI_Wtime();
		total_send_meda_time += (send_meda_end - send_meda_start);

		for(int i = 0; i < sendb_num; i++)
			sendCount += metadata_recv[i];

		// 4) exchange data
		double comm_start = MPI_Wtime();
		MPI_Sendrecv(temp_send_buffer, offset, MPI_CHAR, sendrank, 1, temp_recv_buffer, sendCount*typesize, MPI_CHAR, recvrank, 1, comm, MPI_STATUS_IGNORE);
		double comm_end = MPI_Wtime();
		total_comm_time = (comm_end - comm_start);

		// 5) replace
		double replace_start = MPI_Wtime();
		offset = 0;
		for (int i = 0; i < sendb_num; i++)
		{
			int send_index = rotate_index_array[send_indexes[i]];

			if ((send_indexes[i] - rank + nprocs) % nprocs < (k << 1))
				memcpy(&recvbuf[rdispls[send_indexes[i]]*typesize], &temp_recv_buffer[offset], metadata_recv[i]*typesize);
			else
				memcpy(&extra_buffer[send_indexes[i]*max_send_count*typesize], &temp_recv_buffer[offset], metadata_recv[i]*typesize);

			offset += metadata_recv[i]*typesize;
			pos_status[send_index] = 1;
			sendcounts[send_index] = metadata_recv[i];
		}
		double replace_end = MPI_Wtime();
		total_replace_time += (replace_end - replace_start);
	}
	free(temp_send_buffer);
	free(temp_recv_buffer);
	free(extra_buffer);


	double exchange_end = MPI_Wtime();
	double exchange_time = (exchange_end - exchange_start);

    double u_end = MPI_Wtime();
	double max_u_time = 0;
	double total_u_time = u_end - u_start;
	MPI_Allreduce(&total_u_time, &max_u_time, 1, MPI_DOUBLE, MPI_MAX, comm);
	if (total_u_time == max_u_time)
	{
		 std::cout << "[NewTwoPhase] [" << dist << " " << nprocs << " " << range << " " << max_send_count << "] " << total_u_time << " " << find_count_time << " " << create_rindex_time << " " << exchange_time << " ["
				 << total_find_blocks_time << " " << total_pre_time << " " << total_send_meda_time << " " << total_comm_time << " " << total_replace_time << "] " << std::endl;
	}
}

