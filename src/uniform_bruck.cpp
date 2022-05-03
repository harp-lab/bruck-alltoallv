/*
 * bruck.cpp
 *
 *      Author: kokofan
 */

#include "uniform_bruck.h"

// naive Bruck (without any datatype)
void basic_bruck_uniform_benchmark(char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm)
{
	double u_start = MPI_Wtime();

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    int typesize;
    MPI_Type_size(sendtype, &typesize);

    long long unit_size = sendcount * typesize;
    long long local_size = sendcount * nprocs * typesize;

	// 1. local rotation
	double rotation_start = MPI_Wtime();
	memcpy(recvbuf, sendbuf, local_size);
	for (int i = 0; i < nprocs; i++)
	{
		int index = (i - rank + nprocs) % nprocs;
		memcpy(&sendbuf[index*unit_size], &recvbuf[i*unit_size], unit_size);
	}
    double rotation_end = MPI_Wtime();
    double rotation_time = rotation_end - rotation_start;

    // 2. exchange data with log(P) steps
    double exchange_start =  MPI_Wtime();
    double total_find_blocks_time = 0, total_copy_time = 0, total_comm_time = 0, total_replace_time = 0;
    char* temp_buffer = (char*)malloc(local_size);
    for (int k = 1; k < nprocs; k <<= 1)
    {
    	// 1) find which data blocks to send
    	double find_blocks_start = MPI_Wtime();
    	int send_indexes[(nprocs+1)/2];
    	int sendb_num = 0;
		for (int i = 1; i < nprocs; i++)
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
			memcpy(temp_buffer+(i*unit_size), sendbuf+offset, unit_size);
		}
		double copy_end = MPI_Wtime();
		total_copy_time += (copy_end - copy_start);

		// 3) send and receive
		double comm_start = MPI_Wtime();
		int recv_proc = (rank - k + nprocs) % nprocs; // receive data from rank - 2^step process
		int send_proc = (rank + k) % nprocs; // send data from rank + 2^k process
		long long comm_size = sendb_num * unit_size;
		MPI_Sendrecv(temp_buffer, comm_size, MPI_CHAR, send_proc, 0, recvbuf, comm_size, MPI_CHAR, recv_proc, 0, comm, MPI_STATUS_IGNORE);
		double comm_end = MPI_Wtime();
		total_comm_time += (comm_end - comm_start);

		// 4) replace with received data
		double replace_start = MPI_Wtime();
		for (int i = 0; i < sendb_num; i++)
		{
			long long offset = send_indexes[i] * unit_size;
			memcpy(sendbuf+offset, recvbuf+(i*unit_size), unit_size);
		}
		double replace_end = MPI_Wtime();
		total_replace_time += (replace_end - replace_start);
    }
    free(temp_buffer);
    double exchange_end = MPI_Wtime();
    double exchange_time = exchange_end - exchange_start;

    // 3. second rotation
	double revs_rotation_start = MPI_Wtime();
	for (int i = 0; i < nprocs; i++)
	{
		int index = (rank - i + nprocs) % nprocs;
		memcpy(&recvbuf[index*unit_size], &sendbuf[i*unit_size], unit_size);
	}
    double revs_rotation_end = MPI_Wtime();
    double revs_rotation_time = revs_rotation_end - revs_rotation_start;

    double u_end = MPI_Wtime();
	double max_u_time = 0;
	double total_u_time = u_end - u_start;
	MPI_Allreduce(&total_u_time, &max_u_time, 1, MPI_DOUBLE, MPI_MAX, comm);
	if (total_u_time == max_u_time)
	{
		 std::cout << "[BasicBruck] ["  << nprocs << " " << sendcount << "] " << total_u_time << " " << rotation_time << " " << exchange_time << " ["
				 << total_find_blocks_time << " " << total_copy_time << " " << total_comm_time << " " << total_replace_time
				 << "] " << revs_rotation_time << std::endl;
	}
}


// naive Bruck (with MPI datatype)
void datatype_bruck_uniform_benchmark(char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm)
{
	double u_start = MPI_Wtime();

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    int typesize;
    MPI_Type_size(sendtype, &typesize);

    long long unit_size = sendcount * typesize;
    long long local_size = sendcount * nprocs * typesize;

	// 1. local rotation
	double rotation_start = MPI_Wtime();
	memcpy(recvbuf, sendbuf, local_size);
	for (int i = 0; i < nprocs; i++)
	{
		int index = (i - rank + nprocs) % nprocs;
		memcpy(&sendbuf[index*unit_size], &recvbuf[i*unit_size], unit_size);
	}
    double rotation_end = MPI_Wtime();
    double rotation_time = rotation_end - rotation_start;

    // 2. exchange data with log(P) steps
    double exchange_start =  MPI_Wtime();
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
//    	MPI_Type_vector(block_count, k*unit_size, (k<<1)*unit_size, MPI_CHAR, &vector_type);
    	MPI_Type_create_indexed_block(sendb_num, unit_size, displs, MPI_CHAR, &send_type);
    	MPI_Type_commit(&send_type);
    	int packsize;
    	MPI_Pack_size(1, send_type, MPI_COMM_WORLD, &packsize);
    	double create_datatype_end = MPI_Wtime();
    	total_create_dt_time += create_datatype_end - create_datatype_start;

    	// 2) exchange data
    	double comm_start = MPI_Wtime();
    	int recv_proc = (rank - k + nprocs) % nprocs; // receive data from rank - 2^step process
    	int send_proc = (rank + k) % nprocs; // send data from rank + 2^k process
    	MPI_Sendrecv(sendbuf, 1, send_type, send_proc, 0, recvbuf, packsize, MPI_PACKED, recv_proc, 0, comm, MPI_STATUS_IGNORE);
    	double comm_end = MPI_Wtime();
		total_comm_time += (comm_end - comm_start);

		// 3) replace time
		double replace_start = MPI_Wtime();
		int pos = 0;
		MPI_Unpack(recvbuf, packsize, &pos, sendbuf, 1, send_type, comm);
    	MPI_Type_free(&send_type);
//		for (int i = 0; i < sendb_num; i++)
//			memcpy(sendbuf+displs[i], recvbuf+displs[i], unit_size);
		double replace_end = MPI_Wtime();
		total_replace_time += (replace_end - replace_start);
    }
	double exchange_end = MPI_Wtime();
	double exchange_time = exchange_end - exchange_start;

	// 3. second rotation
	double revs_rotation_start = MPI_Wtime();
	for (int i = 0; i < nprocs; i++)
	{
		int index = (rank - i + nprocs) % nprocs;
		memcpy(&recvbuf[index*unit_size], &sendbuf[i*unit_size], unit_size);
	}
	double revs_rotation_end = MPI_Wtime();
	double revs_rotation_time = revs_rotation_end - revs_rotation_start;

    double u_end = MPI_Wtime();
	double max_u_time = 0;
	double total_u_time = u_end - u_start;
	MPI_Allreduce(&total_u_time, &max_u_time, 1, MPI_DOUBLE, MPI_MAX, comm);
	if (total_u_time == max_u_time)
	{
		 std::cout << "[DTBruck] [" << nprocs << " " << sendcount << "] " << total_u_time << " " << rotation_time << " " << exchange_time << " ["
				 << total_create_dt_time << " " << total_comm_time << " " << total_replace_time << "] " << revs_rotation_time << std::endl;
	}
}


/// modified bruck (without any datatype)
void modified_basic_bruck_uniform_benchmark(char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm)
{
	double u_start = MPI_Wtime();

	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	int typesize;
	MPI_Type_size(sendtype, &typesize);

	long long unit_size = sendcount * typesize;
	long long local_size = sendcount * nprocs * typesize;

    // 1. local rotation
    double rotation_start = MPI_Wtime();
    for (int i = 0; i < nprocs; i++)
    {
    	int index = (2*rank - i + nprocs) % nprocs;
    	memcpy(&recvbuf[index*unit_size], &sendbuf[i*unit_size], unit_size);
    }
    double rotation_end = MPI_Wtime();
    double rotation_time = rotation_end - rotation_start;

    // 2. exchange data with log(P) steps
    double exchange_start =  MPI_Wtime();
	double total_find_blocks_time = 0, total_copy_time = 0, total_comm_time = 0, total_replace_time = 0;
	char* temp_buffer = (char*)malloc(local_size);
	for (int k = 1; k < nprocs; k <<= 1)
	{
    	// 1) find which data blocks to send
    	double find_blocks_start = MPI_Wtime();
    	int send_indexes[(nprocs+1)/2];
    	int sendb_num = 0;
		for (int i = 1; i < nprocs; i++)
		{
			if (i & k)
				send_indexes[sendb_num++] = (rank+i) % nprocs;
		}
		double find_blocks_end = MPI_Wtime();
		total_find_blocks_time += (find_blocks_end - find_blocks_start);

		// 2) copy blocks which need to be sent at this step
		double copy_start = MPI_Wtime();
		for (int i = 0; i < sendb_num; i++)
		{
			long long offset = send_indexes[i] * unit_size;
			memcpy(temp_buffer+(i*unit_size), recvbuf+offset, unit_size);
		}
		double copy_end = MPI_Wtime();
		total_copy_time += (copy_end - copy_start);

		// 3) send and receive
		double comm_start = MPI_Wtime();
		int recv_proc = (rank + k) % nprocs; // receive data from rank + 2^k process
		int send_proc = (rank - k + nprocs) % nprocs; // send data from rank - 2^k process
		long long comm_size = sendb_num * unit_size;
		MPI_Sendrecv(temp_buffer, comm_size, MPI_CHAR, send_proc, 0, sendbuf, comm_size, MPI_CHAR, recv_proc, 0, comm, MPI_STATUS_IGNORE);
		double comm_end = MPI_Wtime();
		total_comm_time += (comm_end - comm_start);

		// 4) replace with received data
		double replace_start = MPI_Wtime();
		for (int i = 0; i < sendb_num; i++)
		{
			long long offset = send_indexes[i] * unit_size;
			memcpy(recvbuf+offset, sendbuf+(i*unit_size), unit_size);
		}
		double replace_end = MPI_Wtime();
		total_replace_time += (replace_end - replace_start);
	}
	free(temp_buffer);
	double exchange_end = MPI_Wtime();
	double exchange_time = exchange_end - exchange_start;

    double u_end = MPI_Wtime();
	double max_u_time = 0;
	double total_u_time = u_end - u_start;
	MPI_Allreduce(&total_u_time, &max_u_time, 1, MPI_DOUBLE, MPI_MAX, comm);
	if (total_u_time == max_u_time)
	{
		 std::cout << "[ModNaiveBruck] ["  << nprocs << " " << sendcount << "] " << total_u_time << " " << rotation_time << " " << exchange_time << " ["
				 << total_find_blocks_time << " " << total_copy_time << " " << total_comm_time << " " << total_replace_time << "] " << std::endl;
	}
}

/// modified bruck (with MPI datatype)
void modified_dt_bruck_uniform_benchmark(char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm)
{
	double u_start = MPI_Wtime();

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    int typesize;
    MPI_Type_size(sendtype, &typesize);

    long long unit_size = sendcount * typesize;
    long long local_size = sendcount * nprocs * typesize;

    // 1. local rotation
    double rotation_start = MPI_Wtime();
    for (int i = 0; i < nprocs; i++)
    {
    	int index = (2*rank-i+nprocs)%nprocs;
    	memcpy(&recvbuf[index*unit_size], &sendbuf[i*unit_size], unit_size);
    }
    double rotation_end = MPI_Wtime();
    double rotation_time = rotation_end - rotation_start;

    // 2. exchange data with log(P) steps
    double exchange_start =  MPI_Wtime();
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
		MPI_Sendrecv(recvbuf, 1, send_type, send_proc, 0, sendbuf, 1, send_type, recv_proc, 0, comm, MPI_STATUS_IGNORE);
		MPI_Type_free(&send_type);
		double comm_end = MPI_Wtime();
		total_comm_time += (comm_end - comm_start);

		// 3) copy time
		double copy_start = MPI_Wtime();
		for (int i = 0; i < sendb_num; i++)
			memcpy(recvbuf+displs[i], sendbuf+displs[i], unit_size);
		double copy_end = MPI_Wtime();
		total_copy_time += (copy_end - copy_start);
 	}
 	double exchange_end = MPI_Wtime();
 	double exchange_time = exchange_end - exchange_start;

    double u_end = MPI_Wtime();
 	double max_u_time = 0;
 	double total_u_time = u_end - u_start;
 	MPI_Allreduce(&total_u_time, &max_u_time, 1, MPI_DOUBLE, MPI_MAX, comm);
	if (total_u_time == max_u_time)
	{
		 std::cout << "[ModDtBruck] [" << nprocs << " " << sendcount << "] " << total_u_time << " " << rotation_time << " " << exchange_time << " ["
				 << total_create_dt_time << " " << total_comm_time << " " << total_copy_time << "] " << std::endl;
	}
}

/// bruck algorithm with no rotation phases
void noRotation_bruck_uniform_benchmark(char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm)
{
	double u_start = MPI_Wtime();

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    int typesize;
    MPI_Type_size(sendtype, &typesize);

    long long unit_size = sendcount * typesize;
    long long local_size = sendcount * nprocs * typesize;

	// 1. create local index array after rotation
	double create_rindex_start = MPI_Wtime();
	int rotate_index_array[nprocs];
    for (int i = 0; i < nprocs; i++)
    	rotate_index_array[i] = (2*rank-i+nprocs)%nprocs;
	double create_rindex_end = MPI_Wtime();
	double create_rindex_time = create_rindex_end - create_rindex_start;

	// 2. Initial receive buffer
	memcpy(recvbuf+rank*unit_size, sendbuf+rank*unit_size, unit_size);
    double exchange_start =  MPI_Wtime();
 	double total_create_dt_time = 0, total_comm_time = 0, total_copy_time = 0;
 	char* temp_buffer = (char*)malloc(local_size);
 	for (int k = 1; k < nprocs; k <<= 1)
 	{
 		// 1) create data type
		double create_datatype_start = MPI_Wtime();
		int send_displs[(nprocs+1)/2];
		int send_index[(nprocs+1)/2];
		int sendb_num = 0;
		for (int i = 1; i < nprocs; i++)
		{
			if (i & k)
			{
				send_index[sendb_num] = (rank+i)%nprocs;
				send_displs[sendb_num] = rotate_index_array[(rank+i)%nprocs]*unit_size;
				sendb_num++;
			}
		}
		MPI_Datatype send_type;
		MPI_Type_create_indexed_block(sendb_num, unit_size, send_displs, MPI_CHAR, &send_type);
		MPI_Type_commit(&send_type);
		int packsize;
		MPI_Pack_size(1, send_type, comm, &packsize);
		double create_datatype_end = MPI_Wtime();
		total_create_dt_time += create_datatype_end - create_datatype_start;

		// 2) exchange data
		double comm_start = MPI_Wtime();
		int recv_proc = (rank + k) % nprocs; // receive data from rank + 2^k process
		int send_proc = (rank - k + nprocs) % nprocs; // send data from rank - 2^k process
		MPI_Sendrecv(sendbuf, 1, send_type, send_proc, 0, temp_buffer, packsize, MPI_PACKED, recv_proc, 0, comm, MPI_STATUS_IGNORE);
		int pos = 0;
		MPI_Unpack(temp_buffer, packsize, &pos, sendbuf, 1, send_type, comm);
		MPI_Type_free(&send_type);
		double comm_end = MPI_Wtime();
		total_comm_time += (comm_end - comm_start);

		// 3) copy data to recvbuf
		double copy_start = MPI_Wtime();
		for (int i = 0; i < sendb_num; i++)
			memcpy(recvbuf+send_index[i]*unit_size, temp_buffer+i*unit_size, unit_size);
		double copy_end = MPI_Wtime();
		total_copy_time += (copy_end - copy_start);
 	}
 	free(temp_buffer);
 	double exchange_end =  MPI_Wtime();
 	double exchange_time = exchange_end - exchange_start;

    double u_end = MPI_Wtime();
 	double max_u_time = 0;
 	double total_u_time = u_end - u_start;
 	MPI_Allreduce(&total_u_time, &max_u_time, 1, MPI_DOUBLE, MPI_MAX, comm);
	if (total_u_time == max_u_time)
	{
		 std::cout << "[NoRotBruck] [" << nprocs << " " << sendcount << "] " << total_u_time << " " << create_rindex_time << " " << exchange_time << " ["
				 << total_create_dt_time << " " << total_comm_time << " " << total_copy_time << "] " << std::endl;
	}
}

/// zero copy bruck
void zerocopy_bruck_uniform_benchmark(char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm)
{
	double u_start = MPI_Wtime();

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    int typesize;
    MPI_Type_size(sendtype, &typesize);

    long long unit_size = sendcount * typesize;
    long long local_size = sendcount * nprocs * typesize;

    // 1. local rotation
    double rotation_start = MPI_Wtime();
    for (int i = 0; i < nprocs; i++)
    {
    	int index = (2*rank-i+nprocs)%nprocs;
    	memcpy(&recvbuf[index*unit_size], &sendbuf[i*unit_size], unit_size);
    }
    double rotation_end = MPI_Wtime();
    double rotation_time = rotation_end - rotation_start;

    // 2. initial data to recv_buffer and intermediate buffer
    double initial_start = MPI_Wtime();
    char* temp_buffer = (char*)malloc(local_size);
	memcpy(temp_buffer, recvbuf, local_size);
	memcpy(sendbuf, recvbuf, local_size);

	int bits[nprocs];
	bits[0] = 0; // the number of bits k' > k
	for (int j = 1; j < nprocs; j++) bits[j] = bits[j>>1]+(j&0x1);

	double initial_end = MPI_Wtime();
	double initial_time = initial_end - initial_start;

	// 3. exchange data with log(P) steps
	double exchange_start =  MPI_Wtime();
	int commblocks[nprocs];
	MPI_Datatype commtypes[nprocs];
	MPI_Aint recvindex[nprocs];
	MPI_Aint sendindex[nprocs];
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
			int index = (rank + j) % nprocs;
			commblocks[b] = unit_size;
			commtypes[b] = MPI_CHAR;

			if ((bits[j&mask]&0x1)==0x1) // send to recvbuf when the number of bits k' > k is odd
			{
				recvindex[b] = (MPI_Aint)((char*)recvbuf+index*unit_size);

				if ((j & mask) == j) // recv from sendbuf when the number of bits k' > k is even
					sendindex[b] = (MPI_Aint)((char*)sendbuf+index*unit_size);
				else // from intermediate buffer
					sendindex[b] = (MPI_Aint)((char*)temp_buffer+index*unit_size);
			}
			else // send to intermediate buffer
			{
				recvindex[b] = (MPI_Aint)((char*)temp_buffer+index*unit_size);

				if ((j & mask) == j) // recv from sendbuf when the number of bits k' > k is even
					sendindex[b] = (MPI_Aint)((char*)sendbuf+index*unit_size);
				else
					sendindex[b] = (MPI_Aint)((char*)recvbuf+index*unit_size);
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
	free(temp_buffer);
	double exchange_end =  MPI_Wtime();
	double exchange_time = exchange_end - exchange_start;

	double u_end = MPI_Wtime();
	double max_u_time = 0;
	double total_u_time = u_end - u_start;
	MPI_Allreduce(&total_u_time, &max_u_time, 1, MPI_DOUBLE, MPI_MAX, comm);
	if (total_u_time == max_u_time)
	{
		 std::cout << "[ZerocopyBruck] [" << nprocs << " " << sendcount << "] " << total_u_time << " " << rotation_time << " " << exchange_time << " ["
				 << total_create_dt_time << " " << total_comm_time << "] " << std::endl;
	}
}


/// no internal copy & no rotation phases
void zeroCopyRot_bruck_uniform_benchmark(char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm)
{
	double u_start = MPI_Wtime();

	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	int typesize;
	MPI_Type_size(sendtype, &typesize);

	long long unit_size = sendcount * typesize;
	long long local_size = sendcount * nprocs * typesize;

	// 1. create local index array after rotation
	double create_rindex_start = MPI_Wtime();
	int rotate_index_array[nprocs];
    for (int i = 0; i < nprocs; i++)
    	rotate_index_array[i] = (2*rank-i+nprocs)%nprocs;
	double create_rindex_end = MPI_Wtime();
	double create_rindex_time = create_rindex_end - create_rindex_start;

	// 2. initial data to recv_buffer and intermediate buffer
	double initial_start = MPI_Wtime();
	char* temp_buffer = (char*)malloc(local_size);
	memcpy(temp_buffer, sendbuf, local_size);
	memcpy(recvbuf, sendbuf, local_size);

	int bits[nprocs];
	bits[0] = 0; // the number of bits k' > k
	for (int j = 1; j < nprocs; j++) bits[j] = bits[j>>1]+(j&0x1);

	double initial_end = MPI_Wtime();
	double initial_time = initial_end - initial_start;

	// 3. exchange data with log(P) steps
	double exchange_start =  MPI_Wtime();
	int commblocks[nprocs];
	MPI_Datatype commtypes[nprocs];
	MPI_Aint recvindex[nprocs];
	MPI_Aint sendindex[nprocs];
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
			int send_index = rotate_index_array[(rank+j)%nprocs];
			int recv_index = (rank+j)%nprocs;

			commblocks[b] = unit_size;
			commtypes[b] = MPI_CHAR;

			if ((bits[j&mask]&0x1)==0x1) // send to recvbuf when the number of bits k' > k is odd
			{
				recvindex[b] = (MPI_Aint)((char*)recvbuf+recv_index*unit_size);

				if ((j & mask) == j) // recv from sendbuf when the number of bits k' > k is even
					sendindex[b] = (MPI_Aint)((char*)sendbuf+send_index*unit_size);
				else // from intermediate buffer
					sendindex[b] = (MPI_Aint)((char*)temp_buffer+send_index*unit_size);
			}
			else // send to intermediate buffer
			{
				recvindex[b] = (MPI_Aint)((char*)temp_buffer+send_index*unit_size);

				if ((j & mask) == j) // recv from sendbuf when the number of bits k' > k is even
					sendindex[b] = (MPI_Aint)((char*)sendbuf+send_index*unit_size);
				else
					sendindex[b] = (MPI_Aint)((char*)recvbuf+recv_index*unit_size);
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
	free(temp_buffer);
	double exchange_end =  MPI_Wtime();
	double exchange_time = exchange_end - exchange_start;

	double u_end = MPI_Wtime();
	double max_u_time = 0;
	double total_u_time = u_end - u_start;
	MPI_Allreduce(&total_u_time, &max_u_time, 1, MPI_DOUBLE, MPI_MAX, comm);
	if (total_u_time == max_u_time)
	{
		 std::cout << "[ZeroRotCopyBruck] [" << nprocs << " " << sendcount << "] " << total_u_time << " " << create_rindex_time << " " << exchange_time << " ["
				 << total_create_dt_time << " " << total_comm_time << "] " << std::endl;
	}
}


