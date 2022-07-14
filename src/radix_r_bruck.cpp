/*
 * r_radix_bruck.cpp
 *
 *  Created on: Jul 09, 2022
 *      Author: kokofan
 */

#include "radix_r_bruck.h"

std::vector<int> convert10tob(int w, int N, int b)
{
	std::vector<int> v(w);
	int i = 0;
	while(N) {
	  v[i++] = (N % b);
	  N /= b;
	}
//	std::reverse(v.begin(), v.end());
	return v;
}

void uniform_radix_r_bruck(double timelist[][7], int it, int r, char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm)
{
	double ts = MPI_Wtime();

	double s = MPI_Wtime();
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    int typesize;
    MPI_Type_size(sendtype, &typesize);

    int unit_size = sendcount * typesize;
    int w = ceil(log(nprocs) / log(r)); // calculate the number of digits when using r-representation
	int nlpow = pow(r, w-1);
	int d = (pow(r, w) - nprocs) / nlpow; // calculate the number of highest digits

    // local rotation
    std::memcpy(recvbuf, &sendbuf[rank*unit_size], (nprocs - rank)*unit_size);
    std::memcpy(&recvbuf[(nprocs - rank)*unit_size], sendbuf, rank*unit_size);
    double e = MPI_Wtime();
    double first_time = e - s;

    // convert rank to base r representation
    s = MPI_Wtime();
    int* rank_r_reps = (int*) malloc(nprocs * w * sizeof(int));
	for (int i = 0; i < nprocs; i++) {
		std::vector<int> r_rep = convert10tob(w, i, r);
		std::memcpy(&rank_r_reps[i*w], r_rep.data(), w*sizeof(int));
	}

	int sent_blocks[nlpow];
	int di = 0;
	int ci = 0;

	int comm_steps = (r - 1)*w - d;
	int nblocks_perstep[comm_steps];
	int total_comm_steps = 0;
	int istep = 0;

	char* temp_buffer = (char*)malloc(nlpow * unit_size); // temporary buffer
	e = MPI_Wtime();
	double conv_time = e - s;

	// communication steps = (r - 1)w - d
	double pre_time = 0, comm_time = 0, replace_time = 0;
    for (int x = 0; x < w; x++) {
    	int ze = (x == w - 1)? r - d: r;
    	for (int z = 1; z < ze; z++) {

    		// get the sent data-blocks
    		// copy blocks which need to be sent at this step
    		s = MPI_Wtime();
    		di = 0;
    		ci = 0;
    		for (int i = 0; i < nprocs; i++) {
    			if (rank_r_reps[i*w + x] == z){
    				sent_blocks[di++] = i;
    				memcpy(&temp_buffer[unit_size*ci++], &recvbuf[i*unit_size], unit_size);
    			}
    		}
    		nblocks_perstep[istep++] = di;
    		total_comm_steps += di;
    		e = MPI_Wtime();
    		pre_time += e - s;

    		// send and receive
    		s = MPI_Wtime();
    		int distance = z * pow(r, x);
    		int recv_proc = (rank - distance + nprocs) % nprocs; // receive data from rank - 2^step process
    		int send_proc = (rank + distance) % nprocs; // send data from rank + 2^k process
    		long long comm_size = di * unit_size;
    		MPI_Sendrecv(temp_buffer, comm_size, MPI_CHAR, send_proc, 0, sendbuf, comm_size, MPI_CHAR, recv_proc, 0, comm, MPI_STATUS_IGNORE);
    		e = MPI_Wtime();
    		comm_time += e - s;

    		s = MPI_Wtime();
    		// replace with received data
    		for (int i = 0; i < di; i++)
    		{
    			long long offset = sent_blocks[i] * unit_size;
    			memcpy(recvbuf+offset, sendbuf+(i*unit_size), unit_size);
    		}
    		e = MPI_Wtime();
    		replace_time += e - s;
    	}
    }

    free(rank_r_reps);
    free(temp_buffer);

    // local rotation
    s = MPI_Wtime();
	for (int i = 0; i < nprocs; i++)
	{
		int index = (rank - i + nprocs) % nprocs;
		memcpy(&sendbuf[index*unit_size], &recvbuf[i*unit_size], unit_size);
	}
	memcpy(recvbuf, sendbuf, nprocs*unit_size);
	e = MPI_Wtime();
	double second_time = e - s;

    double te = MPI_Wtime();
	double total_time = te - ts;

	timelist[it][0] = total_time;
	timelist[it][1] = first_time;
	timelist[it][2] = conv_time;
	timelist[it][3] = pre_time;
	timelist[it][4] = comm_time;
	timelist[it][5] = replace_time;
	timelist[it][6] = second_time;


	if (it % 20 == 0 && rank == 0 && sendcount == 2) {
		std::cout << "UniformRbruck-Metadata: " << nprocs << " " << sendcount << " " << r << " " << istep << " " << total_comm_steps << " [ ";

		for (int i = 0; i < istep; i++) {
			std::cout << nblocks_perstep[i] << " ";
		}
		std::cout << "]" << std::endl;
	}
			//[total_time, first_time, conv_time, pre_time, comm_time, replace_time, second_time];
//	MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, comm);
//	totaltime[i] = total_time;

//	if (total_time == max_time) {
//		std::cout << "[UniformRbruck] " << " [" << nprocs << " " << sendcount << " " << r << "] " <<  total_time << ", " << first_time << ", " << conv_time << ", "
//				<< pre_time << ", " << comm_time << ", " << replace_time << ", " << second_time  << " " << istep << " "<< total_comm_steps << std::endl;
//

//	}
}
