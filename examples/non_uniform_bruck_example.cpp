/*
 * non_uniform_bruck_example.cpp
 *
 *      Author: kokofan
 */

#include "non_uniform_bruck.h"

#define ITERATION_COUNT 20

static int rank, nprocs;
void run_non_uniform(int nprocs, int dist);

int main(int argc, char **argv)
{
    // MPI Initial
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        std::cout << "ERROR: MPI_Init error\n" << std::endl;
    if (MPI_Comm_size(MPI_COMM_WORLD, &nprocs) != MPI_SUCCESS)
    	std::cout << "ERROR: MPI_Comm_size error\n" << std::endl;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
    	std::cout << "ERROR: MPI_Comm_rank error\n" << std::endl;

    run_non_uniform(nprocs, 0);

	MPI_Finalize();
    return 0;
}


void run_non_uniform(int nprocs, int dist)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int range = 100;
    int random_offset = 100 - range;

	for (int entry_count=2; entry_count <= 512; entry_count=entry_count*2)
	{
		int sendcounts[nprocs]; // the size of data each process send to other process
		memset(sendcounts, 0, nprocs*sizeof(int));
		int sdispls[nprocs];
		int soffset = 0;

		// Uniform random distribution
		if (dist == 0)
		{
			srand(time(NULL));
			for (int i=0; i < nprocs; i++)
			{
				int random = random_offset + rand() % range;
				sendcounts[i] = (entry_count * random) / 100;
			}
		}

		// Gausian normal distribution
		if (dist == 1)
		{
			std::default_random_engine generator;
			std::normal_distribution<double> distribution(nprocs/2, nprocs/3); // set mean and deviation

			while(true)
			{
				int p = int(distribution(generator));
				if (p >= 0 && p < nprocs)
				{
					if (++sendcounts[p] >= entry_count) break;
				}
			}
		}

		// Power law distribution
		if (dist == 2)
		{
			double x = (double)entry_count;

			for (int i=0; i<nprocs; ++i)
			{
				sendcounts[i] = (int)x;
				x = x * 0.999;
			}
		}

		// Random shuffling the sentcounts array
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::shuffle(&sendcounts[0], &sendcounts[nprocs], std::default_random_engine(seed));


		// Initial send offset array
		for (int i=0; i<nprocs; ++i)
		{
			sdispls[i] = soffset;
			soffset += sendcounts[i];
		}

		// Initial receive counts and offset array
		int recvcounts[nprocs];
		MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
		int rdispls[nprocs];
		int roffset = 0;
		for (int i=0; i < nprocs; i++)
		{
			rdispls[i] = roffset;
			roffset += recvcounts[i];
		}

		// Initial send buffer
		long long* send_buffer = new long long[soffset];
		long long* recv_buffer = new long long[roffset];

		int scounts[nprocs]; // a copy of sendcounts for each iteration

		// MPI_alltoallv
		for (int it=0; it < ITERATION_COUNT; it++)
		{
			int index = 0;
			for (int i=0; i < nprocs; i++)
			{
				for (int j = 0; j < sendcounts[i]; j++)
					send_buffer[index++] = i + rank * 10;
			}

			double comm_start = MPI_Wtime();
			MPI_Alltoallv(send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG, recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
			double comm_end = MPI_Wtime();

			double max_time = 0;
			double total_time = comm_end - comm_start;
			MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			if (total_time == max_time)
				std::cout << "[MPIAlltoallv]" << " [" << dist << " " << nprocs << " " << range << " " << entry_count << "] "<<  max_time << std::endl;
		}

		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0)
			std::cout << "----------------------------------------------------------------" << std::endl<< std::endl;

		// Padded All-to-all algorithm
		for (int it=0; it < ITERATION_COUNT; it++)
		{
			int index = 0;
			for (int i=0; i < nprocs; i++)
			{
				for (int j = 0; j < sendcounts[i]; j++)
					send_buffer[index++] = i + rank * 10;
			}
			padded_alltoall_non_uniform_benchmark(dist, 0, (char*)send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0)
			std::cout << "----------------------------------------------------------------" << std::endl<< std::endl;

		// Padded Bruck algorithm
		for (int it=0; it < ITERATION_COUNT; it++)
		{
			int index = 0;
			for (int i=0; i < nprocs; i++)
			{
				for (int j = 0; j < sendcounts[i]; j++)
					send_buffer[index++] = i + rank * 10;
			}
			padded_bruck_non_uniform_benchmark(dist, 0, (char*)send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0)
			std::cout << "----------------------------------------------------------------" << std::endl<< std::endl;


		// two-phase algorithm
		for (int it=0; it < ITERATION_COUNT; it++)
		{
			memcpy(&scounts, &sendcounts, nprocs*sizeof(int));

			int index = 0;
			for (int i=0; i < nprocs; i++)
			{
				for (int j = 0; j < sendcounts[i]; j++)
					send_buffer[index++] = i + rank * 10;
			}
			twophase_non_uniform_benchmark(dist, 0, (char*)send_buffer, scounts, sdispls, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0)
			std::cout << "----------------------------------------------------------------" << std::endl<< std::endl;

		delete[] send_buffer;
		delete[] recv_buffer;

	}
}
