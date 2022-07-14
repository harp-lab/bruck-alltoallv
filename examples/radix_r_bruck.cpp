/*
 * radix_r_bruck.cpp
 *
 *  Created on: Jul 09, 2022
 *      Author: kokofan
 */

#include "radix_r_bruck.h"
#include <typeinfo>

#define ITERATION_COUNT 100

static int rank, nprocs;
static void run_radix_r_bruck(int nprocs, std::vector<int> bases);

int main(int argc, char **argv)
{
    // MPI Initial
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        std::cout << "ERROR: MPI_Init error\n" << std::endl;
    if (MPI_Comm_size(MPI_COMM_WORLD, &nprocs) != MPI_SUCCESS)
    	std::cout << "ERROR: MPI_Comm_size error\n" << std::endl;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
    	std::cout << "ERROR: MPI_Comm_rank error\n" << std::endl;

    if (argc < 2) {
    	std::cout << "Usage: mpirun -n <nprocs> " << argv[0] << " <baselist>" << std::endl;
    	return -1;
    }

    std::vector<int> bases;
    for (int i = 1; i < argc; i++)
    	bases.push_back(atoi(argv[i]));

    run_radix_r_bruck(nprocs, bases);

	MPI_Finalize();
    return 0;
}

static void run_radix_r_bruck(int nprocs, std::vector<int> bases)
{
	int basecount = bases.size();
	for (int n = 2; n <= 1024; n = n * 2)
	{
		long long* send_buffer = new long long[n*nprocs];
		for (int i=0; i<n*nprocs; i++)
		{
			long long value = i/n + rank * 10;
			send_buffer[i] = value;
		}

		long long* recv_buffer = new long long[n*nprocs];

		// MPI alltoall
		for (int it=0; it < ITERATION_COUNT; it++) {
			MPI_Alltoall((char*)send_buffer, n, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, n, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0)
			std::cout << "----------------------------------------------------------------" << std::endl<< std::endl;

		// radix-r bruck
		double total_time_lists[ITERATION_COUNT*basecount][7];
		for (int i = 0; i < basecount; i++) {
			for (int it=0; it < ITERATION_COUNT; it++) {
				uniform_radix_r_bruck(total_time_lists, i*ITERATION_COUNT+it, bases[i], (char*)send_buffer, n, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, n, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
			}
		}

		for (int i = 0; i < basecount; i++) {
			for (int it=0; it < ITERATION_COUNT; it++) {
				double max_time = 0;
				MPI_Allreduce(&total_time_lists[i*ITERATION_COUNT + it][0], &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

				if (total_time_lists[i*ITERATION_COUNT + it][0] == max_time) {
					std::cout << "[UniformRbruck] " << nprocs << ", " << n << ", " << bases[i] << ", ";
					for (int t = 0; t < 7; t++)
						std::cout << total_time_lists[i*ITERATION_COUNT + it][t] << ", ";
					std::cout << std::endl;
				}
			}
		}


		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0)
			std::cout << "----------------------------------------------------------------" << std::endl<< std::endl;


		double total_times[ITERATION_COUNT];
		for (int it=0; it < ITERATION_COUNT; it++) {
			double comm_start = MPI_Wtime();
			MPI_Alltoall((char*)send_buffer, n, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, n, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
			double comm_end = MPI_Wtime();
			total_times[it] = comm_end - comm_start;
		}

		for (int it=0; it < ITERATION_COUNT; it++) {
			double max_time = 0;
			MPI_Allreduce(&total_times[it], &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			if (total_times[it] == max_time)
				std::cout << "[MPIAlltoall] " << nprocs << ", " << n << ", " <<  max_time << std::endl;
		}

		delete[] send_buffer;
		delete[] recv_buffer;
	}

}



