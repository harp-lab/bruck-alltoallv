/*
 * radix_r_bruck.cpp
 *
 *  Created on: Jul 09, 2022
 *      Author: kokofan
 */

#include "radix_r_bruck.h"

#define ITERATION_COUNT 1

static int rank, nprocs;
static void run_radix_r_bruck(int nprocs, int r);

int main(int argc, char **argv)
{
    // MPI Initial
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        std::cout << "ERROR: MPI_Init error\n" << std::endl;
    if (MPI_Comm_size(MPI_COMM_WORLD, &nprocs) != MPI_SUCCESS)
    	std::cout << "ERROR: MPI_Comm_size error\n" << std::endl;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
    	std::cout << "ERROR: MPI_Comm_rank error\n" << std::endl;

    if (argc != 2) {
    	std::cout << "Usage: mpirun -n <nprocs> " << argv[0] << " <base>" << std::endl;
    	return -1;
    }

    int r = atoi(argv[1]);

    run_radix_r_bruck(nprocs, r);

	MPI_Finalize();
    return 0;
}

static void run_radix_r_bruck(int nprocs, int r)
{
	for (int n = 8; n <= 8; n = n * 2)
	{
		long long* send_buffer = new long long[n*nprocs];
		for (int i=0; i<n*nprocs; i++)
		{
			long long value = i/n + rank * 10;
			send_buffer[i] = value;
		}

		long long* recv_buffer = new long long[n*nprocs];

		for (int it=0; it < ITERATION_COUNT; it++)
			uniform_radix_r_bruck(r, (char*)send_buffer, n, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, n, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

		delete[] send_buffer;
		delete[] recv_buffer;
	}

}



