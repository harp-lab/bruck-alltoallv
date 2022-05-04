/*
 * uniform_bruck_example.cpp
 *
 *      Author: kokofan
 */

#include "uniform_bruck.h"

#define ITERATION_COUNT 20

static int rank, nprocs;
static void run_uniform(int nprocs);

int main(int argc, char **argv)
{
    // MPI Initial
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        std::cout << "ERROR: MPI_Init error\n" << std::endl;
    if (MPI_Comm_size(MPI_COMM_WORLD, &nprocs) != MPI_SUCCESS)
    	std::cout << "ERROR: MPI_Comm_size error\n" << std::endl;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
    	std::cout << "ERROR: MPI_Comm_rank error\n" << std::endl;

	run_uniform(nprocs);

	MPI_Finalize();
    return 0;
}

static void run_uniform(int nprocs)
{

	// Initial send buffer
    for (long long entry_count=2; entry_count <= 1024; entry_count=entry_count*2)
    {
		long long* send_buffer = new long long[entry_count*nprocs];
		for (int i=0; i<entry_count*nprocs; i++)
		{
			long long value = i/entry_count + rank * 10;
			send_buffer[i] = value;
		}

		long long* recv_buffer = new long long[entry_count*nprocs];

		for (int it=0; it < ITERATION_COUNT; it++)
			basic_bruck_uniform_benchmark((char*)send_buffer, entry_count, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, entry_count, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0)
			std::cout << "----------------------------------------------------------------" << std::endl<< std::endl;

		for (int it=0; it < ITERATION_COUNT; it++)
			datatype_bruck_uniform_benchmark((char*)send_buffer, entry_count, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, entry_count, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0)
			std::cout << "----------------------------------------------------------------" << std::endl<< std::endl;

		for (int it=0; it < ITERATION_COUNT; it++)
			modified_basic_bruck_uniform_benchmark((char*)send_buffer, entry_count, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, entry_count, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0)
			std::cout << "----------------------------------------------------------------" << std::endl<< std::endl;

		for (int it=0; it < ITERATION_COUNT; it++)
			modified_dt_bruck_uniform_benchmark((char*)send_buffer, entry_count, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, entry_count, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0)
			std::cout << "----------------------------------------------------------------" << std::endl<< std::endl;

		for (int it=0; it < ITERATION_COUNT; it++)
			noRotation_bruck_uniform_benchmark((char*)send_buffer, entry_count, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, entry_count, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0)
			std::cout << "----------------------------------------------------------------" << std::endl<< std::endl;

		for (int it=0; it < ITERATION_COUNT; it++)
			zerocopy_bruck_uniform_benchmark((char*)send_buffer, entry_count, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, entry_count, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);


		delete[] send_buffer;
		delete[] recv_buffer;
    }
}
