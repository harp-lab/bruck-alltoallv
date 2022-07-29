/*
 * pingpongcheck.cpp
 *
 *  Created on: Jul 28, 2022
 *      Author: kokofan
 */

#include "../brucks.h"

int main(int argc, char **argv)
{
	int rank, nprocs;
    // MPI Initial
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        std::cout << "ERROR: MPI_Init error\n" << std::endl;
    if (MPI_Comm_size(MPI_COMM_WORLD, &nprocs) != MPI_SUCCESS)
    	std::cout << "ERROR: MPI_Comm_size error\n" << std::endl;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
    	std::cout << "ERROR: MPI_Comm_rank error\n" << std::endl;

    int send_data = rank;
    int receve_data = -1;

    double Start = MPI_Wtime();
    if (rank == (nprocs-1))
    	MPI_Recv(&receve_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (rank == 0)
    	MPI_Send(&send_data, 1, MPI_INT, (nprocs-1), 0, MPI_COMM_WORLD);
    double end = MPI_Wtime();
    double time = end - Start;

    if (rank == 0 || rank == (nprocs-1))
    	std::cout << rank << " " << receve_data << " " << time << std::endl;


	MPI_Finalize();
    return 0;
}
