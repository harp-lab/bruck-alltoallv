/*
 * pingpongcheck.cpp
 *
 *  Created on: Jul 28, 2022
 *      Author: kokofan
 */

#include "../brucks.h"

int ite_count = 100;

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

    for (int i = 0; i < 50; i++) {
		int send_data = rank;
		int receve_data = -1;

		if (rank == (nprocs-1))
			MPI_Recv(&receve_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (rank == 0)
			MPI_Send(&send_data, 1, MPI_INT, (nprocs-1), 0, MPI_COMM_WORLD);
    }

    std::vector<double> times;

    for (int i = 0; i < ite_count; i++) {
    	for (int count = 2; count < 2048; count *= 2) {

    		int send_data[count];
    		std::fill_n(send_data, count, rank);

			int receve_data[count];
			std::fill_n(receve_data, count, -1);

			double Start = MPI_Wtime();
			if (rank == (nprocs-1))
				MPI_Recv(&receve_data, count, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if (rank == 0)
				MPI_Send(&send_data, count, MPI_INT, (nprocs-1), 0, MPI_COMM_WORLD);
			double end = MPI_Wtime();
			double time = end - Start;
			times.push_back(time);
    	}
    }

    int j = 0;
    for (int i = 0; i < ite_count; i++) {
    	for (int count = 2; count < 2048; count *= 2) {

    		double max_time = 0;
			MPI_Allreduce(&times[j], &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

			if (times[j] == max_time)
				std::cout << rank << " " << count << " " << max_time << std::endl;
			j++;
    	}
	}


	MPI_Finalize();
    return 0;
}
