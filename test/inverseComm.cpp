/*
 * inverseComm.cpp
 *
 *  Created on: Aug 23, 2022
 *      Author: kokofan
 */
#include "../brucks.h"

static int rank, nprocs;

void running_test(int loopCount, int iteCount, int warmup);
void exchange_ascending(int loopCount, int mesgsize, char* sendbuf, char* recvbuf);
void exchange_descending(int loopCount, int mesgsize, char* sendbuf, char* recvbuf);

int main(int argc, char **argv)
{
//    if (argc < 2) {
//    	std::cout << "Usage: mpirun -n <nprocs>" << argv[0] << "<loopCount>" << std::endl;
//    }

//    int loopCount = atoi(argv[1]);

    // MPI Initial
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        std::cout << "ERROR: MPI_Init error\n" << std::endl;
    if (MPI_Comm_size(MPI_COMM_WORLD, &nprocs) != MPI_SUCCESS)
    	std::cout << "ERROR: MPI_Comm_size error\n" << std::endl;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
    	std::cout << "ERROR: MPI_Comm_rank error\n" << std::endl;

    int loopCount = ceil(log2(nprocs));

    // for warm-up only
    running_test(loopCount, 4, 1);

    // running test
    running_test(loopCount, 4, 0);

	MPI_Finalize();
    return 0;
}

void running_test(int loopCount, int iteCount, int warmup) {

    int mesgsize = 128;

    char * sendbuf = (char*)malloc(mesgsize*sizeof(char));
	for (int i = 0; i < mesgsize; i++)
		sendbuf[i] = rank;
	char * recvbuf = (char*)malloc(mesgsize*sizeof(char));

	for (int i = 0; i < iteCount; i++) {
		double start = MPI_Wtime();
		exchange_ascending(loopCount, mesgsize, sendbuf, recvbuf);
		double end = MPI_Wtime();
		double time = end - start;

		if (warmup == 0) {
			double max_time = 0;
			MPI_Allreduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			if (time == max_time)
				std::cout << "Ascending " << nprocs << " " << mesgsize << " " << time << std::endl;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	for (int i = 0; i < iteCount; i++) {
		double start = MPI_Wtime();
		exchange_descending(loopCount, mesgsize, sendbuf, recvbuf);
		double end = MPI_Wtime();
		double time = end - start;

		if (warmup == 0) {
			double max_time = 0;
			MPI_Allreduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			if (time == max_time)
				std::cout << "Descending " << nprocs << " " << mesgsize << " " << time << std::endl;
		}
	}

	free(sendbuf);
	free(recvbuf);
}

void exchange_ascending(int loopCount, int mesgsize, char* sendbuf, char* recvbuf) {

	int distance = 1;
	for (int i = 0; i < loopCount; i++) {
		int sendrank = (rank + distance) % nprocs;
		int recvrank = (rank - distance + nprocs) % nprocs;

		MPI_Sendrecv(sendbuf, mesgsize, MPI_CHAR, sendrank, 0, recvbuf, mesgsize, MPI_CHAR, recvrank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		distance *= 2;
	}
}

void exchange_descending(int loopCount, int mesgsize, char* sendbuf, char* recvbuf) {
	int distance = pow(2, loopCount-1);
	for (int i = 0; i < loopCount; i++) {
		int sendrank = (rank + distance) % nprocs;
		int recvrank = (rank - distance + nprocs) % nprocs;

		MPI_Sendrecv(sendbuf, mesgsize, MPI_CHAR, sendrank, 0, recvbuf, mesgsize, MPI_CHAR, recvrank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		distance /= 2;
	}
}


