/*
 * bruck.h
 *
 *      Author: kokofan
 */

#ifndef UNIFORM_BRUCK_H_
#define UNIFORM_BRUCK_H_

#include "../brucks.h"

void basic_bruck_uniform_benchmark(char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
void datatype_bruck_uniform_benchmark(char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm);
void modified_basic_bruck_uniform_benchmark(char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm);
void modified_dt_bruck_uniform_benchmark(char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm);
void noRotation_bruck_uniform_benchmark(char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm);
void zerocopy_bruck_uniform_benchmark(char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm);
void zeroCopyRot_bruck_uniform_benchmark(char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm);

#endif /* UNIFORM_BRUCK_H_ */
