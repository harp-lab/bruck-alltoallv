/*
 * r_radix_bruck.h
 *
 *  Created on: Jul 09, 2022
 *      Author: kokofan
 */

#ifndef SRC_RADIX_R_BRUCK_H_
#define SRC_RADIX_R_BRUCK_H_

#include "../brucks.h"

void uniform_radix_r_bruck(double timelist[][7], int it, int r, char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm);



#endif /* SRC_RADIX_R_BRUCK_H_ */
