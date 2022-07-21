/*
 * radix_r_bruck.cpp
 *
 *  Created on: Jul 09, 2022
 *      Author: kokofan
 */

#include "radix_r_bruck.h"
#include <typeinfo>

#define ITERATION_COUNT 1

static int rank, nprocs;

static void calculate_commsteps_and_datablock_counts(int r, std::vector<int>& the_sd_pstep);
static void run_radix_r_bruck(int nprocs, int r, std::vector<int>& act_sd_pstep);


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

    for ( int r = 2; r < nprocs; r++) {

    	std::vector<int> the_sd_pstep;
        std::vector<int> act_sd_pstep;

//    	calculate_commsteps_and_datablock_counts(r, the_sd_pstep);
//        int r = 51;
    	run_radix_r_bruck(nprocs, r, act_sd_pstep);

//    	if (rank == 0) {
//    		if (the_sd_pstep.size() !=  act_sd_pstep.size()) {
//    			std::cout << "ERROR: Incorrect formulas!" << std::endl;
//    		}
//    		else {
//    			std::cout << nprocs << " " << r << " " <<the_sd_pstep.size() << " [";
//    			int total_dc = 0;
//    			for (int s = 0; s < the_sd_pstep.size(); s++) {
//    				if (the_sd_pstep[s] != act_sd_pstep[s])
//    					std::cout << "ERROR: Incorrect formulas: " << s << ", (THE) " << the_sd_pstep[s] << ", (ACT) " << act_sd_pstep[s] << std::endl;
//    				else {
//    					std::cout << act_sd_pstep[s] << " ";
//    					total_dc += act_sd_pstep[s];
//    				}
//    			}
//				std::cout << "]" << total_dc << std::endl;
//    		}
//    	}
    }

	MPI_Finalize();
    return 0;
}

static void run_radix_r_bruck(int nprocs, int r, std::vector<int>& act_sd_pstep)
{
	for (int n = 1; n <= 1; n = n * 2)
	{
		long long* send_buffer = new long long[n*nprocs];
		long long* recv_buffer = new long long[n*nprocs];

		for (int it=0; it < ITERATION_COUNT; it++) {

			for (int p=0; p<n*nprocs; p++) {
				long long value = p/n + rank * 10;
				send_buffer[p] = value;
			}
			memset(recv_buffer, 0, n*nprocs*sizeof(long long));

			uniform_radix_r_bruck(act_sd_pstep, r, (char*)send_buffer, n, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, n, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

			// check if correct
			if (rank == 0){
				int error = 0;
				for (int d = 0; d < n*nprocs; d++) {
					if ( (recv_buffer[d] % 10) != (rank % 10) ) {
						error += 1;
//						std::cout << "EROOR VALUE: " << rank << " " << r << " " << d << " " << recv_buffer[d] << std::endl;
					}
				}
				if (error > 0) {
					std::cout << "R ERROR " << r << " " << error << std::endl;
//					for (int d = 0; d < n*nprocs; d++) {
//						std::cout << recv_buffer[d] << " ";
//					}
//					std::cout << std::endl;
				}
				else {
					std::cout << "R Correct " << r << std::endl;
				}
			}

		}

		delete[] send_buffer;
		delete[] recv_buffer;
	}

}


static void calculate_commsteps_and_datablock_counts(int r, std::vector<int>& the_sd_pstep) {

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	int w = ceil(log(nprocs) / log(r)); // calculate the number of digits when using r-representation
	int nlpow = pow(r, w-1);
	int d = (pow(r, w) - nprocs) / nlpow; // calculate the number of highest digits

	for (int x = 0; x < w; x++) {
		int ze = (x == w - 1)? r - d: r;
		for (int z = 1; z < ze; z++) {

			int xhpow = pow(r, x+1);
			int xpow = pow(r, x);
			int div = floor(nprocs / xhpow );
			int re = nprocs % xhpow;
			int t  = re - z * xpow;

			int dc = div * xpow; // number of sent data-blocks per step
			if (t > 0) {
				if (t / xpow > 0) { dc += xpow; }
				else { dc += t % xpow; }
			}

			the_sd_pstep.push_back(dc);
		}
	}
}



