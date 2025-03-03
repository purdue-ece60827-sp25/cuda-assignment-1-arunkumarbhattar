/**
 * @file lab1.cuh
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2021-01-16
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#pragma once

#ifndef LAB1_CUH
#define LAB1_CUH

	// #define DEBUG_PRINT_DISABLE
	
	#define VECTOR_SIZE (1 << 15)

	#define MC_SAMPLE_SIZE     1e7             // 10 million samples per thread
	#define MC_ITER_COUNT      32

	#define WARP_SIZE          32
	#define SAMPLE_SIZE        MC_SAMPLE_SIZE
	#define GENERATE_BLOCKS    65536           // Total number of generation threads
	#define REDUCE_SIZE        1024             // Each reduction block reduces 64 elements
	#define REDUCE_BLOCKS      (GENERATE_BLOCKS / REDUCE_SIZE)  // 65536 / 256 = 256


#endif