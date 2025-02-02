
#include "cudaLib.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//simply reading from global GPU memory
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		y[idx] = x[idx] * scale + y[idx];
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	//first we gon allocate host arrays
	float* h_x = new float[vectorSize];
	float* h_y = new float[vectorSize];
	float* h_y_verification = new float[vectorSize];

	std::srand(static_cast<unsigned int>(time(NULL)));

	//now we gon intialize the data
	for (int i = 0; i < vectorSize; i++) {
		h_x[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
		h_y[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
		h_y_verification[i] = h_y[i];
	}

	//random scaling factor
	float scale = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

	//allocate space on GDDR6 memory
	float* g_x = nullptr;
	float* g_y = nullptr;
	gpuErrchk(cudaMalloc((void**)&g_x, vectorSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&g_y, vectorSize*sizeof(float)));

	//copy from CPU into the GPU
	gpuErrchk(cudaMemcpy(g_x, h_x, vectorSize*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(g_y, h_y, vectorSize*sizeof(float), cudaMemcpyHostToDevice));

	//specify block size and other config
	int blockSize = 1024;
	int gridSize = (vectorSize + blockSize - 1) / blockSize;

	//run on CPU to generate ground truth
	saxpy_cpu(h_x, h_y_verification, scale, vectorSize);

	//now we gon launch the kernel
	saxpy_gpu<<<gridSize, blockSize>>>(g_x, g_y, scale, vectorSize);

	gpuErrchk(cudaDeviceSynchronize());

	//Copy the results back
	gpuErrchk(cudaMemcpy(h_y, g_y, vectorSize*sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaDeviceSynchronize());

	bool isCorrecto = true;
	for (int i = 0; i < vectorSize; i++) {
		if (fabs(h_y[i] - h_y_verification[i]) > 1e-5) {
			isCorrecto = false;
			break;
		}
	}

	std::cout <<" SAXPY GPU kernel verification: "<<isCorrecto<<std::endl;

	//free up
	gpuErrchk(cudaFree((void*)g_x));
	gpuErrchk(cudaFree((void*)g_y));

	delete[] h_x;
	delete[] h_y;
	delete[] h_y_verification;

	std::cout << "Not Lazy I am!\n";
	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//global thread index
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= pSumSize) {
		return;
	}

	curandState state;
	curand_init(1234ULL, tid, 0, &state);

	uint64_t count = 0;

	for (uint64_t i  = 0; i < sampleSize; i++) {
		//we gon generate random numbers between 0-1
		float x = curand_uniform(&state);
		float y = curand_uniform(&state);

		//map to [-1 1]
		x = 2.0f *x - 1.0f;
		y = 2.0f *y - 1.0f;

		if (x*x + y*y <= 1.0f) {
			count++;
		}
	}
	pSums[tid] = count;
}

__global__
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
/*
 * This kernel basically reduces a segment of reduceSize elements from the input array pSums into a single sum
 * Each threadblock gon handle one segment. The per-block reduction uses shared memory
 */
	uint64_t segmentStart = blockIdx.x * reduceSize;

	extern __shared__ uint64_t smem[];

	uint64_t sum = 0;
	//loop over all the elments in this segment
	for (uint64_t i = threadIdx.x; i < reduceSize && (segmentStart + i) < pSumSize; i += blockDim.x) {
		sum += pSums[segmentStart + i];
	}

	smem[threadIdx.x] = sum;

	__syncthreads();

	//we gon do a binary tree reduction in shared memory
	for (unsigned int s =blockDim.x/2; s>0; s>>=1) {
		if (threadIdx.x < s) {
			smem[threadIdx.x] += smem[threadIdx.x + s];
		}
		__syncthreads();
	}

	//first thread gon write the result fo this segment
	if (threadIdx.x == 0) {
		totals[blockIdx.x] = smem[0];
	}
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	double approximate_pi = 0;

    uint64_t *g_pSums = nullptr;
    cudaMalloc(&g_pSums, generateThreadCount * sizeof(uint64_t));

    int genBlockSize = 1024;
    int genGridSize = (generateThreadCount + genBlockSize - 1) / genBlockSize;
    generatePoints<<<genGridSize, genBlockSize>>>(g_pSums, generateThreadCount, sampleSize);
    cudaDeviceSynchronize();

    uint64_t totalSegments = (generateThreadCount + reduceSize - 1) / reduceSize;

    uint64_t *g_totals = nullptr;
    cudaMalloc(&g_totals, totalSegments * sizeof(uint64_t));

    reduceCounts<<<totalSegments, reduceThreadCount, reduceThreadCount * sizeof(uint64_t)>>>(
        g_pSums, g_totals, generateThreadCount, reduceSize);
    cudaDeviceSynchronize();

    // Copy the reduced totals back to the CPU -> host.
    uint64_t *h_totals = new uint64_t[totalSegments];
    cudaMemcpy(h_totals, g_totals, totalSegments * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Sum the totals on the CPU.
    uint64_t totalHits = 0;
    for (uint64_t i = 0; i < totalSegments; i++) {
        totalHits += h_totals[i];
    }
    delete[] h_totals;

    cudaFree(g_pSums);
    cudaFree(g_totals);

    uint64_t totalPoints = generateThreadCount * sampleSize;

    // We gon approximate Pi as 4 * (hits / total points).
    approximate_pi = 4.0 * static_cast<double>(totalHits) / static_cast<double>(totalPoints);

    return approximate_pi;
}
