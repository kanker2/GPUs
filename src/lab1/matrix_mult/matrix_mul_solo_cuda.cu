#include <stdio.h>
#include <sys/time.h>
#include "matrix_mul.h"

// Thread block size
#define BLOCK_SIZE 16 

double wtime(void)
{
        static struct timeval   tv0;
        double time_;

        gettimeofday(&tv0,(struct timezone*)0);
        time_=(double)((tv0.tv_usec + (tv0.tv_sec)*1000000));
        return( time_/1000000);
}

// Forward declaration of the device multiplication function
__global__ void Muld(float*A, float*B, int wA, int wB, float*C)
{
	//Asumiré que las dimensiones de C son múltiplos de BLOCK_SIZE
	int i = blockIdx.y * blockDim.y + threadIdx.y,
			j = blockIdx.x * blockDim.x + threadIdx.x;
	C[i*wB + j] = 0;
	for (int k = 0; k < wA; k++)
		C[i*wB + j] += A[i*wA + k] * B[k*wB + j];
}

// Host multiplication function
// Compute C = A * B
// hA is the height of A
// wA is the width of A
// wB is the width of B


void Mul___(float* A, float* B, int hA, int wA, int wB, float* C)
{
	double t0,t1;
	int size;

	// Load A and B to the device
	float* Ad;
	size = hA * wA * sizeof(float);
	cudaMalloc((void**)&Ad, size);
	cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
	float* Bd;
	size = wA * wB * sizeof(float);
	cudaMalloc((void**)&Bd, size);
	cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);

	// Allocate C on the device
	float* Cd;
	size = hA * wB * sizeof(float);
	cudaMalloc((void**)&Cd, size);

	// Compute the execution configuration assuming
	// the matrix dimensions are multiples of BLOCK_SIZE
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(wB / dimBlock.x, hA / dimBlock.y);

	// Launch the device computation
	t0 = wtime();
	Muld<<<dimGrid, dimBlock>>>(Ad, Bd, wA, wB, Cd);
	cudaDeviceSynchronize();
	t1 = wtime(); printf("Time GPU=%f\n", t1-t0);

	// Read C from the device
	cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(Ad);
	cudaFree(Bd);
	cudaFree(Cd);
}

#if 0
#endif
