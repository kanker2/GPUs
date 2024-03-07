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

__global__ void Muld(float* A, float* B, int wA, int wB, float* C)
{
    // Declarar memoria compartida para bloques de A y B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Calcular índices globales e índices locales
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    float value = 0.0;

    // Iterar sobre bloques
    for (int m = 0; m < wA / BLOCK_SIZE; ++m) {
        // Cargar bloques de A y B en memoria compartida
        As[ty][tx] = A[row * wA + m * BLOCK_SIZE + tx];
        Bs[ty][tx] = B[(m * BLOCK_SIZE + ty) * wB + col];

        // Sincronizar para asegurar que todos los hilos han cargado los bloques
        __syncthreads();

        // Realizar la multiplicación en los bloques cargados
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            value += As[ty][k] * Bs[k][tx];
        }

        // Sincronizar para asegurar que la multiplicación ha terminado antes de cargar el siguiente bloque
        __syncthreads();
    }

    // Escribir el resultado en la matriz C
    C[row * wB + col] = value;
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
