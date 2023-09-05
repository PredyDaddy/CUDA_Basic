#ifndef MATMUL_GPU_BASIC_H
#define MATMUL_GPU_BASIC_H

// CUDA runtime
#include "cuda_runtime.h"
#include "cuda.h"

// Function Declarations

/**
 * CUDA kernel for matrix multiplication.
 * 
 * @param M_device Pointer to the first matrix on the device.
 * @param N_device Pointer to the second matrix on the device.
 * @param P_device Pointer to the output matrix on the device.
 * @param width The width (and height, assuming square matrices) of the matrices.
 */
__global__ void MatmulKernel(float *M_device, float *N_device, float *P_device, int width);

/**
 * Host function to multiply two matrices on the device.
 * 
 * @param M_host Pointer to the first matrix on the host.
 * @param N_host Pointer to the second matrix on the host.
 * @param P_host Pointer to the output matrix on the host.
 * @param width The width (and height, assuming square matrices) of the matrices.
 * @param blockSize The size of the CUDA block.
 */
void MatmulOnDevice(float *M_host, float *N_host, float* P_host, int width, int blockSize);

#endif // MATMUL_GPU_BASIC_H
