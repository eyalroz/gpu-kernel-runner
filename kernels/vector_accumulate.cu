#include <cstddef>
#include <cstdio>

__global__ void vectorAccumulate(
	unsigned char       * __restrict  A,
	unsigned char const * __restrict  B,
	size_t length)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    // if (i == 0) { printf("Thread 0 in the grid!\n"); }
    if (i < length) {
        A[i] += B[i] + A_LITTLE_EXTRA;
    }
}


