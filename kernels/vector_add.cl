__kernel void vectorAdd(
   __global unsigned char       * __restrict C,
   __global unsigned char const * __restrict A,
   __global unsigned char const * __restrict B,
   unsigned long length)
{
   int i = get_global_id(0);
   if (i < length)
       C[i] = A[i] + B[i] + A_LITTLE_EXTRA;
}

