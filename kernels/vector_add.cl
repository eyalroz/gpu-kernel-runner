__kernel void vectorAdd(
   __global unsigned char       * __restrict c,
   __global unsigned char const * __restrict a,
   __global unsigned char const * __restrict b,
   unsigned int length)
{
   int i = get_global_id(0);
   if (i < length)
       c[i] = a[i] + b[i] + A_LITTLE_EXTRA;
}

