#ifndef KERNEL_RUNNER_OPENCL_UGLY_ERROR_HANDLING_HPP_
#define KERNEL_RUNNER_OPENCL_UGLY_ERROR_HANDLING_HPP_

#include <CL/cl_platform.h>

char const * clGetErrorString(cl_int const err);

#endif /* KERNEL_RUNNER_OPENCL_UGLY_ERROR_HANDLING_HPP_ */
