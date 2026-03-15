/**
 * @file
 *
 * @brief OpenCL-flavor definitions for porting OpenCL kernel code to CUDA
 * with fewer changes required.
 *
 * @copyright (c) 2020-2026, GE HealthCare
 * @copyright (c) 2020-2026, Eyal Rozenberg
 *
 * @license BSD 3-clause license; see the `LICENSE` file or
 * @url https://opensource.org/licenses/BSD-3-Clause
 *
 * @note Can be used for writing kernels targeting both CUDA and OpenCL
 * at once (alongside @ref port_from_cuda.cl.h ). To do this, the kernel code
 * must follow the following conventions:
 *
 *  | Instead of                | Use                                                   | Explanation/Note                              |
 *  |:--------------------------|:------------------------------------------------------|:----------------------------------------------|
 *  | `__local` / `__shared`    | `__local_array` , `__local_variable` or `__local_ptr` | Let a macro sort out the memory space marking |
 *  | `max(x,y)`                | `fmax(x,y)`                                           | it's too risky to define a `max(x,y)` macro   |
 *  | struct foo = { 12, 3.4 }; | struct foo = make_compound(foo){ 12, 3.4; }           | Allow for different construction syntax       |
 *  | constexpr                 | either CONSTEXPR_OR_CONSTANT_MEM, or an enum          |                                               |
 *  | vector type literals      | definition of a vector type variable, then assignment | We could implement vector-type named          |
 *  |                           |                                                       | constructor idioms, but we haven't yet.       |
 *  | .r/g/b/a vector members   | .x/y/z/w vector members                               | The latter is what CUDA offers, and we can't  |
 *  |                           |                                                       | alias members easily without proper getters,  |
 *  |                           |                                                       | which C++ does not have.                      |
 *  | global, private, constant | __global, __private, __constant                       | ... or just avoid these where possible        |
 *  | (N/A)                     | Follow the restrictions in OpenCL spec §6.11          |                                               |
 *
 *  and
 *
 *  | Don't use                                  | Explanation/Note                              |
 *  |:-------------------------------------------|:----------------------------------------------|
 *  | __ROUNDING_MODE__                          | It's deprecated by OpenCL 1.1                 |
 *  | addressing multiple vector members at once | Just don't do it; address them individually   |
 *  | clang Blocks (as per §6.14 Blocks)         | Not supported in CUDA C++                     |
 *  | subgroups and related builtin functions    | Not supported in CUDA                         |
 *
 *
 * @note Use of dynamic shared memory is very different between OpenCL and CUDA, you'll
 * have to either avoid it or work the differences out yourself.
 */
#ifndef PORT_FROM_OPENCL_CUH_
#define PORT_FROM_OPENCL_CUH_

#include "port_from_opencl/opencl_defines.cuh"
#include "port_from_opencl/opencl_scalar_types.cuh"
#include "port_from_opencl/opencl_vector_types.cuh"
#include "port_from_opencl/opencl_builtin_functions.cuh"
    // including 6.4.3. Explicit Conversions

/**
 * TODO/Not yet implemented (by sections of
 * @url https://registry.khronos.org/OpenCL/specs/3.0-unified ):
 *
 * - Feature macros: 6.2.1
 * - Subgroup-related functions: 6.2.2
 * - standalone & member operators: 6.5
 */

#endif // PORT_FROM_OPENCL_CUH_
