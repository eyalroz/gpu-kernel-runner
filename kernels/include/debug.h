/**
 * @file
 *
 * @copyright (c) 2024 Eyal Rozenberg <eyalroz1@gmx.com>.
 * @copyright (c) 2024 GE Healthcare.
 *
 * @brief OpenCL-and-CUDA-compatible definitions supporting
 * kernel debugging activities
 *
 * @license BSD 3-Clause license:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef KERNEL_DEBUGGING_OPENCL_AND_CUDA_H_
#define KERNEL_DEBUGGING_OPENCL_AND_CUDA_H_

#if defined(__CDT_PARSER__) || defined (__JETBRAINS_IDE__)
#include "opencl_syntax_for_ide_parser.cl.h"
#include "cuda_syntax_for_ide_parser.cuh"
#endif

#include "port_from_opencl.cuh"
#include "port_from_cuda.cl.h"


inline bool am_first_thread_in_grid() {
    return
        (get_global_id(0) == 0) &&
        (get_global_id(1) == 0) &&
        (get_global_id(2) == 0);
}

inline bool am_first_workitem_in_ndrange() { return am_first_thread_in_grid(); }


#endif // KERNEL_DEBUGGING_OPENCL_AND_CUDA_H_
