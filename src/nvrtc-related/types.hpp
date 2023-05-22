#ifndef KERNEL_RUNNER_CUDA_TYPES_HPP_
#define KERNEL_RUNNER_CUDA_TYPES_HPP_

#include <cuda/api/event.hpp>
#include <utility>

struct cuda_event_pair_t {
    cuda::event_t before, after;
};

#endif // KERNEL_RUNNER_CUDA_TYPES_HPP_
