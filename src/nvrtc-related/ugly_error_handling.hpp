#ifndef KERNEL_RUNNER_CUDA_UGLY_ERROR_HANDLING_HPP_
#define KERNEL_RUNNER_CUDA_UGLY_ERROR_HANDLING_HPP_

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <nvrtc.h>
#include <iostream>
#include <string>

#define cuda_api_call(_f, ...) cuda_api_call_( #_f , _f, ##__VA_ARGS__ )

// Avoiding warnings... with C++20, we could use VA_OPT
#define cuda_api_call_noargs(_f) cuda_api_call_( #_f , _f)

template <typename Function, typename... Args>
inline void cuda_api_call_(const char* api_call_name, Function f, Args&&... args)
{
    auto result = f(std::forward<Args>(args)...);

    constexpr const bool from_runtime = std::is_same<decltype(result), cudaError_t>::value;
    constexpr const bool from_driver  = std::is_same<decltype(result), CUresult   >::value;
    constexpr const bool from_nvrtc   = std::is_same<decltype(result), nvrtcResult>::value;

    static_assert(from_runtime or from_driver or from_nvrtc, "Unsupported function");

    if (from_runtime) {
        auto result_ = static_cast<cudaError_t>(result);
        if (result_ != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA Runtime API call failed with status ") + cudaGetErrorName(result_) + ": " + cudaGetErrorString(result_));
        }
    }
    else if (from_driver) {
        auto result_ = static_cast<CUresult>(result);
        if (result_ != CUDA_SUCCESS) {
            const char* error_description;
            const char* error_name;
            cuGetErrorName(result_, &error_name);
            cuGetErrorString(result_, &error_description);
            throw std::runtime_error(std::string("CUDA Driver API call ") + api_call_name + " failed: " + error_description);
        }
    }
    else { // from_nvrtc
        auto result_ = static_cast<nvrtcResult>(result);
        if (result_ != NVRTC_SUCCESS) {
            throw std::runtime_error(std::string("NVRTC call ") + api_call_name + " failed: " + nvrtcGetErrorString(result_));
        }
    }
}

#endif // KERNEL_RUNNER_CUDA_UGLY_ERROR_HANDLING_HPP_
