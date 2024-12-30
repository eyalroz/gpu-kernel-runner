// The code in this file was extracted and adapted from the
// NVIDIA jitify library sources, at:
// https://github.com/NVIDIA/jitify/
//
// And is made available under the terms of the BSD 3-Clause license:
// https://github.com/NVIDIA/jitify/blob/master/LICENSE

#ifndef KERNEL_RUNNER_STANDARD_HEADER_SUBSTITUTES_HPP_
#define KERNEL_RUNNER_STANDARD_HEADER_SUBSTITUTES_HPP_

#include <utility>
#include <vector>

std::vector<std::pair<const char*, const char*>> const& get_standard_header_substitutes();

#endif // KERNEL_RUNNER_STANDARD_HEADER_SUBSTITUTES_HPP_