#ifndef BUFFER_IO_HPP_
#define BUFFER_IO_HPP_

#include "filesystem.hpp"
#include "memory_region.hpp"
#include <spdlog/common.h>

namespace util {

// TODO: The following should really return a dynarray (or take a memory region destination)
std::vector<byte_type> read_input_file(const filesystem::path &src, size_t extra_buffer_size = 0);

// TODO: If string could reuse an existing buffer, I'd return that. Or this could get a destination
std::vector<byte_type> read_file_as_null_terminated_string(const filesystem::path &source);

// Note: data_description is only used for generating an exception on failure
void write_data_to_file(
    std::string         data_description,
    const_memory_region data,
    filesystem::path    destination,
    bool                overwrite_allowed);

} // namespace util

#endif /* BUFFER_IO_HPP_ */