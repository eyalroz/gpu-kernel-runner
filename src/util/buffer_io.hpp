#ifndef BUFFER_IO_HPP_
#define BUFFER_IO_HPP_

#include "filesystem.hpp"
#include "memory_region.hpp"
#include <spdlog/common.h>

namespace util {

// TODO: I don't like returning vectors :-(

std::vector<byte_type> read_input_file(const filesystem::path& src, size_t min_buffer_size = 0, bool zero_padding = false);
std::vector<byte_type> read_input_file_and_pad(const filesystem::path& src, size_t extra_buffer_size, bool zero_padding = false);
std::vector<byte_type> read_file_as_null_terminated_string(const filesystem::path& source);

// Note: data_description is only used for generating an exception on failure
void write_data_to_file(
    std::string         data_description,
    const_memory_region data,
    filesystem::path    destination,
    bool                overwrite_allowed);

} // namespace util

#endif /* BUFFER_IO_HPP_ */
