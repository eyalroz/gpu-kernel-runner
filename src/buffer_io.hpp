#ifndef BUFFER_IO_HPP_
#define BUFFER_IO_HPP_

#include <common_types.hpp>
#include <spdlog/common.h>


void verify_path(const filesystem::path& path, path_check_kind check_kind, bool allow_overwrite);
host_buffer_type read_input_file(const filesystem::path& src, size_t extra_buffer_size = 0);
host_buffer_type read_file_as_null_terminated_string(const filesystem::path& source);
void write_data_to_file(
    std::string kind,
    std::string name,
    poor_mans_span data,
    filesystem::path destination,
    bool overwrite_allowed,
    spdlog::level::level_enum level);
void write_buffer_to_file(
    std::string buffer_name,
    const host_buffer_type& buffer,
    filesystem::path destination,
    bool overwrite_allowed);

inline void verify_input_path(const filesystem::path& path)
{
    return verify_path(path, for_reading, false);
}

#endif /* BUFFER_IO_HPP_ */
