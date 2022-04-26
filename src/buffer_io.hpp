#ifndef BUFFER_IO_HPP_
#define BUFFER_IO_HPP_

#include <common_types.hpp>

host_buffer_type read_input_file(filesystem::path src, size_t extra_buffer_size = 0);
host_buffer_type read_file_as_null_terminated_string(const filesystem::path& source);
void write_buffer_to_file(const std::string& buffer_name, const host_buffer_type& buffer, filesystem::path destination);

#endif /* BUFFER_IO_HPP_ */
