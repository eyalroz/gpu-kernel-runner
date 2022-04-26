#include <buffer_io.hpp>
#include <spdlog/spdlog.h>

host_buffer_type read_input_file(filesystem::path src, size_t extra_buffer_size)
{
    auto file_size = filesystem::file_size(src);
    auto buffer_size = file_size + extra_buffer_size;
    std::ifstream file(src, std::ios::binary | std::ios::ate);
    file.seekg(0, std::ios::beg);

    host_buffer_type result(buffer_size);
    file.read(result.data(), file_size);
    if (file.fail()) {
        throw std::system_error(errno, std::generic_category());
    }
    return result;
}

host_buffer_type read_file_as_null_terminated_string(const filesystem::path& source)
{
    auto add_one_extra_byte { 1 };
    auto buffer = read_input_file(source, add_one_extra_byte);
    buffer.back() = '\0';
    return buffer;
}

void write_buffer_to_file(const std::string& buffer_name, const host_buffer_type& buffer, filesystem::path destination)
{
    spdlog::debug("Writing output buffer '{}' to file {}", buffer_name, destination.c_str());
    auto file = std::fstream(destination, std::ios::out | std::ios::binary);
    file.write(buffer.data(), buffer.size());
    if (file.fail()) {
        throw std::system_error(errno, std::generic_category(),
            "trying to write output buffer '" + buffer_name + "' to file " + destination.native());
    }
    file.close();
}
