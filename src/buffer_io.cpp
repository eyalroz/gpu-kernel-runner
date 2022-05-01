#include <buffer_io.hpp>
#include <spdlog/spdlog.h>

host_buffer_type read_input_file(filesystem::path src, size_t extra_buffer_size)
{
    auto file_size = filesystem::file_size(src);
    auto buffer_size = file_size + extra_buffer_size;
    std::ifstream file(src, std::ios::binary | std::ios::ate);
    file.seekg(0, std::ios::beg);

    host_buffer_type result(buffer_size);
    file.read(result.data(), (std::streamsize) file_size);
    if (file.fail()) {
        throw std::system_error(errno, std::generic_category());
    }
    return result;
}

host_buffer_type read_file_as_null_terminated_string(const filesystem::path& source)
{
    size_t add_one_extra_byte { 1 };
    auto buffer = read_input_file(source, add_one_extra_byte);
    buffer.back() = '\0';
    return buffer;
}

void write_data_to_file(std::string kind, std::string name, poor_mans_span data, filesystem::path destination, spdlog::level::level_enum level)
{
    spdlog::log(level, "Writing {} '{}' to file {}", kind, name, destination.c_str());
    auto file = std::fstream(destination, std::ios::out | std::ios::binary);
    file.write(data.data(), (std::streamsize) data.size());
    if (file.fail()) {
        throw std::system_error(errno, std::generic_category(),
            "trying to write " + kind + "'" + name + "' to file " + destination.native());
    }
    file.close();
}

void write_buffer_to_file(std::string buffer_name, const host_buffer_type& buffer, filesystem::path destination)
{
    write_data_to_file("output buffer", buffer_name, poor_mans_span{const_cast<byte_type*>(buffer.data()),
        buffer.size()}, destination, spdlog::level::debug);
}
