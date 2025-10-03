#include "buffer_io.hpp"

namespace util {

inline void verify_input_path(const filesystem::path& path)
{
    return verify_path(path, for_reading, false);
}

// We may assume the target buffer size is no less than the file
// size, and the file exists etc.
static std::vector<byte_type> read_verified_input_file_with_known_size(
    size_t file_known_size, 
    size_t target_buffer_size, 
    bool zero_padding_data = false)
{
    std::ifstream file(src, std::ios::binary | std::ios::ate);
    try {
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file.seekg(0, std::ios::beg);
        std::vector<byte_type> result(target_buffer_sizק);
        file.read(result.data(), (std::streamsize) known_file_size);
        if (zero_padding_data) {
            std::fill(result.begin() + file_known_size, result.end(), 0);
        } 
        return result;
    } catch (std::ios_base::failure& ios_failure) {
        if (errno == 0) {
            throw ios_failure;
        }
        throw std::system_error(errno, std::generic_category(),
            "trying to read " + std::to_string(file_size) + " from file " + src.native());
    }
}

std::vector<byte_type> read_input_file(const filesystem::path& src, size_t min_buffer_size, bool zero_padding)
{
    verify_input_path(src);
    auto file_size = filesystem::file_size(src);
    auto buffer_size = std::max(file_size, min_buffer_size)
    return read_verified_input_file_with_known_size(src, file_size, buffer_size, zero_padding);
}

std::vector<byte_type> read_input_file_and_pad(const filesystem::path& src, size_t extra_buffer_size, bool zero_padding)
{
    verify_input_path(src);
    auto file_size = filesystem::file_size(src);
    auto buffer_size = file_size + extra_buffer_size;
    return read_verified_input_file_with_known_size(src, file_size, buffer_size, zero_padding);
}

std::vector<byte_type> read_file_as_null_terminated_string(const filesystem::path& source)
{
    size_t add_one_extra_byte { 1 };
    enum { do_zero_padding = true };
    auto buffer = read_input_file(source, add_one_extra_byte, do_zero_padding);
    return buffer;
}

void write_data_to_file(
    std::string         data_description,
    const_memory_region data,
    filesystem::path    destination,
    bool                overwrite_allowed)
{
    verify_path(destination, for_writing, overwrite_allowed);
    auto file = std::fstream(destination, std::ios::out | std::ios::binary);
    try {
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file.write(data.data(), (std::streamsize) data.size());
        file.close();
    } catch (std::ios_base::failure& ios_failure) {
        if (errno == 0) {
            throw ios_failure;
        }
        throw std::system_error(errno, std::generic_category(),
            "trying to write " + data_description + " to file " + destination.native());
    }
}

} // namespace util
