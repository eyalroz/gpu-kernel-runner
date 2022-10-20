#include "buffer_io.hpp"

namespace util {

void verify_path(const filesystem::path& path, path_check_kind check_kind, bool allow_overwrite)
{
    // TODO: Consider checking path execution permissions down to the parent directory
    if (filesystem::exists(path)) {
        if (is_directory(path)) {
            throw std::invalid_argument("Path is a directory, not a (readable) path: " + path.native());
        }
        if (check_kind == for_writing and not allow_overwrite) {
            throw std::invalid_argument("File already exists, and overwrite is not allowed: " + path.native());
        }
        if (not has_permission(path, check_kind))
        {
            throw std::invalid_argument("No read permissions for path: " + path.native());
        }
    }
    else {
        if (check_kind != for_writing)
        {
            throw std::invalid_argument("File does not exist: " + path.native());
        }
        auto parent = path.parent_path();
        if (not parent.empty() and not is_directory(parent)) {
            throw std::invalid_argument("Parent path of intended file is not a directory: " + path.native());
        }
        if (parent.empty()) {
            parent = filesystem::current_path();
        }
        if (not has_permission(parent, for_writing))
        {
            throw std::invalid_argument("No write permissions for directory " + parent.native()
                                        + " , where it is necessary to write the new file " + path.filename().native());
        }
    }
}

inline void verify_input_path(const filesystem::path& path)
{
    return verify_path(path, for_reading, false);
}

std::vector<byte_type> read_input_file(const filesystem::path& src, size_t extra_buffer_size)
{
    verify_input_path(src);
    auto file_size = filesystem::file_size(src);
    auto buffer_size = file_size + extra_buffer_size;
    std::ifstream file(src, std::ios::binary | std::ios::ate);
    try {
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file.seekg(0, std::ios::beg);
        std::vector<byte_type> result(buffer_size);
        file.read(result.data(), (std::streamsize) file_size);
        return result;
    } catch (std::ios_base::failure& ios_failure) {
        if (errno == 0) {
            throw ios_failure;
        }
        throw std::system_error(errno, std::generic_category(),
            "trying to read " + std::to_string(file_size) + " from file " + src.native());
    }
}

std::vector<byte_type> read_file_as_null_terminated_string(const filesystem::path& source)
{
    size_t add_one_extra_byte { 1 };
    auto buffer = read_input_file(source, add_one_extra_byte);
    buffer.back() = '\0';
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
