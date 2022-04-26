#ifndef UTIL_FILESYSTEM_HPP_
#define UTIL_FILESYSTEM_HPP_

#include <fstream>

static_assert(__cplusplus >= 201402L, "C++2014 is required to compile this program");

#if __cplusplus == 201402L
#include <experimental/filesystem>
namespace filesystem = std::experimental::filesystem;
#elif __cplusplus >= 201703L
#include <filesystem>
namespace filesystem = std::filesystem;
#endif

inline filesystem::path maybe_prepend_base_dir(
    const filesystem::path& prefix,
    const filesystem::path& suffix_or_absolute_path)
{
    bool can_prepend_buffer_dir = (prefix != filesystem::path{}) and (suffix_or_absolute_path.is_relative());
    return can_prepend_buffer_dir ? (prefix / suffix_or_absolute_path) : suffix_or_absolute_path;
}


// We could theoretically support C++11 with boost filesystem
// #include <boost/filesystem.hpp> // Can't use std::filesystem before C++17

/*
inline void create_sized_file(filesystem::path path, size_t size)
{
    if (filesystem::exists(path)) {
        throw std::invalid_argument("A file already exists at " + path.string());
    }
    std::ofstream ofs(path, std::ios::binary | std::ios::out);
    if (size == 0) { return; }
    ofs.seekp(size - 1);
    ofs.write("", 1);
}

// Note: This will work for an existing file, but will not "clip" its size to
// the desired one
inline void ensure_sized_file_existence(filesystem::path path, size_t size)
{
    if (filesystem::exists(path)) {
        filesystem::resize_file(path, size);
        return;
    }
    create_sized_file(path, size);
}
*/

//namespace std {
//
//template<typename OStream>
//OStream& operator<<(OStream &os, const filesystem::path &p)
//{
//    return os << p.native();
//}
//
//} // namespace std

#endif // UTIL_FILESYSTEM_HPP_
