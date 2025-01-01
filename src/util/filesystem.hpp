#ifndef UTIL_FILESYSTEM_HPP_
#define UTIL_FILESYSTEM_HPP_

#include <fstream>
#if _POSIX_SOURCE
#include <unistd.h>
#else
#ifdef WIN32
#include <io.h>
#endif
#endif

static_assert(__cplusplus >= 201402L, "C++2014 is required to compile this program");

#if __cplusplus == 201402L
#include <experimental/filesystem>
namespace filesystem = std::experimental::filesystem;
#elif __cplusplus >= 201703L
#include <filesystem>
namespace filesystem = std::filesystem;
#endif

filesystem::path maybe_prepend_base_dir(
    const filesystem::path& prefix,
    const filesystem::path& suffix_or_absolute_path);

//namespace std {
//
//template<typename OStream>
//OStream& operator<<(OStream &os, const filesystem::path &p)
//{
//    return os << p.native();
//}
//
//} // namespace std

enum path_check_kind { for_reading, for_writing, for_recursion };

bool has_permission(const filesystem::path& path, path_check_kind permissions_kind);

void verify_path(const filesystem::path& path, path_check_kind check_kind, bool allow_overwrite);

#endif // UTIL_FILESYSTEM_HPP_
