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

inline filesystem::path maybe_prepend_base_dir(
    const filesystem::path& prefix,
    const filesystem::path& suffix_or_absolute_path)
{
    bool can_prepend_buffer_dir = (prefix != filesystem::path{}) and (suffix_or_absolute_path.is_relative());
    return can_prepend_buffer_dir ? (prefix / suffix_or_absolute_path) : suffix_or_absolute_path;
}

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

inline bool has_permission(const filesystem::path& path, path_check_kind permissions_kind)
{
#ifdef _POSIX_SOURCE
    static constexpr int have_access{ 0 };
    auto perm_flags =
        (permissions_kind == for_reading) ? R_OK :
        (permissions_kind == for_recursion) ? X_OK :
        W_OK;
    return (access(path.native().c_str(), perm_flags) == have_access);
#else
#ifdef WIN32
    if (permissions_kind == for_recrusion) { return; }
    static constexpr int have_access{ 0 };
    auto perm_flags = (permissions_kind == for_reading) ? 2 : 4;
    return (_access_s( path.native.c_str(), perm_flags) == have_access);
#endif
#endif
}


#endif // UTIL_FILESYSTEM_HPP_
