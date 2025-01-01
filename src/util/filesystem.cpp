#include "filesystem.hpp"

filesystem::path maybe_prepend_base_dir(
    const filesystem::path& prefix,
    const filesystem::path& suffix_or_absolute_path)
{
    bool can_prepend_buffer_dir = (prefix != filesystem::path{}) and (suffix_or_absolute_path.is_relative());
    return can_prepend_buffer_dir ? (prefix / suffix_or_absolute_path) : suffix_or_absolute_path;
}

bool has_permission(const filesystem::path& path, path_check_kind permissions_kind)
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
