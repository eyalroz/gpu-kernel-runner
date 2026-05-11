#include "filesystem.hpp"

filesystem::path maybe_prepend_base_dir(
    const filesystem::path& prefix,
    const filesystem::path& suffix_or_absolute_path)
{
    bool can_prepend_buffer_dir = (prefix != filesystem::path{}) && (suffix_or_absolute_path.is_relative());
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
    return (access(path.string().c_str(), perm_flags) == have_access);
#elif WIN32
    if (permissions_kind == for_recursion)
    {
        // _access_s only reports on existence, read and write permissions,
        // so let's assume we have recursion permission
        return true;
    }
    static constexpr int have_access{ 0 };
    auto perm_flags = (permissions_kind == for_reading) ? 2 : 4;
    return (_access_s( path.string().c_str(), perm_flags) == have_access);
#else
    (void) permissions_kind;
    (void) path;
    return true;
#endif
}

void verify_path(const filesystem::path& path, path_check_kind check_kind, bool allow_overwrite)
{
    // TODO: Consider checking path execution permissions down to the parent directory
    if (filesystem::exists(path)) {
        if (is_directory(path)) {
            throw std::invalid_argument(::std::string("Path is a directory, not a (readable) path: ") + std::string() + path.string());
        }
        if (check_kind == for_writing and not allow_overwrite) {
            throw std::invalid_argument("File already exists, and overwrite is not allowed: " + path.string());
        }
        if (not has_permission(path, check_kind))
        {
            throw std::invalid_argument("No read permissions for path: " + path.string());
        }
    }
    else {
        if (check_kind != for_writing)
        {
            throw std::invalid_argument("File does not exist: " + path.string());
        }
        auto parent = path.parent_path();
        if (not parent.empty() and not is_directory(parent)) {
            throw std::invalid_argument("Parent path of intended file is not a directory: " + path.string());
        }
        if (parent.empty()) {
            parent = filesystem::current_path();
        }
        if (not has_permission(parent, for_writing))
        {
            throw std::invalid_argument("No write permissions for directory " + parent.string()
                + " , where it is necessary to write the new file " + path.filename().string());
        }
    }
}
