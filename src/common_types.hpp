#ifndef COMMON_TYPES_HPP_
#define COMMON_TYPES_HPP_

#include <util/filesystem.hpp>
#include <util/optional_and_any.hpp>

#include <string>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

#if __cplusplus >= 201712L
using byte_type = std::byte;
#else
using byte_type = char;
#endif

using std::uint32_t;
using std::int32_t;
using std::uint16_t;
using std::int16_t;
using std::uint64_t;
using std::int64_t;
using std::uint8_t;
using std::int8_t;

using device_id_t = int;
using run_index_t = unsigned;

using string_map = std::unordered_map<std::string, std::string>;
using preprocessor_value_definitions_t = std::unordered_map<std::string, std::string>;
using preprocessor_definitions_t = std::unordered_set<std::string>;
using include_paths_t = std::vector<std::string>;

enum class execution_ecosystem_t {
    cuda = 0,
    opencl = 1,
};

constexpr const char* ecosystem_name(execution_ecosystem_t e)
{
    constexpr const char* names[] = { "CUDA", "OpenCL" };
    return names[(int) e];
}

inline constexpr const char* kernel_source_file_suffix(execution_ecosystem_t ecosystem)
{
    return ecosystem == execution_ecosystem_t::cuda ? "cu" : "cl";
}

inline constexpr const char* ptx_file_extension(execution_ecosystem_t ecosystem) {
    return ecosystem == execution_ecosystem_t::cuda ? "ptx" : "clptx";
}

enum class parameter_direction_t {
    input = 0,
    in = input,
    output = 1,
    out = output,
    inout = 2,
    io = inout,
};

constexpr const char* parameter_direction_name(parameter_direction_t dir)
{
    constexpr const char* names[] = { "input", "output", "inout" };
    return names[(int) dir];
}

using host_buffer_type = std::vector<byte_type>;
using host_buffers_map = std::unordered_map<std::string, host_buffer_type>;

struct poor_mans_span {
    byte_type* data_;
    std::size_t size_;

    byte_type * const & data() const { return data_; }
    byte_type * & data() { return data_; }
    const std::size_t& size() const { return size_; }
    std::size_t& size() { return size_; }
};

#endif /* COMMON_TYPES_HPP_ */
