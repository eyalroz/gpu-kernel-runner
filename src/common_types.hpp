#ifndef COMMON_TYPES_HPP_
#define COMMON_TYPES_HPP_

#include "util/filesystem.hpp"
#include "util/optional_and_any.hpp"
#include "util/memory_region.hpp"

#include <string>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

using util::byte_type;
using util::memory_region;
using util::const_memory_region;
using util::as_region;

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

using duration_t = std::chrono::duration<std::uint64_t, std::nano>;
using durations_t = std::vector<duration_t>;

using string_map = std::unordered_map<std::string, std::string>;
using maybe_string_map = std::unordered_map<std::string, optional<std::string>>;
using valued_preprocessor_definitions_t = std::unordered_map<std::string, std::string>;
using string_set = std::unordered_set<std::string>;
using preprocessor_definitions_t = string_set;
using argument_values_t = std::unordered_map<std::string, std::string>;
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

namespace kernel_parameters {

enum : bool { is_required = true, isnt_required = false };
enum class kind_t { buffer, scalar };

} // namespace kernel_parameters


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
    scratch = 3
};

inline bool is_input(parameter_direction_t dir)
{
    return dir == parameter_direction_t::in or dir == parameter_direction_t::inout;
}

inline bool is_output(parameter_direction_t dir)
{
    return dir == parameter_direction_t::inout or dir == parameter_direction_t::out;
}

inline constexpr const char* parameter_direction_name(parameter_direction_t dir)
{
    constexpr const char* names[] = { "input", "output", "inout", "scratch" };
    return names[(int) dir];
}

using host_buffer_t = std::vector<byte_type>;
using host_buffers_t = std::unordered_map<std::string, host_buffer_t>;

#endif /* COMMON_TYPES_HPP_ */
