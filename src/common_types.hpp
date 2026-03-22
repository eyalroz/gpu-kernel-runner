#ifndef COMMON_TYPES_HPP_
#define COMMON_TYPES_HPP_

#include "util/filesystem.hpp"
#include "util/optional_and_any.hpp"
#include "util/memory_region.hpp"

#include <string>
#include <cstdint>
#include <climits>
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
using std::size_t;

using device_id_t = int;
using run_index_t = unsigned;

using duration_t = std::chrono::duration<uint64_t, std::nano>;
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
enum class kind_t { scalar, buffer };

enum : bool { is_normalized = true, isnt_normalized = false };
enum : bool { is_signed = true, isnt_signed = false, is_unsigned = false };
enum : int  { integral, floating_point };
using numeric_category_t = int;

struct element_type_descriptor_t {
    size_t num_bits;
    bool signed_;
    bool normalized;
    numeric_category_t numeric_category;

    size_t size_in_bytes() const noexcept { return num_bits / CHAR_BIT; }
};

inline element_type_descriptor_t channel_descriptor_for(const std::string& type_name)
{
    // Note: This does not cover all of OpenCL's channel data types
    static const std::unordered_map<std::string, element_type_descriptor_t>
    image_channel_element_type_descriptors = {
        { "CL_SNORM_INT8",       {  8, is_signed,   is_normalized,   integral       } },
        { "CL_SNORM_INT16",      { 16, is_signed,   is_normalized,   integral       } },
        { "CL_UNORM_INT8",       {  8, is_unsigned, is_normalized,   integral       } },
        { "CL_UNORM_INT16",      { 16, is_unsigned, is_normalized,   integral       } },
        { "CL_SIGNED_INT8",      {  8, is_signed,   isnt_normalized, integral       } },
        { "CL_SIGNED_INT16",     { 16, is_signed,   isnt_normalized, integral       } },
        { "CL_SIGNED_INT32",     { 32, is_signed,   isnt_normalized, integral       } },
        { "CL_UNSIGNED_INT8",    {  8, is_unsigned, isnt_normalized, integral       } },
        { "CL_UNSIGNED_INT16",   { 16, is_unsigned, isnt_normalized, integral       } },
        { "CL_UNSIGNED_INT32",   { 32, is_unsigned, isnt_normalized, integral       } },
        { "CL_HALF_FLOAT",       { 16, is_signed,   isnt_normalized, floating_point } },
        { "CL_FLOAT",            { 32, is_signed,   isnt_normalized, floating_point } },
        // from here on out - just aliases
        { "int8_t",              {  8, is_signed,   isnt_normalized, integral       } },
        { "int16_t",             { 16, is_signed,   isnt_normalized, integral       } },
        { "int32_t",             { 32, is_signed,   isnt_normalized, integral       } },
        { "uint8_t",             {  8, is_unsigned, isnt_normalized, integral       } },
        { "uint16_t",            { 16, is_unsigned, isnt_normalized, integral       } },
        { "uint16_t",            { 32, is_unsigned, isnt_normalized, integral       } },
        { "float16_t",           { 16, is_signed,   isnt_normalized, floating_point } },
        { "float32_t",           { 32, is_signed,   isnt_normalized, floating_point } },
        // No double support... i think
        // { "float64_t",           { 64, is_signed,   isnt_normalized, floating_point } },
        { "half",                { 16, is_signed,   isnt_normalized, floating_point } },
        { "float",               { 32, is_signed,   isnt_normalized, floating_point } },
        { "int8",                {  8, is_signed,   isnt_normalized, integral       } },
        { "int16",               { 16, is_signed,   isnt_normalized, integral       } },
        { "int32",               { 32, is_signed,   isnt_normalized, integral       } },
        { "uint8",               {  8, is_unsigned, isnt_normalized, integral       } },
        { "uint16",              { 16, is_unsigned, isnt_normalized, integral       } },
        { "char",                {  8, is_signed,   isnt_normalized, integral       } },
        { "uchar",               {  8, is_unsigned, isnt_normalized, integral       } },
        { "short",               { 16, is_signed,   isnt_normalized, integral       } },
        { "ushort",              { 16, is_unsigned, isnt_normalized, integral       } },
        { "int",                 { 32, is_signed,   isnt_normalized, integral       } },
        { "uint",                { 32, is_unsigned, isnt_normalized, integral       } },
        { "long",                { 64, is_signed,   isnt_normalized, integral       } },
        { "ulong",               { 64, is_unsigned, isnt_normalized, integral       } },
    };
    return image_channel_element_type_descriptors.at(type_name);
}

// Q: Why don't we distinguish CUDA and OpenCL for purposes of naming channel descriptors?
// A: Because we have to parse the command-line for these, at which point we have not determined
//    whether they're being used with a CUDA or OpenCL ecosystem. We could perhaps arrange it differently.

} // namespace kernel_parameters

inline kernel_parameters::element_type_descriptor_t operator ""_desc(const char* str, size_t len)
{
    std::string type_name{str};
    if (type_name.length() != len) { throw std::invalid_argument("Invalid string length"); }
    return kernel_parameters::channel_descriptor_for(type_name);
}

inline bool is_buffer(kernel_parameters::kind_t kind) noexcept { return (kind == kernel_parameters::kind_t::buffer); }
inline bool is_scalar(kernel_parameters::kind_t kind) noexcept { return (kind == kernel_parameters::kind_t::scalar); }

inline constexpr const char* kernel_source_file_suffix(execution_ecosystem_t ecosystem)
{
    return ecosystem == execution_ecosystem_t::cuda ? "cu" : "cl";
}

inline constexpr const char* ptx_file_extension(execution_ecosystem_t ecosystem) {
    return ecosystem == execution_ecosystem_t::cuda ? "ptx" : "clptx";
}

enum class buffer_kind_t { raw, image /* no support for image array */ };

enum class parameter_direction_t {
    input = 0,
    in = input,
    output = 1,
    out = output,
    inout = 2,
    io = inout,
    scratch = 3
};

inline bool is_input(parameter_direction_t dir) noexcept
{
    return dir == parameter_direction_t::in or dir == parameter_direction_t::inout;
}

inline bool is_output(parameter_direction_t dir) noexcept
{
    return dir == parameter_direction_t::inout or dir == parameter_direction_t::out;
}

inline constexpr const char* parameter_direction_name(parameter_direction_t dir)
{
    constexpr const char* names[] = { "input", "output", "inout", "scratch" };
    return names[(int) dir];
}

// A dynarray would be useful here
using dimensions_t = std::vector<size_t>;

using host_buffer_t = std::vector<byte_type>;
using host_buffers_t = std::unordered_map<std::string, host_buffer_t>;

struct device_side_buffer_info_t {
    bool is_image;
    // The other fields can be default-initialized, so we won't use a case-class here
    size_t size;
    dimensions_t dimensions;
    optional<dimensions_t> pitches;
    std::size_t num_channels;
    kernel_parameters::element_type_descriptor_t channel_elements_type;
};

inline device_side_buffer_info_t make_raw_device_side_buffer_info(size_t size) noexcept {
    device_side_buffer_info_t result;
    result.is_image = false;
    result.size = size;
    return result;
}

#endif /* COMMON_TYPES_HPP_ */
