#include "common_types.hpp"
#include "util/filesystem.hpp"
#include "execution_context.hpp"

host_buffers_t read_input_buffers_from_files(
    const name_set&           buffer_names,
    const string_map&         filenames,
    const filesystem::path&   buffer_directory);

void read_input_buffers_from_files(execution_context_t& context);


/**
 * Writes output buffers, generated by the kernel, to the files specified at the
 * command-line - one file per buffer.
 */
void write_buffers_to_files(execution_context_t& context);

void copy_buffer_to_device(
    const execution_context_t& context,
    const std::string&         buffer_name,
    const device_buffer_type&  device_side_buffer,
    const host_buffer_t&    host_side_buffer);

void copy_buffer_on_device(
    execution_ecosystem_t      ecosystem,
    cl::CommandQueue*          queue,
    const device_buffer_type&  destination,
    const device_buffer_type&  origin);

void copy_input_buffers_to_device(const execution_context_t& context);

void copy_buffer_to_host(
    execution_ecosystem_t      ecosystem,
    cl::CommandQueue*          opencl_queue,
    const device_buffer_type&  device_side_buffer,
    host_buffer_t&          host_side_buffer);

// Note: must take the context as non-const, since it has vector members, and vectors
// are value-types, not reference-types, i.e. copying into those vectors changes
// the context.
void copy_outputs_from_device(execution_context_t& context);

void schedule_zero_buffer(
    execution_ecosystem_t                  ecosystem,
    const device_buffer_type               buffer,
    const optional<const cuda::stream_t*>  cuda_stream,
    const cl::CommandQueue*                opencl_queue,
    const std::string&                     buffer_name);

void schedule_zero_output_buffers(execution_context_t& context);

void schedule_zero_single_buffer(const execution_context_t& context, const device_buffer_type& buffer);

void create_device_side_buffers(execution_context_t& context);

// Note: Will create buffers also for each inout buffers
void create_host_side_output_buffers(execution_context_t& context);

void schedule_reset_of_inout_buffers_working_copy(execution_context_t& context);

void write_buffers_to_files(execution_context_t& context);

enum : bool {
    log_file_write_at_info_level = true,
    log_file_write_at_debug_level = false,
    dont_log_file_write_at_info_level = false,
};

void write_data_to_file(
    std::string          kind,
    std::string          name,
    const_memory_region  data,
    filesystem::path     destination,
    bool                 overwrite_allowed,
    bool                 log_at_info_level);

