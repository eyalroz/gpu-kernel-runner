/**
 * @file
 *
 * @brief the main() function of the GPU kernel runner and functions not
 * otherwise factored away.
 */
#include "common_types.hpp"
#include "cmdline_arguments.hpp"
#include "execution_context.hpp"
#include "kernel_adapter.hpp"
#include "buffer_ops.hpp"

#include "nvrtc-related/build.hpp"
#include "nvrtc-related/execution.hpp"
#include "nvrtc-related/miscellany.hpp"
#include "opencl-related/build.hpp"
#include "opencl-related/execution.hpp"
#include "opencl-related/miscellany.hpp"

#include "util/miscellany.hpp"
#include "util/spdlog-extra.hpp"

#include <cxxopts/cxxopts.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/cfg/env.h>

#include <iostream>
#include <vector>

void configure_device_and_built_kernel(execution_context_t const& context);

using std::size_t;
using std::string;

void ensure_necessary_terms_were_defined(const execution_context_t& context)
{
    auto required_terms = util::transform_if<std::unordered_set<string>>(
        context.get_kernel_adapter().preprocessor_definition_details(),
        [](const auto& sad) { return sad.required; },
        [](const auto& sad) { return string{sad.name};});
    auto defined_valued_terms = util::keys(context.preprocessor_definitions.finalized.valued);
    auto all_defined_terms = util::union_(defined_valued_terms, context.preprocessor_definitions.finalized.valueless);
    auto required_but_undefined = util::difference(required_terms, all_defined_terms);
    if (not required_but_undefined.empty()) {
        die("Missing obligatory preprocessor definitions: {}", required_but_undefined);
    }
}

std::vector<filesystem::path> get_ecosystem_include_paths(execution_context_t& context)
{
    return (context.ecosystem == execution_ecosystem_t::cuda) ?
        get_ecosystem_include_paths_<execution_ecosystem_t::cuda>() :
        get_ecosystem_include_paths_<execution_ecosystem_t::opencl>();
}

void collect_include_paths(execution_context_t& context)
{
    // Note the relative order in which we place the includes; it is non-trivial.
    context.finalized_include_dir_paths = context.options.include_dir_paths;
    if (not context.options.set_default_compilation_options) {
        return;
    }

    auto source_file_include_dir = context.options.kernel.source_file.parent_path();
    if (source_file_include_dir.empty()) {
        // We can't rely on the dynamic compilation libraries accepting empty paths.
        // ... and note that "." is guaranteed to be portable to any platform
        source_file_include_dir = filesystem::path{"."};
    }
    context.finalized_include_dir_paths.insert(context.finalized_include_dir_paths.begin(), source_file_include_dir.native());

    auto ecosystem_include_paths = get_ecosystem_include_paths(context);
    util::append(ecosystem_include_paths, context.finalized_include_dir_paths);
}

std::string resolve_input_buffer_filename(
    const execution_context_t&                      context,
    optional<string>                                buffer_cmdline_arg,
    const kernel_adapter::single_parameter_details& buffer_param_details)
{
    const auto& name = buffer_param_details.name;
    // Note: We're willing to accept user-requested filenames as-is; but if we're
    // performing a search, we'll only accept names for which the files actually exist
    if (buffer_cmdline_arg) {
        return buffer_cmdline_arg.value();
    }
    spdlog::debug("Input filename for buffer parameter '{}' not specified; will try its name and aliases as fallback filenames.", name);

    if (filesystem::exists(context.options.buffer_base_paths.input / name)) {
        return name;
    }
    else {
        spdlog::debug("Fallback input filename search for input buffer parameter '{}': No such file {}", name, (context.options.buffer_base_paths.input / name).native());
    }
    for(const auto& alias : buffer_param_details.get_aliases()) {
        filesystem::path input_file_for_alias = context.options.buffer_base_paths.input / alias;
        if (filesystem::exists(input_file_for_alias)) {
            return alias;
        }
        else {
            spdlog::debug("Fallback input filename search for input buffer parameter '{}': No such file {}", name, input_file_for_alias.native());
        }
    }
    die("Cannot locate an input buffer file for parameter {}", name);
}

std::string resolve_output_buffer_filename(
    execution_context_t&                     context,
    optional<string>                         buffer_cmdline_arg,
    kernel_adapter::single_parameter_details buffer_param_details)
{
    const auto& name = buffer_param_details.name;
    auto output_file =[&]() -> filesystem::path {
        if (is_input(buffer_param_details)) {
            // Note that, in this case, we ignore the command-line argument, since it has
            // already been taken into account via the input filename
            auto input_file = context.buffers.filenames.inputs[name].value();
            return input_file + ".out";
        }
        if (buffer_cmdline_arg) {
            return buffer_cmdline_arg.value();
        }
        auto default_filename = fmt::format("{}.out", name);
        return default_filename;
    }();

    // Note that if the output file gets created while the kernel runs, we might miss this fact
    // when trying to write to it.

    // TODO: Move this verification elsewhere? We could have a separate function called from main()
    // just for performing these checks
    if (filesystem::exists(output_file)) {
        if (not context.options.overwrite_allowed and not context.options.compile_only) {
            die("Writing the contents of output buffer '{}' would overwrite output buffer file: {}",
                name, output_file.native());
        }
        spdlog::info("Output buffer '{}' will overwrite {}", name, output_file.native());
    }
    return output_file;
}

void resolve_buffer_filenames(execution_context_t& context)
{
    // Note that, at this point, we assume all input buffer entries in the map
    // are resolved, i.e. engaged optionals.
    const auto& ka = *context.kernel_adapter_;

    const auto& args = context.kernel_arguments;
    auto params_with_args = util::keys(args);
    for(const auto& buffer : ka.buffer_details()) {
        if (buffer.name == nullptr or *(buffer.name) == '\0') {
            die("Empty/missing kernel parameter name encountered");
        }
        std::string buffer_name = buffer.name; // TODO: Yes, we should really switch to string_view's at some point...
        auto got_arg = util::contains(params_with_args, buffer.name);
        auto maybe_cmdline_name = value_if(got_arg, [&]() { return args.at(buffer.name); });
        if (is_input(buffer)) {
            context.buffers.filenames.inputs[buffer_name] = resolve_input_buffer_filename(context, maybe_cmdline_name, buffer);
            spdlog::debug("Input buffer file for parameter '{}': {}", buffer.name, context.buffers.filenames.inputs[buffer_name].value());
        }
        if (context.options.write_output_buffers_to_files and is_output(buffer)) {
            // Note: For an inout buffer, both name resolutions are called, and the
            // latter depends on the former having succeeded.
            context.buffers.filenames.outputs[buffer_name] = resolve_output_buffer_filename(context, maybe_cmdline_name, buffer);
            spdlog::debug("Output buffer file for parameter '{}': {}", buffer.name, context.buffers.filenames.outputs[buffer_name]);
        }
    }
}

// Note: We need the kernel adapter, and can't just be satisfied with the argument details,
// because the adapter might have overridden the parsing method with something more complex.
// If we eventually decide that's not a useful ability to have, we can avoid passing the
// adapter to this function.
void parse_scalars(execution_context_t &context)
{
    const auto& args = context.kernel_arguments;
    auto& adapter = context.get_kernel_adapter();
    auto all_scalar_details = adapter.scalar_parameter_details();
    for(const auto& spd : all_scalar_details) {
        if (not util::contains(args, spd.name)) continue; // Maybe we can generate this later
        // TODO: Consider not parsing anything at this stage, and just marshaling all the scalar arguments together.
        auto& arg_value = args.at(spd.name);
        spdlog::trace("Parsing command-line argument for scalar parameter '{}' from \"{}\"", spd.name, arg_value);
        context.scalar_input_arguments.raw[spd.name] = arg_value;
        context.scalar_input_arguments.typed[spd.name] =
            adapter.parse_cmdline_scalar_argument(spd, arg_value);
        spdlog::trace("Successfully parsed command-line argument for scalar parameter '{}'.", spd.name);
    }
}

void ensure_gpu_device_validity(
    execution_ecosystem_t  ecosystem,
    optional<unsigned>     platform_id,
    int                    device_id,
    bool                   need_ptx)
{
    std::size_t device_count;

    spdlog::debug("Ensuring the requested GPU device exists{}",  (platform_id ? " on the specified platform" : ""));
    constexpr const unsigned OpenCLDefaultPlatformID { 0 };
    switch(ecosystem) {
    case execution_ecosystem_t::opencl : {
        auto actual_platform_id = platform_id.value_or(OpenCLDefaultPlatformID);
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        not platforms.empty() or die("No OpenCL platforms found.");
        platforms.size() > actual_platform_id
            or die ("No OpenCL platform exists with ID {}", actual_platform_id);
        auto& platform = platforms[actual_platform_id];
        if (spdlog::level_is_at_least(spdlog::level::debug)) {
            spdlog::debug("Using OpenCL platform {}: {}", actual_platform_id, get_name(platform));
        }
        if (need_ptx and not uses_ptx(platform)) {
            die("PTX file requested, but chosen OpenCL platform '{}' does not generate PTX files during build", get_name(platform));
        }
        cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties) (platform)(), 0
        };
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        if (devices.empty()) { die("No OpenCL devices found on the platform {}", actual_platform_id); }
        device_count = (std::size_t) devices.size();
        break;
    }
    case execution_ecosystem_t::cuda:
    default:
        device_count = (std::size_t) cuda::device::count();
        if(device_count == 0) die("No CUDA devices detected on this system");
        break;
    }
    if(device_id < 0 or device_id >= (int) device_count)
        die ("Please specify a valid device index (in the range 0.. {})", cuda::device::count()-1);
}

void print_registered_kernel_keys() {
    auto const &scf = kernel_adapter::get_subclass_factory();
    auto kernel_names = util::keys(scf.instantiators());
    for (const auto &key : kernel_names) {
        std::cout << key << '\n';
    }
}

void list_opencl_platforms()
{
    auto platforms = [] {
        std::vector<cl::Platform> platforms_;
        cl::Platform::get(&platforms_);
        return platforms_;
    }();
    auto platform_id = 0;
    for(const auto& platform : platforms) {
        std::cout << "Platform " << platform_id << ": "
            << platform.getInfo<CL_PLATFORM_NAME>() << " (by "
            << platform.getInfo<CL_PLATFORM_VENDOR>() << ")\n";
        platform_id++;
    }
    std::cout << '\n';
}

void perform_early_exit_action(parsed_cmdline_options_t const& parsed)
{
    if (parsed.early_exit_action) {
        throw std::invalid_argument("Parsed command-line options do not necessitate an early-exit action");
    }
    switch(parsed.early_exit_action.value()) {
        case early_exit_action_t::list_kernels:
            print_registered_kernel_keys(); break;
        case early_exit_action_t::list_opencl_platforms:
            list_opencl_platforms(); break;
        case early_exit_action_t::print_help:
        default:
            auto &ostream = parsed.valid ? std::cout : std::cerr;
            ostream << parsed.help_text.value() << '\n';
    }
}

void verify_key_validity(const std::string& key)
{
    if (not kernel_adapter::can_produce_subclass(string(key))) {
        die("No kernel adapter is registered for key {}", key);
    }
}

// TODO: Yes, make execution_context_t a proper class... and be less lax with the initialization
execution_context_t initialize_execution_context(const parsed_cmdline_options_t& parsed_options)
{
    // Somewhat redundant with later code
    ensure_gpu_device_validity(
        parsed_options.gpu_ecosystem,
        parsed_options.platform_id,
        parsed_options.gpu_device_id,
        parsed_options.write_ptx_to_file);

    spdlog::debug("Initializing kernel execution context");

    execution_context_t execution_context {};
    execution_context.options = parsed_options;
    execution_context.ecosystem = parsed_options.gpu_ecosystem;

    if (parsed_options.gpu_ecosystem == execution_ecosystem_t::cuda) {
        initialize_execution_context<execution_ecosystem_t::cuda>(execution_context);
    }
    else { // OpenCL
        initialize_execution_context<execution_ecosystem_t::opencl>(execution_context);
    }
    execution_context.kernel_adapter_ =
        kernel_adapter::produce_subclass(string(parsed_options.kernel.key));

    collect_include_paths(execution_context);
    execution_context.language_standard =
        parsed_options.language_standard ? parsed_options.language_standard :
        parsed_options.set_default_compilation_options ?
        execution_context.kernel_adapter_->default_language_standard(execution_context.ecosystem) :
        nullopt;

    return execution_context;
}

// The user may have specified arguments via aliases rather than their proper names
void dealias_arguments(execution_context_t &context)
{
    auto param_details = context.kernel_adapter_->parameter_details();
    auto key_mapper = [&param_details](const string& alias) -> string {
        auto iter = std::find_if(std::cbegin(param_details), std::cend(param_details),
            [&](const kernel_adapter::single_parameter_details& spd) {
                bool result = spd.has_alias(alias);
                return result;
            } );
        return (iter == param_details.cend()) ? alias : iter->name;
    };

    context.kernel_arguments = util::transform<argument_values_t>(
        context.options.aliased_kernel_arguments,
        [&key_mapper](const auto& pair) -> std::pair<string, string> {
            const auto& key = pair.first;
            const auto& value = pair.second;
            return {key_mapper(key), value};
        } );

    auto params_with_args = util::keys(context.kernel_arguments);
    spdlog::trace("Arguments have been specified on the command-line for parameters {}", params_with_args);
}

void finalize_kernel_function_name(execution_context_t& context)
{
    auto& kinfo = context.options.kernel;
    if (kinfo.function_name.empty()) {
        kinfo.function_name = context.kernel_adapter_->kernel_function_name();
        if (not util::is_valid_identifier(kinfo.function_name)) {
            die("The registered kernel function name for adapter '{}' is invalid: '{}'",
                kinfo.key, kinfo.function_name);
        }
    }

    if (context.options.write_ptx_to_file and
        context.options.ptx_output_file.empty())
    {
        context.options.ptx_output_file =
        context.options.kernel.function_name + '.' +
        ptx_file_extension(context.options.gpu_ecosystem);
    }
}

bool build_kernel(execution_context_t& context)
{
    finalize_kernel_function_name(context);
    const auto& source_file = context.options.kernel.source_file;
    spdlog::debug("Reading the kernel from {}", source_file.native());
    auto kernel_source_buffer = util::read_file_as_null_terminated_string(source_file);
    auto kernel_source = static_cast<const char*>(kernel_source_buffer.data());
    bool build_succeeded;

    if (context.ecosystem == execution_ecosystem_t::cuda) {
        auto result = build_cuda_kernel(
            *context.cuda.context,
            source_file.c_str(),
            kernel_source,
            context.options.kernel.function_name.c_str(),
            context.options.set_default_compilation_options,
            context.options.generate_debug_info,
            context.options.generate_source_line_info,
            context.language_standard,
            context.finalized_include_dir_paths,
            context.options.preinclude_files,
            context.preprocessor_definitions.finalized,
            context.options.extra_compilation_options);
        build_succeeded = result.succeeded;
        context.compilation_log = std::move(result.log);
        if (result.succeeded) {
            context.cuda.module = std::move(result.module);
            context.compiled_ptx = std::move(result.ptx);
            context.cuda.kernel = std::move(result.kernel);
        }
    }
    else {
        // Notes:
        // 1. We don't pass context.options.set_default_compilation_parameters,
        //    because at this point we've already applied this setting if necessary,
        //    so that this function will not issue compilation options not explicitly
        //    requested.
        // 2. Not using the language_standard option
        //
        auto result = build_opencl_kernel(
            context.opencl.context,
            context.opencl.device,
            context.device_id,
            context.options.kernel.function_name.c_str(),
            kernel_source,
            context.options.generate_debug_info,
            context.options.generate_source_line_info,
            context.options.write_ptx_to_file,
            context.finalized_include_dir_paths,
            context.options.preinclude_files,
            context.preprocessor_definitions.finalized,
            context.options.extra_compilation_options);
        build_succeeded = result.succeeded;
        context.compilation_log = std::move(result.log);
        if (result.succeeded) {
            context.opencl.program = std::move(result.program);
            context.compiled_ptx = std::move(result.ptx);
            context.opencl.built_kernel = std::move(result.kernel);
        }
    }
    if (build_succeeded) {
        spdlog::info("Kernel {} built successfully.", context.options.kernel.key);
    }
    else {
        spdlog::error("Kernel {} build failed.", context.options.kernel.key);
    }
    // In some cases (perhaps even due to my own fault, the compilation log string
    // contains the trailing C-language string terminating null character '\0'. Let's clean
    // that up
    if (context.compilation_log and context.compilation_log->length() > 0 ) {
        if (context.compilation_log->back() == '\0') {
            context.compilation_log->pop_back();
        }
    }
    return build_succeeded;
}

void validate_scalars(execution_context_t& context)
{
    spdlog::debug("Validating scalar argument values");

    const auto& available_args = util::keys(context.scalar_input_arguments.typed);
    spdlog::trace("Available scalar arguments: {}", available_args);
    auto required_args = util::transform_if<std::vector<const char *>>(
        context.get_kernel_adapter().scalar_parameter_details(),
        [](const auto& sad) { return sad.required; },
        [](const auto& sad) { return sad.name; });

    spdlog::trace("Required scalar arguments: {}", required_args);

    for(const auto& required : required_args) {
        util::contains(available_args, required)
            or die("Required scalar argument {} not provided (and could not be deduced)", required);
    }
}

void validate_input_buffer_sizes(execution_context_t& context)
{
    spdlog::debug("Validating input buffer sizes");
    auto& all_params = context.kernel_adapter_->parameter_details();
    auto input_buffer_details = util::filter(all_params,
        [&](const auto& param_details) {
            return is_input(param_details.direction) and param_details.kind == kernel_parameters::kind_t::buffer;
        });
    for (auto const& buffer_details : input_buffer_details) {
        auto const &buffer = context.buffers.host_side.inputs[buffer_details.name];
        if (not buffer_details.size_calculator) {
            spdlog::debug("No size calculator for input buffer '{}'; assuming size is valid", buffer_details.name);
            continue;
        }
        auto calculated = apply_size_calc(buffer_details.size_calculator, context);
        if (calculated == buffer.size()) {
            spdlog::trace("Input buffer '{}' is of size {} bytes, as expected",
                buffer_details.name, buffer.size());
            continue;
        }
        if (context.options.accept_oversized_input_buffers and calculated < buffer.size()) {
            spdlog::info("Input buffer '{}' is of size {} bytes, exceeding the expected size of {} bytes",
                buffer_details.name, buffer.size(), calculated);
            continue;
        }
        if (context.options.accept_undersized_input_buffers and calculated > buffer.size()) {
            spdlog::info("Input buffer '{}' is of size {} bytes, less than the expected size of {} bytes",
                buffer_details.name, buffer.size(), calculated);
            continue;
        }
        die("Input buffer '{}' is of size {} bytes, {} its required {}size of {} bytes",
            buffer_details.name, buffer.size(),
            (buffer.size() > calculated ? "exceeding" : "less than"),
            (context.options.accept_oversized_inputs ? "minimum " : ""),
            calculated);
    }
}

void validate_arguments(execution_context_t& context)
{
    validate_scalars(context);
    validate_input_buffer_sizes(context);

    if (not context.kernel_adapter_->extra_validity_checks(context)) {
        // TODO: Have the kernel adapter report an error instead of just a boolean;
        // but we don't want it to know about spdlog, so it should probably
        // return a runtime_error (?)
        die("The combination of input arguments (scalars and buffers) and preprocessor definitions is invalid.");
    }
    spdlog::info("Kernel arguments are fully valid (scalars and input buffers, including any inout)");
}

void generate_additional_scalar_arguments(execution_context_t& context)
{
    auto& adapter = context.get_kernel_adapter();
    auto generated_scalars = adapter.generate_additional_scalar_arguments(context);
    if (not generated_scalars.empty()) {
        spdlog::debug("Generated additional scalar arguments: {}", util::keys(generated_scalars));
    }
    context.scalar_input_arguments.typed.insert(generated_scalars.begin(), generated_scalars.end());
    auto all_scalar_details = adapter.scalar_parameter_details();
}

void schedule_single_run(execution_context_t& context, run_index_t run_index)
{
    if (spdlog::level_is_at_least(spdlog::level::debug)) {
        spdlog::debug("Preparing for kernel run {1:>{0}} of {2:>{0}} (1-based).",
            util::naive_num_digits(context.options.num_runs),
            run_index + 1, context.options.num_runs);
    }
    if (context.options.zero_output_buffers) {
        schedule_zero_output_buffers(context);
    }
    if (context.options.clear_l2_cache) {
        schedule_zero_single_buffer(context, context.buffers.device_side.l2_cache_clearing_gadget.value());
        spdlog::debug("Hopefully cleared the L2 cache by memset'ing a dummy buffer");
    }
    schedule_reset_of_inout_buffers_working_copy(context);

    if (context.ecosystem == execution_ecosystem_t::cuda) {
        launch_and_time_cuda_kernel(context, run_index);
    }
    else {
        launch_and_time_opencl_kernel(context, run_index);
    }
    spdlog::debug("{0} {2:>{1}} done",
        context.options.sync_after_kernel_execution ? "Scheduling of kernel run" : "Kernel run",
        util::naive_num_digits(context.options.num_runs), run_index+1);
}

void finalize_kernel_arguments(execution_context_t& context)
{
    spdlog::debug("Marshaling kernel arguments.");
    context.finalized_arguments = context.kernel_adapter_->marshal_kernel_arguments(context);
    if (context.ecosystem == execution_ecosystem_t::opencl) {
        set_opencl_kernel_arguments(context.opencl.built_kernel, context.finalized_arguments);
    }
    else {
        auto num_args = context.finalized_arguments.pointers.size();
        auto num_digits = util::naive_num_digits(num_args);
        for(size_t i = 0; i < context.finalized_arguments.pointers.size() - 1; i++ ) {
            auto arg = context.finalized_arguments.pointers[i];
            auto spd = context.kernel_adapter_->parameter_details()[i];
            if (spd.kind != kernel_parameters::kind_t::buffer) {
                spdlog::trace("Kernel argument {0:1}: Scalar", i, num_digits);
                continue;
            }
            auto as_double_ptr = static_cast<const void* const *>(arg);
            void const* buffer_ptr = *as_double_ptr;
            spdlog::trace("Kernel argument {}: Buffer at {}", i, buffer_ptr);
        }
    }
    spdlog::debug("Finalized {} arguments for kernel function \"{}\"",
        context.finalized_arguments.pointers.size() - 1,
        context.options.kernel.function_name);
}

void prepare_kernel_launch_config(execution_context_t& context)
{
    spdlog::debug("Creating a launch configuration.");
    auto lc_components = context.kernel_adapter_->make_launch_config(context);
    has_zeros(lc_components) and die(
        "The kernel adapter provided invalid launch configuration components, containing zero-dimensions: "
        "{}{}{}",
        has_zeros(lc_components.block_dimensions) ? "block dimensions  " : "",
        has_zeros(lc_components.grid_dimensions) ? "grid dimensions  " : "",
        has_zeros(lc_components.overall_grid_dimensions) ? "overall grid dimensions" : "");
    lc_components.deduce_missing();
    context.kernel_launch_configuration = realize_launch_config(lc_components, context.ecosystem);

    auto gd = lc_components.grid_dimensions.value();
    auto bd = lc_components.block_dimensions.value();
    auto ogd = lc_components.overall_grid_dimensions.value();

    spdlog::info("Launch configuration: Block dimensions:   {:>11} x {:>5} x {:>5} = {:15} threads", bd[0], bd[1], bd[2], bd[0] * bd[1] * bd[2]);
    spdlog::info("Launch configuration: Grid dimensions:    {:>11} x {:>5} x {:>5} = {:15} blocks ", gd[0], gd[1], gd[2], gd[0] * gd[1] * gd[2]);
    spdlog::info("                                          ----------------------------------------------------");
    spdlog::info("Launch configuration: Overall dimensions: {:>11} x {:>5} x {:>5} = {:15} threads", ogd[0], ogd[1], ogd[2], ogd[0] * ogd[1] * ogd[2]);
    if (context.ecosystem == execution_ecosystem_t::cuda) {
        spdlog::info("Launch configuration: Dynamic shared memory:  {} bytes", lc_components.dynamic_shared_memory_size.value_or(0));
    }
    spdlog::debug("Overall dimensions cover full blocks? {}", lc_components.full_blocks());
}

void handle_compilation_log(bool compilation_succeeded, execution_context_t& context)
{
    bool empty_log = context.compilation_log and
        std::all_of(context.compilation_log.value().cbegin(),context.compilation_log.value().cend(),isspace);

    if (not compilation_succeeded) {
        if (not context.compilation_log or empty_log) {
            spdlog::error("No compilation log produced.");
        }
        else {
            spdlog::error("Kernel compilation log:\n{}\n", context.compilation_log.value());
        }
    }
    if (context.options.always_print_compilation_log) {
        if (not context.compilation_log or empty_log) {
            if (compilation_succeeded) {
                spdlog::error("No compilation log produced.");
            }
        }
        else {
            spdlog::debug("Printing kernel compilation log:");
            std::cout << context.compilation_log.value()
                << util::newline_if_missing(context.compilation_log.value());
        }
    }

    if (context.compilation_log and not context.options.compilation_log_file.empty()) {
        auto log { context.compilation_log.value() };
        write_data_to_file(
            "compilation log for", context.options.kernel.key,
            // TODO: Get rid of this, use a proper span and const span...
            as_region(log),
            context.options.compilation_log_file,
            context.options.overwrite_allowed,
            log_file_write_at_info_level);
    }
}

void maybe_write_intermediate_representation(execution_context_t& context)
{
    if (not context.options.write_ptx_to_file) { return; }
    const auto& ptx = context.compiled_ptx.value();
    write_data_to_file(
        "generated PTX for kernel", context.options.kernel.key,
        as_region(ptx),
        context.options.ptx_output_file,
        context.options.overwrite_allowed,
        log_file_write_at_info_level);
}

void print_execution_durations(std::ostream& os, const durations_t& execution_durations)
{
    for (const auto &duration: execution_durations) {
        os << duration.count() << '\n';
    }
    os << std::flush;
}

void handle_execution_durations(execution_context_t &context)
{
    if (not context.options.time_with_events) { return; }
    context.durations = (context.ecosystem == execution_ecosystem_t::cuda) ?
        compute_durations(context.cuda.timing_events):
        compute_durations(context.opencl.timing_events);

    if (context.options.print_execution_durations) {
        print_execution_durations(std::cout, context.durations);
    }
    if (not context.options.execution_durations_file.empty()) {
        std::ofstream ofs(context.options.execution_durations_file);
        ofs.exceptions();
        print_execution_durations(ofs, context.durations);
    }
}

void apply_preprocessor_definition_defaults(execution_context_t& context)
{
    auto ppds_with_defaults = util::filter(
        context.get_kernel_adapter().preprocessor_definition_details(),
        [](const auto& spd) { return spd.default_value != nullptr;});
    for(auto const& ppd : ppds_with_defaults) {
        if (not util::contains(context.options.preprocessor_definitions.valued, ppd.name)) {
            context.preprocessor_definitions.generated.valued.emplace(ppd.name, ppd.default_value);
        }
    }
}

void generate_additional_preprocessor_defines(execution_context_t& context)
{
    spdlog::debug("Generating additional preprocessor definitions");
    context.kernel_adapter_->generate_additional_preprocessor_definitions(context);
    const auto& generated = context.preprocessor_definitions.generated;
    auto nothing_generated = (generated.valued.empty() and generated.valueless.empty());
    {
        auto log_level = nothing_generated ? spdlog::level::debug : spdlog::level::info;
        spdlog::log(log_level, "Generated {} value-less and {} valued preprocessor definitions",
                    generated.valueless.size(), generated.valued.size());
    }
    for(const auto& valued_def : generated.valued) {
        spdlog::debug("Generated preprocessor definition: {}={}", valued_def.first, valued_def.second);
    }
    for(const auto& valueless_def : generated.valueless) {
        spdlog::debug("Generated preprocessor definition: {}", valueless_def);
    }
}

void finalize_preprocessor_definitions(execution_context_t& context)
{
    context.preprocessor_definitions.finalized.valueless =
        util::union_(
            context.options.preprocessor_definitions.valueless,
            context.preprocessor_definitions.generated.valueless);
    context.preprocessor_definitions.finalized.valued =
        util::incremental_map_union(
            context.options.preprocessor_definitions.valued,
            context.preprocessor_definitions.generated.valued);
}

void complete_execution(const execution_context_t& context)
{
    if (not context.options.sync_after_kernel_execution) {
        if (context.ecosystem == execution_ecosystem_t::cuda) {
            context.cuda.stream->synchronize();
        }
        else {
            context.opencl.queue.finish();
        }
    }
}

void verify_launch_configuration(execution_context_t const& context)
{
    switch(context.ecosystem) {
    case execution_ecosystem_t::opencl:
        validate_launch_configuration_<execution_ecosystem_t::opencl>(context);
        break;
    case execution_ecosystem_t::cuda:
    default:
        validate_launch_configuration_<execution_ecosystem_t::cuda>(context);
    }
}

void configure_device_and_built_kernel(execution_context_t const& context)
{
    if (not context.options.expand_shmem_carveout_if_necessary) {
        spdlog::trace("Not checking whether we need to expand the shared memory per block.");
        return;
    }
    // There may be different "knobs" we can "turn" and "switches" we could "flip";
    // for now, we only use very few of them.

    if (context.ecosystem == execution_ecosystem_t::cuda) {
        auto required_shared_memory = determine_required_shared_memory_size(context);
        enable_sufficient_shared_memory(context, required_shared_memory);
    }
}


int main(int argc, char** argv)
{
    spdlog::set_level(spdlog::level::info);
    spdlog::cfg::load_env_levels(); // support setting the logging verbosity with an environment variable

    auto parsed_cmdline_options = parse_command_line(argc, argv);

    if (parsed_cmdline_options.early_exit_action) {
        perform_early_exit_action(parsed_cmdline_options);
        exit (parsed_cmdline_options.valid ? EXIT_SUCCESS : EXIT_FAILURE);
    }

    execution_context_t context = initialize_execution_context(parsed_cmdline_options);

    if (not context.options.compile_only) {
        dealias_arguments(context);
        parse_scalars(context);
        resolve_buffer_filenames(context);
    }

    apply_preprocessor_definition_defaults(context);
    generate_additional_preprocessor_defines(context);
    finalize_preprocessor_definitions(context);
    ensure_necessary_terms_were_defined(context);
    auto build_succeeded = build_kernel(context);
    handle_compilation_log(build_succeeded, context);
    build_succeeded or die();

    maybe_write_intermediate_representation(context);

    if (context.options.compile_only) { return EXIT_SUCCESS; }

    read_input_buffers_from_files(context);
    generate_additional_scalar_arguments(context);
    validate_arguments(context);
    create_host_side_output_buffers(context);
    create_device_side_buffers(context);
    copy_input_buffers_to_device(context);

    finalize_kernel_arguments(context);
    prepare_kernel_launch_config(context);
    verify_launch_configuration(context);
    configure_device_and_built_kernel(context);

    for(run_index_t ri = 0; ri < context.options.num_runs; ri++) {
        schedule_single_run(context, ri);
    }
    complete_execution(context);
    handle_execution_durations(context);
    if (context.options.write_output_buffers_to_files) {
        copy_outputs_from_device(context);
        write_buffers_to_files(context);
    }

    spdlog::info("All done.");
}
