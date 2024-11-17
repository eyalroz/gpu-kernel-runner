#include "common_types.hpp"
#include "parsed_cmdline_options.hpp"
#include "execution_context.hpp"
#include "kernel_adapter.hpp"
#include "buffer_ops.hpp"

#include <nvrtc-related/build.hpp>
#include <nvrtc-related/execution.hpp>
#include <opencl-related/build.hpp>
#include <opencl-related/execution.hpp>
#include <opencl-related/miscellany.hpp>

#include <util/miscellany.hpp>
#include <util/cxxopts-extra.hpp>
#include <util/spdlog-extra.hpp>

#include <cxxopts/cxxopts.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/cfg/helpers.h>
#include <spdlog/cfg/env.h>

#include <system_error>
#include <iostream>
#include <vector>

template <typename... Ts>
[[noreturn]] inline bool die(std::string message_format_string = "", Ts&&... args)
{
    if(not message_format_string.empty()) {
        spdlog::critical(message_format_string, std::forward<Ts>(args)...);
    }
    exit(EXIT_FAILURE);
}

using std::size_t;
using std::string;

cxxopts::Options basic_cmdline_options(const char* program_name)
{
    cxxopts::Options options { program_name, "A runner for dynamically-compiled CUDA kernels"};
    options.add_options()
        ("l,log-level,log", "Set logging level", cxxopts::value<string>()->default_value("warning"))
        ("log-flush-threshold", "Set the threshold level at and above which the log is flushed on each message", cxxopts::value<string>()->default_value("info"))
        ("w,write-output,save-output", "Write output buffers to files", cxxopts::value<bool>()->default_value("true"))
        ("n,num-runs,runs,repetitions", "Number of times to run the compiled kernel", cxxopts::value<unsigned>()->default_value("1"))
        ("e,ecosystem,execution-ecosystem", "Execution ecosystem (CUDA or Opencl)", cxxopts::value<string>())
        ("opencl,OpenCL", "Use OpenCL", cxxopts::value<bool>())
        ("cuda,CUDA", "Use CUDA", cxxopts::value<bool>())
        ("p,platform,platform-id", "Use the OpenCL platform with the specified index", cxxopts::value<unsigned>())
        ("list-opencl-platforms,list-platforms,platforms", "List available OpenCL platforms")
        ("a,arg,argument", "Set one of the kernel's argument, keyed by name, with a serialized value for a scalar (e.g. foo=123) or a path to the contents of a buffer (e.g. bar=/path/to/data.bin)", cxxopts::value<std::vector<string>>())
        ("A,no-implicit-compilation-options,no-implicit-compile-options,suppress-default-compile-options,suppress-default-compilation-options,no-default-compile-options,no-default-compilation-options", "Avoid setting any compilation options not explicitly requested by the user", cxxopts::value<bool>()->default_value("false"))
        ("output-buffer-size,output-size,buffer-size,arg-size", "Set one of the output buffers' sizes, keyed by name, in bytes (e.g. myresult=1048576)", cxxopts::value<std::vector<string>>())
        ("d,dev,device", "Device index", cxxopts::value<int>()->default_value("0"))
        ("D,preprocessor-definition,define", "Set a preprocessor definition for NVRTC (can be used repeatedly; specify either DEFINITION or DEFINITION=VALUE)", cxxopts::value<std::vector<string>>())
        ("c,compile,compilation,compile-only", "Compile the kernel, but don't actually run it", cxxopts::value<bool>()->default_value("false"))
        ("G,debug-info,generate-debug-info,device-debug-info", "Generate debugging information for debugging the kernel (and possible avoid most/all optimizations)", cxxopts::value<bool>()->default_value("false"))
        ("P,write-ptx,write-PTX,save-ptx,save-PTX,generate-ptx,generate-PTX", "Write the intermediate representation code (PTX) resulting from the kernel compilation, to a file", cxxopts::value<bool>()->default_value("false"))
        ("ptx-output-file", "File to which to write the kernel's intermediate representation", cxxopts::value<string>())
        ("print-compilation-log", "Print the compilation log to the standard output", cxxopts::value<bool>()->default_value("false"))
        ("write-compilation-log", "Path of a file into which to write the compilation log (regardless of whether it's printed to standard output)", cxxopts::value<string>()->default_value(""))
        ("t,print-execution-durations,print-durations,print-time,print-times,execution-durations,report-duration,report-durations,time", "Print the execution duration, in nanoseconds, of each kernel invocation to the standard output", cxxopts::value<bool>()->default_value("false"))
        ("write-execution-durations,write-durations,write-time,write-times", "Path of a file into which to write the execution durations, in nanoseconds, for each kernel invocation (regardless of whether they're printed to standard output)", cxxopts::value<string>()->default_value(""))
        ("generate-source-line-info,generate-line-info,source-line-info,line-info", "Add source line information to the intermediate representation code (PTX)", cxxopts::value<bool>())
        ("b,local-work-size,block-dims,blockdim,block,block-dimensions", "Set grid block dimensions in threads  (OpenCL: local work size); a comma-separated list", cxxopts::value<std::vector<unsigned>>() )
        ("g,grid-dims,griddim,grid,grid-dimensions", "Set grid dimensions in blocks; a comma-separated list", cxxopts::value<std::vector<unsigned>>() )
        ("o,overall-work-size,overall-dimensions,overall-dims,overall,overall-grid-dimensions", "Set grid dimensions in threads (OpenCL: global work size); a comma-separated list", cxxopts::value<std::vector<unsigned>>() )
        ("O,compile-option,append-compile-option,compilation-option,extra-compile-option,extra-compilation-option,append-compilation-option", "Append an arbitrary extra compilation option", cxxopts::value<std::vector<string>>())
        ("S,dynamic-shmem-size,dynamic-shmem,dshmem,dynamic-shared-memory-size", "Force specific amount of dynamic shared memory", cxxopts::value<unsigned>() )
        ("W,overwrite,overwrite-output-files", "Overwrite the files for buffer and/or PTX output if they already exists", cxxopts::value<bool>()->default_value("false"))
        ("i,direct-include,include", "Include a specific file into the kernels' translation unit", cxxopts::value<std::vector<string>>())
        ("I,include-search-path,include-dir,include-search-dir,include-path", "Add a directory to the search paths for header files included by the kernel (can be used repeatedly)", cxxopts::value<std::vector<string>>())
        ("s,kernel-source-file,kernel-file,source-code,kernel-source-code,kernel-source,source,source-file", "Path to CUDA source file with the kernel function to compile; may be absolute or relative to the sources dir", cxxopts::value<string>())
        ("k,function,kernel-function", "Name of function within the source file to compile and run as a kernel (if different than the key)", cxxopts::value<string>())
        ("K,key,kernel-key,adapter,kernel-adapter,adapter-key,kernel-adapter-key", "The key identifying the kernel among all registered runnable kernels", cxxopts::value<string>())
        ("L,list-kernels,list,all-kernels,kernels,adapters,list-adapters", "List the (keys of the) kernels which may be run with this program")
        ("z,zero-output-buffers,zero-outputs", "Set the contents of output(-only) buffers to all-zeros", cxxopts::value<bool>()->default_value("false"))
        ("C,clear-l2-cache,clear-cache,clear-l2,clear-L2,clear-L2-cache,clear-cache-before-run", "(Attempt to) clear the GPU L2 cache before each run of the kernel", cxxopts::value<bool>()->default_value("false"))
        ("sync-after-execution,sync-after-invocation", "Have the GPU finish all work for a given run before scheduling the next", cxxopts::value<bool>()->default_value("false"))
        ("sync-after-buffer-op,sync-after-buffer-write", "Have the GPU finish all work for a given run after scheduling a buffer operation", cxxopts::value<bool>()->default_value("false"))
        ("language-standard,std", "Set the language standard to use for CUDA compilation (options: c++11, c++14, c++17)", cxxopts::value<string>())
        ("input-buffer-directory,in-dir,input-buffer-dir,input-buffers-directory,input-buffers-dir,inbuf-dir,inbufs,inbufs-dir,inbuf-directory,inbufs-directory,in-directory", "Base location for locating input buffers", cxxopts::value<string>()->default_value( filesystem::current_path().native() ))
        ("output-buffer-directory,output-buffer-dir,output-buffers-directory,output-buffers-dir,outbuf-dir,outbufs,outbufs-dir,outbuf-directory,outbufs-directory,out-directory", "Base location for writing output buffers", cxxopts::value<string>()->default_value( filesystem::current_path().native() ))
        ("kernel-sources-dir,kernel-source-dir,kernel-source-directory,kernel-sources-directory,kernel-sources,source-dir,sources", "Base location for locating kernel source files", cxxopts::value<string>()->default_value( filesystem::current_path().native() ))
        ("h,help", "Print usage information")
        ;
    return options;
}

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

    if (context.ecosystem == execution_ecosystem_t::cuda) {
        auto cuda_include_dir = locate_cuda_include_directory();
        if (cuda_include_dir) {
            spdlog::debug("Using CUDA include directory {}", cuda_include_dir.value());
            context.finalized_include_dir_paths.emplace_back(cuda_include_dir.value());
        }
        else {
            spdlog::warn("Cannot locate CUDA include directory - trying to build the kernel with it missing.");
        }
    }
    // What about OpenCL? Should it get some defaulted include directory?
}

[[noreturn]] void print_help_and_exit(
    const cxxopts::Options &options,
    bool user_asked_for_help = true)
{
    auto &ostream = user_asked_for_help ? std::cout : std::cerr;
    ostream << options.help() << "\n";
    exit(user_asked_for_help ? EXIT_SUCCESS : EXIT_FAILURE);
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
    if(device_id < 0 or device_id >= cuda::device::count())
        die ("Please specify a valid device index (in the range 0.. {})", cuda::device::count()-1);
}

void print_registered_kernel_keys() {
    auto &scf = kernel_adapter::get_subclass_factory();
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
    std::cout << std::endl;
}

parsed_cmdline_options_t parse_command_line(int argc, char** argv)
{
    // TODO: Break off lots of smaller functions handling the just-parsed parameters
    // to populate various data structures

    auto program_name = argv[0];
    cxxopts::Options options = basic_cmdline_options(program_name);
    options.allow_unrecognised_options();

    // Note that the following will be printed based only on the compiled-in
    // default log level
    spdlog::debug("Parsing the command line for non-kernel-specific options.");
    auto parse_result = non_consumptive_parse(options, argc, argv);

    parsed_cmdline_options_t parsed_options;

    bool user_asked_for_help = contains(parse_result, "help");
        // Note that we will not immediately provide the help, because if we can figure
        // out what kernel was asked for, we will want to provide help regarding
        // the kernel-specific command-line arguments
    bool user_asked_for_list_of_kernels = contains(parse_result, "list-kernels");
    bool user_asked_for_list_of_platforms = contains(parse_result, "list-opencl-platforms");
    struct { bool key, function_name, source_file_path; } got {
        contains(parse_result, "kernel-key"),
        contains(parse_result, "kernel-function"),
        contains(parse_result, "kernel-source")
    };

    if (user_asked_for_help) {
        print_help_and_exit(options);
    }

    if (user_asked_for_list_of_kernels) {
        print_registered_kernel_keys();
        exit(EXIT_SUCCESS);
    }

    if (user_asked_for_list_of_platforms) {
        list_opencl_platforms();
        exit(EXIT_SUCCESS);
    }

    (got.key or got.function_name or got.source_file_path) or die(
        "No kernel key was specified, nor enough information provided "
        "to deduce the kernel key, source filename and kernel function name");

    // No need to exit (at least not until second parsing), let's
    // go ahead and collect the parsed data

    auto log_level_name = parse_result["log-level"].as<string>();
    auto log_level = spdlog::level::from_str(log_level_name);
    if (spdlog::level_is_at_least(spdlog::level::debug)) {
        spdlog::log(spdlog::level::debug, "Setting log level to {}", log_level_name);
    }
    spdlog::set_level(log_level);

    auto log_flush_threshold_name = parse_result["log-flush-threshold"].as<string>();
    auto log_flush_threshold = spdlog::level::from_str(log_flush_threshold_name);
    spdlog::debug("Setting log level flush threshold to \"{}\"", log_flush_threshold_name);
    spdlog::flush_on(log_flush_threshold);

    //---------------------------------------
    // CUDA and OpenCL-related options

    parsed_options.gpu_ecosystem = [&]() -> execution_ecosystem_t {
        // Yes, this IILE is kind of messy. You could consider dropping --cuda nnd
        // --opencl entirely

        bool specified_cuda = contains(parse_result, "cuda");
        bool specified_opencl = contains(parse_result, "opencl");
        bool use_opencl = specified_opencl and parse_result["opencl"].as<bool>();
        bool use_cuda = specified_cuda and parse_result["cuda"].as<bool>();

        optional<execution_ecosystem_t> specified_exec_ecosystem = {};
        if (contains(parse_result, "execution-ecosystem")) {
            auto unparsed_see = parse_result["execution-ecosystem"].as<string>();
            if (util::case_insensitive_equals(unparsed_see, ecosystem_name(execution_ecosystem_t::cuda))) {
                specified_exec_ecosystem = execution_ecosystem_t::cuda;
            }
            else if (util::case_insensitive_equals(unparsed_see, ecosystem_name(execution_ecosystem_t::opencl))) {
                specified_exec_ecosystem = execution_ecosystem_t::opencl;
            }
            else die("Invalid execution ecosystem: {}", unparsed_see);
            if ((use_cuda and specified_exec_ecosystem.value() == execution_ecosystem_t::opencl) or
                (use_opencl and specified_exec_ecosystem.value() == execution_ecosystem_t::cuda))
            {
                die("Execution ecosystem specifier options disagree");
            }
        }
        if (specified_exec_ecosystem) return specified_exec_ecosystem.value();
        if(not use_cuda and not use_opencl) die("Please specify an execution system to use (either CUDA or OpenCL).\n");
        if (use_cuda and use_opencl) {
            die("Execution ecosystem specifier options disagree");
        }
        return use_cuda ? execution_ecosystem_t::cuda : execution_ecosystem_t::opencl;
    }();
    spdlog::debug("Using the {} execution ecosystem.", ecosystem_name(parsed_options.gpu_ecosystem));

    parsed_options.gpu_device_id = parse_result["device"].as<int>();
    if (parsed_options.gpu_device_id < 0) die("Please specify a non-negative device index");

    if (contains(parse_result, "platform-id")) {
        if (parsed_options.gpu_ecosystem == execution_ecosystem_t::cuda) {
            die("CUDA does not support multiple per-machine platforms; thus any 'platform-id' value is unacceptable");
        }
            // TODO: We could theoretically just ignore this, or warn later on
        parsed_options.platform_id = parse_result["platform-id"].as<unsigned>();
    }
    else {
        parsed_options.platform_id = 0;
    }

    //---------------------------------------

    string source_file_path;

    if (got.source_file_path) {
        source_file_path = parse_result["kernel-source"].as<string>();
    }

    if (got.function_name) {
        parsed_options.kernel.function_name = parse_result["kernel-function"].as<string>();
        if (not util::is_valid_identifier(parsed_options.kernel.function_name)) {
            throw std::invalid_argument("Function name must be non-empty.");
        }
    }
    if (got.key) {
        parsed_options.kernel.key = parse_result["kernel-key"].as<string>();
        if (parsed_options.kernel.key.empty()) {
            throw std::invalid_argument("Kernel key may not be empty.");
        }
    }

    string clipped_key = [&]() {
        if (got.key) {
            auto pos_of_last_invalid = parsed_options.kernel.key.find_last_of("/-;.[]{}(),");
            return parsed_options.kernel.key.substr(
                (pos_of_last_invalid == string::npos ? 0 : pos_of_last_invalid + 1), string::npos);
        }
        return string{};
    }();

    if (not got.function_name and got.source_file_path and not got.key) {
        struct { bool key, source_file_path; } usable_as_function_name =
        {
            util::is_valid_identifier(clipped_key),
            util::is_valid_identifier(parsed_options.kernel.source_file.filename().native())
        };
        if (usable_as_function_name.source_file_path and not usable_as_function_name.key) {
            parsed_options.kernel.function_name = parsed_options.kernel.source_file.filename().native();
            spdlog::info("Inferring the kernel function name from the kernel source filename: '{}'",
                parsed_options.kernel.function_name);
        }
        else if (usable_as_function_name.key and not usable_as_function_name.source_file_path) {
            parsed_options.kernel.function_name = clipped_key;
            spdlog::info("Inferring the kernel function name from the kernel key: '{}'",
                parsed_options.kernel.function_name);
        }
    }

    if (not got.key and (got.source_file_path or got.function_name)) {
        if (got.source_file_path) {
            parsed_options.kernel.key = parsed_options.kernel.source_file.filename().native();
        }
        else {
            parsed_options.kernel.key = parsed_options.kernel.function_name;
            spdlog::info("Inferring the kernel key from the kernel function name: '{}'", parsed_options.kernel.key);
        }
    }
    spdlog::debug("Using kernel key: {}", parsed_options.kernel.key);

    if (not got.source_file_path and (got.key or got.function_name)) {
        auto suffix = kernel_source_file_suffix(parsed_options.gpu_ecosystem);
        source_file_path = (got.function_name ? parsed_options.kernel.function_name : clipped_key) + '.' + suffix;
    }

    // Complete the source file into an absolute path

    parsed_options.kernel_sources_base_path = parse_result["kernel-sources-dir"].as<string>();
    parsed_options.kernel.source_file = maybe_prepend_base_dir(
           parsed_options.kernel_sources_base_path, source_file_path);
    (got.source_file_path or filesystem::exists(parsed_options.kernel.source_file)) or die(
        "No source file specified, and inferred source file path does not exist: {}",
        parsed_options.kernel.source_file.native());

    spdlog::debug("Resolved kernel source file path: {}", parsed_options.kernel.source_file.native());

    // Note: Doing nothing if the kernel source file is missing. Since we must have gotten
    // the kernel name, we'll prefer printing usage information with kernel-specific options,
    // alongside the error message about the missing kernel file

    // The following can't fail due to defaults

    parsed_options.num_runs = parse_result["num-runs"].as<unsigned>();

    parsed_options.overwrite_allowed = parse_result["overwrite"].as<bool>();
    if (parsed_options.overwrite_allowed and
        (parsed_options.write_ptx_to_file or
         parsed_options.write_output_buffers_to_files or
         parsed_options.write_compilation_log) )
    {
        spdlog::info("Existing output files will be overwritten.");
    }

    parsed_options.buffer_base_paths.input = parse_result["input-buffer-dir"].as<string>();
    parsed_options.buffer_base_paths.output = parse_result["output-buffer-dir"].as<string>();
    parsed_options.write_ptx_to_file = parse_result["write-ptx"].as<bool>();
    parsed_options.set_default_compilation_options = not parse_result["no-default-compile-options"].as<bool>();
    parsed_options.generate_source_line_info =
        (contains(parse_result, "generate-line-info")) ?
        parse_result["no-default-compile-options"].as<bool>() :
        parsed_options.set_default_compilation_options;
        // If we were explicitly instructed about this, follow the instruction. Otherwise, we would _like_
        // to get line info, and do so by default, but the user may also want us to avoid setting
        // _any_ unnecessary compilation options by default, and we need to respect that too.
        // We don't have this logic elsewhere, since other compilation flags don't default to true...

    if (parsed_options.write_ptx_to_file) {
        if (contains(parse_result, "ptx-output-file")) {
            parsed_options.ptx_output_file = parse_result["ptx-output-file"].as<string>();
            if (filesystem::exists(parsed_options.ptx_output_file)) {
                if (not parsed_options.overwrite_allowed) {
                    throw std::invalid_argument("Specified PTX output file "
                        + parsed_options.ptx_output_file.native() + " exists, and overwrite is not allowed.");
                }
                // Note that there could theoretically be a race condition in which the file gets created
                // between our checking for its existence and our wanting to write to it after compilation.
            }
        }
    }
    parsed_options.print_execution_durations = parse_result["print-execution-durations"].as<bool>();
    if (contains(parse_result, "write-execution-durations")) {
        parsed_options.execution_durations_file = parse_result["write-execution-durations"].as<string>();
        if (filesystem::exists(parsed_options.execution_durations_file)) {
            if (not parsed_options.overwrite_allowed) {
                throw std::invalid_argument("Specified execution durations file "
                    + parsed_options.execution_durations_file.native() + " exists, and overwrite is not allowed.");
            }
            // Note that there could theoretically be a race condition in which the file gets created
            // between our checking for its existence and our wanting to write to it after compilation.
        }
    }

    parsed_options.always_print_compilation_log = parse_result["print-compilation-log"].as<bool>();
    if (contains(parse_result, "write-compilation-log")) {
        parsed_options.compilation_log_file = parse_result["write-compilation-log"].as<string>();
        if (filesystem::exists(parsed_options.compilation_log_file)) {
            if (not parsed_options.overwrite_allowed) {
                throw std::invalid_argument("Specified compilation log file "
                    + parsed_options.compilation_log_file.native() + " exists, and overwrite is not allowed.");
            }
            // Note that there could theoretically be a race condition in which the file gets created
            // between our checking for its existence and our wanting to write to it after compilation.
        }
    }

    if (not filesystem::exists(parsed_options.buffer_base_paths.output)) {
        spdlog::info("Trying to create path to missing directory {}", parsed_options.buffer_base_paths.output.native());
        filesystem::create_directories(parsed_options.buffer_base_paths.output)
            or die("Failed creating path: {}", parsed_options.buffer_base_paths.output.native());
    }

    for (const auto& path : {
             parsed_options.buffer_base_paths.input,
             parsed_options.buffer_base_paths.output,
             parsed_options.kernel_sources_base_path
         } )
    {
        if (not filesystem::exists(path)) die("No such directory {}", path.native());
        if (not filesystem::is_directory(path)) die("{} is not a directory.", path.native());
    }

    parsed_options.write_output_buffers_to_files = parse_result["write-output"].as<bool>();
    parsed_options.write_ptx_to_file = parse_result["write-ptx"].as<bool>();
    parsed_options.always_print_compilation_log = parse_result["print-compilation-log"].as<bool>();


    parsed_options.compile_only = parse_result["compile-only"].as<bool>();

    if (parse_result.count("language-standard") > 0) {
        auto language_standard = parse_result["language-standard"].as<string>();
        language_standard = util::to_lowercase(language_standard);
        if ((language_standard == "c++11") or
            (language_standard == "c++14") or
            (language_standard == "c++17")) {
            parsed_options.language_standard = language_standard;
        }
        else {
            die("Unsupported language standard for kernel compilation: {}", language_standard);
        }
    }
    else {
        parsed_options.language_standard = nullopt;
    }

    if (parse_result.count("append-compilation-option") > 0) {
        parsed_options.extra_compilation_options = parse_result["append-compilation-option"].as<std::vector<string>>();
    }

    parsed_options.generate_debug_info = parse_result["generate-debug-info"].as<bool>();
    parsed_options.zero_output_buffers = parse_result["zero-output-buffers"].as<bool>();
    parsed_options.clear_l2_cache = parse_result["clear-l2-cache"].as<bool>();
    parsed_options.sync_after_kernel_execution = parse_result["sync-after-execution"].as<bool>();
    parsed_options.sync_after_buffer_op = parse_result["sync-after-buffer-op"].as<bool>();

    if (parse_result.count("block-dimensions") > 0) {
        auto dims = parse_result["block-dimensions"].as<std::vector<unsigned>>();
        if (dims.empty() or dims.size() > 3) {
            throw std::invalid_argument("Invalid forced block dimensions: Found 0 dimensions");
        }
        while (dims.size() < 3) { dims.push_back(1u); }
        parsed_options.forced_launch_config_components.block_dimensions = { dims[0], dims[1], dims[2] };
    }

    if (parse_result.count("grid-dimensions") > 0 and parse_result.count("overall-grid-dimensions") > 0) {
        die("You can specify the grid dimensions either in blocks or in overall threads, but not both");
    }

    if (parse_result.count("grid-dimensions") > 0) {
        auto dims = parse_result["grid-dimensions"].as<std::vector<unsigned>>();
        if (dims.empty() or dims.size() > 3) {
            die("Invalid forced grid dimensions in blocks: Got {} dimensions", dims.size());
        }
        while (dims.size() < 3) { dims.push_back(1u); }
        parsed_options.forced_launch_config_components.grid_dimensions = { dims[0], dims[1], dims[2] };
    }

    if (parse_result.count("overall-grid-dimensions") > 0) {
        auto dims = parse_result["overall-grid-dimensions"].as<std::vector<unsigned>>();
        if (dims.empty() or dims.size() > 3) {
            die("Invalid forced overall grid dimensions: Got {} dimensions", dims.size());
        }
        while (dims.size() < 3) { dims.push_back(1u); }
        parsed_options.forced_launch_config_components.overall_grid_dimensions = { dims[0], dims[1], dims[2] };
    }

    if (parse_result.count("dynamic-shared-memory-size") > 0) {
        parsed_options.forced_launch_config_components.dynamic_shared_memory_size =
            parse_result["dynamic-shared-memory-size"].as<unsigned>();
    }

    if (parse_result.count("define") > 0) {
        const auto& parsed_defines = parse_result["define"].as<std::vector<string>>();
        for(const auto& definition : parsed_defines) {
            auto equals_pos = definition.find('=');
            switch (equals_pos) {
                case string::npos:
                    spdlog::trace("Preprocessor definition: {}", definition);
                    parsed_options.preprocessor_definitions.valueless.emplace(definition);
                    break;
                case 0:
                    die("Preprocessor definition specified with an empty name: \"{}\" ", definition);
                default:
                    auto defined_term = definition.substr(0, equals_pos);
                    auto value = definition.substr(equals_pos + 1);
                    spdlog::trace("Preprocessor definition: {}={}", defined_term, value);
                    parsed_options.preprocessor_definitions.valued[defined_term] = value;
            }
        }
    }

    if (parse_result.count("output-buffer-size") > 0) {
        const auto& buffer_size_settings = parse_result["output-buffer-size"].as<std::vector<string>>();
        for(const auto& setting : buffer_size_settings) {
            auto equals_pos = setting.find('=');
            switch (equals_pos) {
                case string::npos:
                case 0:
                    die("Output buffer size setting is not in a name=size format: \"{}\"", setting);
                default:
                    auto buffer_name = setting.substr(0, equals_pos);
                    auto buffer_size = std::stoul(setting.substr(equals_pos + 1));
                    spdlog::trace("Size of output buffer '{}' set to {} bytes", buffer_name, buffer_size);
                    parsed_options.output_buffer_sizes.emplace(buffer_name, buffer_size);
            }
        }
    }

    if (parse_result.count("argument") > 0) {
        const auto& kernel_arguments_assignments = parse_result["argument"].as<std::vector<string>>();
        for(const auto& kernel_arg_definition : kernel_arguments_assignments) {
            auto equals_pos = kernel_arg_definition.find('=');
            switch(equals_pos) {
                case string::npos:
                    die("Kernel argument name/alias \"{}\" specified without a value", kernel_arg_definition);
                case 0:
                    die("Kernel argument value specified with an empty name/alias: \"{}\" ", kernel_arg_definition);
                default:
                    auto param_name = kernel_arg_definition.substr(0, equals_pos);
                    auto value = kernel_arg_definition.substr(equals_pos+1);
                    parsed_options.aliased_kernel_arguments.emplace(param_name, value);
            }
        }
    }

    if (parse_result.count("include-path") > 0) {
        parsed_options.include_dir_paths = parse_result["include-path"].as<std::vector<string>>();
        for (const auto& p : parsed_options.include_dir_paths) {
            spdlog::trace("User-specified include path: {}", p);
        }
    }
    if (parse_result.count("include") > 0) {
        parsed_options.preinclude_files = parse_result["include"].as<std::vector<string>>();
        for (const auto& p : parsed_options.preinclude_files) {
            spdlog::trace("User-specified pre-include file: {}", p);
        }
    }

    if (not kernel_adapter::can_produce_subclass(string(parsed_options.kernel.key))) {
        die("No kernel adapter is registered for key {}", parsed_options.kernel.key);
    }

    parsed_options.time_with_events =
        parsed_options.print_execution_durations or (not parsed_options.execution_durations_file.empty())
        or spdlog::level_is_at_least(spdlog::level::info);

    return parsed_options;
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
            context.cuda.mangled_kernel_signature = std::move(result.mangled_signature);
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
        if (buffer_details.size_calculator) {
            auto calculated = apply_size_calc(buffer_details.size_calculator, context);
            (calculated == buffer.size()) or die(
                "Input buffer {} expected has size {} bytes, but its size calculator requires a size of {}",
                buffer_details.name, buffer.size(), calculated);
            spdlog::trace("Input buffer {} has size {} bytes, as expected by size calculator}",
                buffer_details.name, buffer.size());
        } else{
            spdlog::trace("No size calculator for buffer {}; assuming size is valid", buffer_details.name);
        }
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

int main(int argc, char** argv)
{
    spdlog::set_level(spdlog::level::info);
    spdlog::cfg::load_env_levels(); // support setting the logging verbosity with an environment variable

    auto parsed_cmdline_options = parse_command_line(argc, argv);

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
    auto log = context.compilation_log.value();
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
