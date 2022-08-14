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
#include <cxx-prettyprint/prettyprint.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/cfg/helpers.h>
#include <spdlog/cfg/env.h>

#include <system_error>
#include <cerrno>
#include <iostream>
#include <cstdio>
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
        ("log-flush-threshold", "Set the threshold level at and above which the log is flushed on each message",
            cxxopts::value<string>()->default_value("info"))
        ("w,write-output,save-output", "Write output buffers to files", cxxopts::value<bool>()->default_value("true"))
        ("n,num-runs,runs,repetitions", "Number of times to run the compiled kernel", cxxopts::value<unsigned>()->default_value("1"))
        ("opencl,OpenCL", "Use OpenCL", cxxopts::value<bool>())
        ("cuda,CUDA", "Use CUDA", cxxopts::value<bool>())
        ("p,platform-id,platform", "Use the OpenCL platform with the specified index", cxxopts::value<unsigned>())
        ("a,argument,arg", "Set one of the kernel's argument, keyed by name, with a serialized value for a scalar (e.g. foo=123) or a path to the contents of a buffer (e.g. bar=/path/to/data.bin)", cxxopts::value<std::vector<string>>())
        ("d,device,dev", "Device index", cxxopts::value<int>()->default_value("0"))
        ("D,preprocessor-definition,define", "Set a preprocessor definition for NVRTC (can be used repeatedly; specify either DEFINITION or DEFINITION=VALUE)", cxxopts::value<std::vector<string>>())
        ("c,compile-only,compile,compilation", "Compile the kernel, but don't actually run it", cxxopts::value<bool>()->default_value("false"))
        ("G,debug-mode,device-debug,debug", "Have the NVRTC compile the kernel in debug mode (no optimizations)", cxxopts::value<bool>()->default_value("false"))
        ("P,write-ptx", "Write the intermediate representation code (PTX) resulting from the kernel compilation, to a file", cxxopts::value<bool>()->default_value("false"))
        ("ptx-output-file", "File to which to write the kernel's intermediate representation", cxxopts::value<string>())
        ("print-compilation-log", "Print the compilation log to the standard output", cxxopts::value<bool>()->default_value("false"))
        ("write-compilation-log", "Path of a file into which to write the compilation log (regardless of whether it's printed to standard output)", cxxopts::value<string>()->default_value(""))
        ("print-execution-durations,print-durations,print-times", "Print the execution duration, in nanoseconds, of each kernel invocation to the standard output", cxxopts::value<bool>()->default_value("false"))
        ("write-execution-durations,write-durations,write-times", "Path of a file into which to write the execution durations, in nanoseconds, for each kernel invocation (regardless of whether they're printed to standard output)", cxxopts::value<string>()->default_value(""))
        ("generate-line-info,line-info", "Add source line information to the intermediate representation code (PTX)", cxxopts::value<bool>()->default_value("true"))
        ("b,block-dimensions,block-dims,blockdim,block", "Set grid block dimensions in threads  (OpenCL: local work size); a comma-separated list", cxxopts::value<std::vector<unsigned>>() )
        ("g,grid-dimensions,grid-dims,griddim,grid", "Set grid dimensions in blocks; a comma-separated list", cxxopts::value<std::vector<unsigned>>() )
        ("o,overall-grid-dimensions,overall-dimensions,overall-dims,overall", "Set grid dimensions in threads (OpenCL: global work size); a comma-separated list", cxxopts::value<std::vector<unsigned>>() )
        ("O,append-compilation-option,compile-option,append-compile-option", "Append an arbitrary extra compilation option", cxxopts::value<std::vector<string>>())
        ("S,dynamic-shared-memory-size,dynamic-shmem-size,dynamic-shmem,dshmem", "Force specific amount of dynamic shared memory", cxxopts::value<unsigned>() )
        ("W,overwrite,overwrite-output-files", "Overwrite the files for buffer and/or PTX output if they already exists", cxxopts::value<bool>()->default_value("false"))
        ("i,include,direct-include", "Include a specific file into the kernels' translation unit", cxxopts::value<std::vector<string>>())
        ("I,include-path,include-search-path,include-dir,include-search-dir", "Add a directory to the search paths for header files included by the kernel (can be used repeatedly)", cxxopts::value<std::vector<string>>())
        ("s,kernel-source,kernel-source-file,kernel-file,source-code,kernel-source-code", "Path to CUDA source file with the kernel function to compile; may be absolute or relative to the sources dir", cxxopts::value<string>())
        ("k,kernel-function,function", "Name of function within the source file to compile and run as a kernel (if different than the key)", cxxopts::value<string>())
        ("K,kernel-key,key", "The key identifying the kernel among all registered runnable kernels", cxxopts::value<string>())
        ("L,list-kernels,list,all-kernels", "List the (keys of the) kernels which may be run with this program")
        ("z,zero-output-buffers,zero-outputs", "Set the contents of output(-only) buffers to all-zeros", cxxopts::value<bool>()->default_value("false"))
        ("language-standard,std", "Set the language standard to use for CUDA compilation (options: c++11, c++14, c++17)", cxxopts::value<string>())
        ("input-buffer-dir,inbufs", "Base location for locating input buffers", cxxopts::value<string>()->default_value( filesystem::current_path().native() ))
        ("output-buffer-dir,outbufs", "Base location for writing output buffers", cxxopts::value<string>()->default_value( filesystem::current_path().native() ))
        ("kernel-sources-dir,sources", "Base location for locating kernel source files", cxxopts::value<string>()->default_value( filesystem::current_path().native() ))
        ("h,help", "Print usage information")
        ;
    return options;
}

void ensure_necessary_terms_were_defined(const execution_context_t& context)
{
    const auto& ka = *context.kernel_adapter_;
    auto required_terms = util::transform_if<std::unordered_set<string>>(
        ka.preprocessor_definition_details(),
        [](const auto& sad) { return sad.required; },
        [](const auto& sad) { return string{sad.name};});
    auto defined_valued_terms = util::keys(context.options.preprocessor_value_definitions);
    auto all_defined_terms = util::union_(defined_valued_terms, context.options.preprocessor_definitions);
    auto required_but_undefined = util::difference(required_terms, all_defined_terms);
    if (not required_but_undefined.empty()) {
        std::ostringstream oss;
        oss << required_but_undefined;
        die("The following preprocessor definitions must be specified, but have not been: {}", oss.str());
    }
}

void collect_include_paths(execution_context_t& context)
{
    // Note the relative order in which we place the includes; it is non-trivial.
    context.finalized_include_dir_paths = context.options.include_dir_paths;

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

void resolve_buffer_filenames(execution_context_t& context)
{
    // if (context.kernel_adapter_.get() == nullptr) { throw std::runtime_error("Null kernel adapter pointer"); }
    const auto& ka = *(context.kernel_adapter_.get());

    const auto& args = context.options.kernel_arguments;
    auto params_with_args = util::keys(args);
//    auto in_and_inout_buffers =  buffer_names(ka, parameter_direction_t::input, parameter_direction_t::inout);
    for(const auto& buffer : ka.buffer_details()) {
        const auto &name = buffer.name;
        auto got_arg = util::contains(params_with_args, name);
        if (not got_arg) {
            spdlog::debug("Filename for buffer '{}' not specified; using fallback names.");
        }
        if (is_input(buffer)) {
            const auto default_filename = name;
            auto filename = got_arg ? args.at(name) : default_filename;
            spdlog::trace("Filename for input buffer '{}': {}", name, filename);
            context.buffers.filenames.inputs[name] = filename;
        }
        if (context.options.write_output_buffers_to_files and is_output(buffer)) {
            const auto default_filename = fmt::format("{}.out", name);
            auto filename = is_input(buffer) ?
                fmt::format("{}.out", context.buffers.filenames.inputs[name]) :
                (got_arg ? args.at(name) : default_filename);
            context.buffers.filenames.outputs[name] = filename;

            // TODO: Move this verification elsewhere
            if (filesystem::exists(filename)) {
                if (not context.options.overwrite_allowed and not context.options.compile_only) {
                    die("Writing the contents of output buffer {} would overwrite an existing file: ",
                        name, filename);
                }
                spdlog::info("Output buffer '{}' will overwrite {}", name, filename);
            }
            // Note that if the output file gets created while the kernel runs, we might miss this fact
            // when trying to write to it.
            spdlog::trace("Filename for output buffer '{}': {}", name, filename);
        }
    }
}

// Note: We need the kernel adapter, and can't just be satisfied with the argument details,
// because the adapter might have overridden the parsing method with something more complex.
// If we eventually decide that's not a useful ability to have, we can avoid passing the
// adapter to this function.
void parse_scalars(execution_context_t &context)
{
    const auto& args = context.options.kernel_arguments;
    const kernel_adapter &kernel_adapter = (*context.kernel_adapter_.get());
    auto params_with_args = util::keys(args);
    {
        std::ostringstream oss;
        oss << params_with_args;
        spdlog::trace("Arguments we specified for parameters {}", oss.str());
    }
    auto all_scalar_details = kernel_adapter.scalar_parameter_details();
    for(const auto& spd : all_scalar_details ) {
        std::string param_name { spd.name };
        if (not util::contains(params_with_args, param_name)) {
            if (not spd.required) {
                spdlog::trace("No argument provided for kernel parameter '{}'.", param_name);
                continue;
            }
            die("Required scalar parameter '{}' for kernel '{}' was not specified.\n\n", param_name, kernel_adapter.key());
        }
        // TODO: Consider not parsing anything at this stage, and just marshaling all the scalar arguments together.
        auto& arg_value = args.at(param_name);
        spdlog::trace("Parsing argument for scalar parameter '{}' from \"{}\"", param_name, arg_value);
        context.scalar_input_arguments.raw[param_name] = arg_value;
        context.scalar_input_arguments.typed[param_name] =
            kernel_adapter.parse_cmdline_scalar_argument(spd, arg_value);
        spdlog::trace("Successfully parsed argument for scalar parameter '{}'.", param_name);
    }
}

void ensure_gpu_device_validity(
    execution_ecosystem_t  ecosystem,
    optional<unsigned>     platform_id,
    int                    device_id,
    bool                   need_ptx)
{
    std::size_t device_count;

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
    struct { bool key, function_name, source_file_path; } got;

    got.source_file_path  = contains(parse_result, "kernel-source");
    got.function_name     = contains(parse_result, "kernel-function");
    got.key               = contains(parse_result, "kernel-key");

    // Need to exit?

    if (user_asked_for_list_of_kernels) {
        print_registered_kernel_keys();
        exit(EXIT_SUCCESS);
    }

    if (not (got.key or got.function_name or got.source_file_path)) {
        if (user_asked_for_help) {
            print_help_and_exit(options);
        }
        else die(
            "No kernel key was specified, nor enough information provided "
            "to deduce the kernel key, source filename and kernel function name");
    }

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

    bool specified_cuda = contains(parse_result, "cuda");
    bool specified_opencl = contains(parse_result, "opencl");
    bool use_opencl = specified_opencl ? parse_result["opencl"].as<bool>() : false;
    bool use_cuda = specified_cuda ? parse_result["cuda"].as<bool>() : true;
    if(not use_cuda and not use_opencl) die("Please specify either CUDA or OpenCL to be used.\n");
    if (use_cuda and use_opencl) {
        if(specified_cuda and specified_opencl) die("Please specify either CUDA or OpenCL, not both.\n");
        use_cuda = false;
    }
    parsed_options.gpu_ecosystem = use_cuda ? execution_ecosystem_t::cuda : execution_ecosystem_t::opencl;
    spdlog::debug("Using the {} execution ecosystem.", ecosystem_name(parsed_options.gpu_ecosystem));

    parsed_options.gpu_device_id = parse_result["device"].as<int>();
    if (parsed_options.gpu_device_id < 0) die("Please specify a non-negative device index");

    if (contains(parse_result, "platform-id")) {
        if (not use_opencl) die("CUDA does not support multiple per-machine platforms; thus any 'platform-id' value is unacceptable");
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
    // if we haven't got the function name, but have got the key - we'll factory-produce the
    // adapter, then user it get the key.

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
    if (not got.source_file_path and not filesystem::exists(parsed_options.kernel.source_file)) {
        spdlog::critical(
            string("No source file specified, and inferred source file path does not exist") +
            (user_asked_for_help ? ", so kernel-specific help cannot be provided" : "") + ": {}",
            parsed_options.kernel.source_file.native());
        if (user_asked_for_help) {
            print_help_and_exit(options);
        }
        else die();
    }
    spdlog::debug("Resolved kernel source file path: {}", parsed_options.kernel.source_file.native());

    // Note: Doing nothing if the kernel source file is missing. Since we must have gotten
    // the kernel name, we'll prefer printing usage information with kernel-specific options,
    // alongside the error message about the missing kernel file

    // The following can't fail due to defaults

    parsed_options.num_runs = parse_result["num-runs"].as<unsigned>();

    parsed_options.overwrite_allowed = parse_result["overwrite"].as<bool>();
    spdlog::info("Existing output files will be overwritten.");

    parsed_options.buffer_base_paths.input = parse_result["input-buffer-dir"].as<string>();
    parsed_options.buffer_base_paths.output = parse_result["output-buffer-dir"].as<string>();
    parsed_options.write_ptx_to_file = parse_result["write-ptx"].as<bool>();
    parsed_options.generate_line_info = parse_result["generate-line-info"].as<bool>();
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
    if (parse_result.count("append-compilation-option") > 0) {
        parsed_options.extra_compilation_options = parse_result["append-compilation-option"].as<std::vector<string>>();
    }

    parsed_options.compile_in_debug_mode = parse_result["debug-mode"].as<bool>();
    parsed_options.zero_output_buffers = parse_result["zero-output-buffers"].as<bool>();

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

//    parsed_options.compare_outputs_against_expected = parse_results["compare-outputs"].as<string>();

    if (parse_result.count("define") > 0) {
        const auto& parsed_defines = parse_result["define"].as<std::vector<string>>();
        for(const auto& definition : parsed_defines) {
            auto equals_pos = definition.find('=');
            switch (equals_pos) {
                case string::npos:
                    spdlog::trace("Preprocessor definition: {}", definition);
                    parsed_options.preprocessor_definitions.emplace(definition);
                    break;
                case 0:
                    die("Preprocessor definition specified with an empty name: \"{}\" ", definition);
                default:
                    auto defined_term = definition.substr(0, equals_pos);
                    auto value = definition.substr(equals_pos + 1);
                    spdlog::trace("Preprocessor definition: {} with value {}", defined_term, value);
                    parsed_options.preprocessor_value_definitions.emplace(defined_term, value);
            }
        }
    }

    if (parse_result.count("argument") > 0) {
        const auto& kernel_arguments_assignments = parse_result["argument"].as<std::vector<string>>();
        for(const auto& kernel_arg_definition : kernel_arguments_assignments) {
            auto equals_pos = kernel_arg_definition.find('=');
            switch(equals_pos) {
                case string::npos:
                    die("Kernel argument name \"{}\" specified without a value", kernel_arg_definition);
                case 0:
                    die("Kernel argument value specified with an empty name: \"{}\" ", kernel_arg_definition);
                default:
                    auto param_name = kernel_arg_definition.substr(0, equals_pos);
                    auto value = kernel_arg_definition.substr(equals_pos+1);
                    parsed_options.kernel_arguments.emplace(param_name, value);
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
execution_context_t initialize_execution_context(parsed_cmdline_options_t parsed_options)
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

    return execution_context;
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
            context.options.compile_in_debug_mode,
            context.options.generate_line_info,
            context.options.language_standard,
            context.finalized_include_dir_paths,
            context.options.preinclude_files,
            context.options.preprocessor_definitions,
            context.options.preprocessor_value_definitions,
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
        auto result = build_opencl_kernel(
            context.opencl.context,
            context.opencl.device,
            context.device_id,
            context.options.kernel.function_name.c_str(),
            kernel_source,
            context.options.compile_in_debug_mode,
            context.options.generate_line_info,
            context.options.write_ptx_to_file,
            context.finalized_include_dir_paths,
            context.options.preinclude_files,
            context.options.preprocessor_definitions,
            context.options.preprocessor_value_definitions,
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

inline void write_data_to_file(
    std::string kind,
    std::string name,
    const_memory_region data,
    filesystem::path destination,
    bool overwrite_allowed,
    spdlog::level::level_enum level)
{
    spdlog::log(level, "Writing {} '{}' to file {}", kind, name, destination.c_str());
    util::write_data_to_file(kind, name, data, destination, overwrite_allowed);
}

void validate_scalars(execution_context_t& context)
{
    spdlog::debug("Validating scalar argument values");
    auto& ka = *context.kernel_adapter_;

    std::stringstream ss;
    const auto& available_args = util::keys(context.scalar_input_arguments.raw);
    ss << available_args;
    spdlog::trace("Available scalar arguments: {}", ss.str()); ss.str("");
    auto required_args = util::transform_if<std::vector<const char *>>(
        ka.scalar_parameter_details(),
        [](const auto& sad) { return sad.required; },
        [](const auto& sad) { return sad.name; });

    ss << required_args;
    spdlog::trace("Required scalar arguments: {}", ss.str()); ss.str("");

    for(const auto& required : required_args) {
        util::contains(available_args, required)
            or die("Required scalar argument {} not provided", required);
    }
}

void validate_arguments(execution_context_t& context)
{
    auto& ka = *context.kernel_adapter_;

    validate_scalars(context);

    spdlog::debug("Validating input sizes");
    ka.input_sizes_are_valid(context) or die("Input buffer sizes are invalid, cannot execute kernel");

    if (not context.kernel_adapter_->extra_validity_checks(context)) {
        // TODO: Have the kernel adapter report an error instead of just a boolean;
        // but we don't want it to know about spdlog, so it should probably
        // return a runtime_error (?)
        die("The combination of input arguments (scalars and buffers) and preprocessor definitions is invalid.");
    }
    spdlog::info("Kernel arguments are fully valid (scalars and buffers, including any inout)");
}

void generate_additional_scalar_arguments(execution_context_t& context)
{
    auto generated_scalars = context.kernel_adapter_->generate_additional_scalar_arguments(context);
    context.scalar_input_arguments.typed.insert(generated_scalars.begin(), generated_scalars.end());
}

void perform_single_run(execution_context_t& context, run_index_t run_index)
{
    spdlog::info("Preparing for kernel run {} of {} (1-based).", run_index+1, context.options.num_runs);
    if (context.options.zero_output_buffers) {
        zero_output_buffers(context);
    }
    reset_working_copy_of_inout_buffers(context);

    if (context.ecosystem == execution_ecosystem_t::cuda) {
        launch_time_and_sync_cuda_kernel(context, run_index);
    }
    else {
        launch_time_and_sync_opencl_kernel(context, run_index);
    }

    spdlog::debug("Kernel execution run complete.");
}

void finalize_kernel_arguments(execution_context_t& context)
{
    spdlog::debug("Marshaling kernel arguments.");
    context.finalized_arguments = context.kernel_adapter_->marshal_kernel_arguments(context);
}

void configure_launch(execution_context_t& context)
{
    spdlog::debug("Creating a launch configuration.");
    auto lc_components = context.kernel_adapter_->make_launch_config(context);
    lc_components.deduce_missing();
    context.kernel_launch_configuration = realize_launch_config(lc_components, context.ecosystem);

    auto gd = lc_components.grid_dimensions.value();
    auto bd = lc_components.block_dimensions.value();
    auto ogd = lc_components.overall_grid_dimensions.value();

    spdlog::info("Launch configuration: Block dimensions:   {:>9} x {:>5} x {:>5} threads", bd[0],bd[1], bd[2]);
    spdlog::info("Launch configuration: Grid dimensions:    {:>9} x {:>5} x {:>5} blocks ", gd[0], gd[1], gd[2]);
    spdlog::info("                                          -----------------------------------");
    spdlog::info("Launch configuration: Overall dimensions: {:>9} x {:>5} x {:>5} threads", ogd[0], ogd[1], ogd[2]);
    if (context.ecosystem == execution_ecosystem_t::cuda) {
        spdlog::info("Launch configuration: Dynamic shared memory:  {} bytes", lc_components.dynamic_shared_memory_size.value_or(0));
    }
    spdlog::debug("Overall dimensions cover full blocks? {}", lc_components.full_blocks());
}

void handle_compilation_log(bool compilation_succeeded, execution_context_t& context)
{
    bool empty_log = context.compilation_log and
        std::all_of(context.compilation_log.value().cbegin(),context.compilation_log.value().cend(),isspace) == true;

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
            spdlog::level::info);
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
        spdlog::level::info);
}

void print_execution_durations(std::ostream& os, const durations_type& execution_durations)
{
    for (const auto &duration: execution_durations) {
        os << duration.count() << '\n';
    }
    os << std::flush;
}

void handle_execution_durations(const execution_context_t &context)
{
    if (not context.options.time_with_events) { return; }
    if (context.options.print_execution_durations) {
        print_execution_durations(std::cout, context.durations);
    }
    if (not context.options.execution_durations_file.empty()) {
        std::ofstream ofs(context.options.execution_durations_file);
        ofs.exceptions();
        print_execution_durations(ofs, context.durations);
    }
}

int main(int argc, char** argv)
{
    spdlog::set_level(spdlog::level::info);
    spdlog::cfg::load_env_levels(); // support setting the logging verbosity with an environment variable

    auto parsed_cmdline_options = parse_command_line(argc, argv);

    execution_context_t context = initialize_execution_context(parsed_cmdline_options);

    if (not context.options.compile_only) {
        parse_scalars(context);
        resolve_buffer_filenames(context);
    }

    ensure_necessary_terms_were_defined(context);
    auto build_succeeded = build_kernel(context);
    auto log = context.compilation_log.value();
    handle_compilation_log(build_succeeded, context);
    build_succeeded or die();

    maybe_write_intermediate_representation(context);

    if (context.options.compile_only) { return EXIT_SUCCESS; }

    read_buffers_from_files(context);
    validate_arguments(context);
    create_host_side_output_buffers(context);
    create_device_side_buffers(context);
    generate_additional_scalar_arguments(context);
    copy_input_buffers_to_device(context);

    finalize_kernel_arguments(context);
    configure_launch(context);

    for(run_index_t ri = 0; ri < context.options.num_runs; ri++) {
        perform_single_run(context, ri);
    }
    handle_execution_durations(context);
    if (context.options.write_output_buffers_to_files) {
        copy_outputs_from_device(context);
        write_buffers_to_files(context);
    }

    spdlog::info("All done.");
}
