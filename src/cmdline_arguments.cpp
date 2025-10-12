#include "common_types.hpp"
#include "cmdline_arguments.hpp"

#include "util/miscellany.hpp"
#include "util/cxxopts-extra.hpp"
#include "util/spdlog-extra.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/cfg/helpers.h>

#include <vector>

using std::size_t;
using std::string;

[[noreturn]] void print_help_and_exit(
    const cxxopts::Options &options,
    bool user_asked_for_help = true);
void print_registered_kernel_keys();
void list_opencl_platforms();
void set_log_level(std::string log_level_name);
void verify_key_validity(const std::string& key);

cxxopts::Options basic_cmdline_options(const char* program_name)
{
    cxxopts::Options options { program_name, "A runner for dynamically-compiled CUDA kernels"};
    options.add_options()
        ("l,log-level,log", "Set logging level", cxxopts::value<string>()->default_value("warning"))
        ("log-flush-threshold", "Set the threshold level at and above which the log is flushed on each message", cxxopts::value<string>()->default_value("info"))
        ("w,write,write-output,write-outputs,save-output,save-outputs", "Write output buffers to files", cxxopts::value<bool>()->default_value("true"))
        ("n,num-runs,runs,repetitions", "Number of times to run the compiled kernel", cxxopts::value<unsigned>()->default_value("1"))
        ("e,ecosystem,execution-ecosystem", "Execution ecosystem (CUDA or Opencl)", cxxopts::value<string>())
        ("opencl,OpenCL", "Use OpenCL", cxxopts::value<bool>())
        ("cuda,CUDA", "Use CUDA", cxxopts::value<bool>())
        ("p,platform,platform-id", "Use the OpenCL platform with the specified index", cxxopts::value<unsigned>())
        ("list-opencl-platforms,list-platforms,platforms", "List available OpenCL platforms")
        ("a,arg,argument", "Set one of the kernel's argument, keyed by name, with a serialized value for a scalar (e.g. foo=123) or a path to the contents of a buffer (e.g. bar=/path/to/data.bin)", cxxopts::value<std::vector<string>>())
        ("A,no-implicit-compilation-options,no-implicit-compile-options,suppress-default-compile-options,suppress-default-compilation-options,no-default-compile-options,no-default-compilation-options", "Avoid setting any compilation options not explicitly requested by the user", cxxopts::value<bool>()->default_value("false"))
        ("buffer-size,output-buffer-size,output-size,arg-size", "Set one of the buffers' sizes, keyed by name, in bytes (e.g. myresult=1024)", cxxopts::value<std::vector<string>>())
        ("d,dev,device,device-index", "Device index", cxxopts::value<int>()->default_value("0"))
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
        ("accept-oversized-inputs,accept-oversized,allow-oversized-inputs,allow-oversized,lax-size-validation", "Accept input buffers exceeding the expected size calculated for them", cxxopts::value<bool>()->default_value("false"))
        ("kernel-sources-dir,kernel-source-dir,kernel-source-directory,kernel-sources-directory,kernel-sources,source-dir,sources", "Base location for locating kernel source files", cxxopts::value<string>()->default_value( filesystem::current_path().native() ))
        ("h,help", "Print usage information")
        ;
    return options;
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

    parsed_options.valid = true;
    if (contains(parse_result, "help")) {
        parsed_options.early_exit_action = early_exit_action_t::print_help;
        // Note that we will not immediately provide the help, because if we can figure
        // out what kernel was asked for, we will want to provide help regarding
        // the kernel-specific command-line arguments
        parsed_options.help_text = options.help();
        return parsed_options;
    }
    else if (contains(parse_result, "list-kernels")) {
        parsed_options.early_exit_action = early_exit_action_t::list_kernels;
        return parsed_options;
    }
    else if (contains(parse_result, "list-opencl-platforms")) {
        parsed_options.early_exit_action = early_exit_action_t::list_opencl_platforms;
        return parsed_options;
    }

    struct { bool key, function_name, source_file_path; } got {
        contains(parse_result, "kernel-key"),
        contains(parse_result, "kernel-function"),
        contains(parse_result, "kernel-source")
    };

    (got.key or got.function_name or got.source_file_path) or die(
        "No kernel key was specified, nor enough information provided "
        "to deduce the kernel key, source filename and kernel function name");

    // No need to exit (at least not until second parsing), let's
    // go ahead and collect the parsed data

    auto log_level_name = parse_result["log-level"].as<string>();
    set_log_level(log_level_name);

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
    parsed_options.accept_oversized_inputs = parse_result["accept-oversized-inputs"].as<bool>();

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

    verify_key_validity(parsed_options.kernel.key);

    parsed_options.time_with_events =
        parsed_options.print_execution_durations or (not parsed_options.execution_durations_file.empty())
        or spdlog::level_is_at_least(spdlog::level::info);

    return parsed_options;
}

