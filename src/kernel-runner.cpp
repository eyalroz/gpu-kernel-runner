#include "common_types.hpp"
#include "kernel_inspecific_cmdline_options.hpp"
#include "execution_context.hpp"
#include "kernel_adapter.hpp"
#include "buffer_io.hpp"

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
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/cfg/env.h>

#include <system_error>
#include <cerrno>
#include <iostream>
#include <cstdio>
#include <vector>

using std::size_t;
using std::string;

template <typename... Ts>
[[noreturn]] inline bool die(
    std::string message_format_string = "",
    Ts&&... args)
{
    if(not message_format_string.empty()) {
        spdlog::critical(message_format_string, std::forward<Ts>(args)...);
    }
    exit(EXIT_FAILURE);
}

host_buffers_map read_buffers_from_files(
    const parameter_name_set& buffer_names,
    const string_map& filenames,
    const filesystem::path& buffer_directory)
{
    host_buffers_map result;

    auto determine_and_read =
        [&filenames, &buffer_directory
        //, expected
        ](const parameter_name_set::value_type& buffer_name)
        {
            filesystem::path buffer_file_path = maybe_prepend_base_dir(buffer_directory, filenames.at(buffer_name));
            spdlog::debug("Reading buffer '{}' of size {} bytes from: {}", buffer_name, filesystem::file_size(buffer_file_path), buffer_file_path.c_str());
            host_buffer_type buffer = read_input_file(buffer_file_path);
            return host_buffers_map::value_type{ buffer_name, std::move(buffer)
        };
    };
    std::transform(buffer_names.begin(), buffer_names.end(), std::inserter(result, result.end()), determine_and_read);
    return result;
}

cxxopts::Options basic_cmdline_options(const char* program_name)
{
    cxxopts::Options options { program_name, "A runner for dynamically-compiled CUDA kernels"};
    options.add_options()
        ("l,log-level", "Set logging level", cxxopts::value<string>()->default_value("warning"))
        ("log-flush-threshold", "Set the threshold level at and above which the log is flushed on each message",
            cxxopts::value<string>()->default_value("info"))
        ("w,write-output", "Write output buffers to files", cxxopts::value<bool>()->default_value("true"))
        ("n,num-runs", "Number of times to run the compiled kernel", cxxopts::value<unsigned>()->default_value("1"))
        ("opencl", "Use OpenCL", cxxopts::value<bool>())
        ("cuda", "Use CUDA", cxxopts::value<bool>())
        ("p,platform-id", "Use the OpenCL platform with the specified index", cxxopts::value<unsigned>())
        ("d,device", "Device index", cxxopts::value<int>()->default_value("0"))
        ("D,define", "Set a preprocessor definition for NVRTC (can be used repeatedly; specify either DEFINITION or DEFINITION=VALUE)", cxxopts::value<std::vector<string>>())
        ("c,compile-only", "Compile the kernel, but don't actually run it", cxxopts::value<bool>()->default_value("false"))
        ("G,debug-mode", "Have the NVRTC compile the kernel in debug mode (no optimizations)", cxxopts::value<bool>()->default_value("false"))
        ("P,write-ptx", "Write the intermediate representation code (PTX) resulting from the kernel compilation, to a file", cxxopts::value<bool>()->default_value("false"))
        ("ptx-output-file", "File to which to write the kernel's intermediate representation", cxxopts::value<string>())
        ("print-compilation-log", "Print the compilation log to the standard output", cxxopts::value<bool>()->default_value("false"))
        ("write-compilation-log", "Write the compilation log to a file", cxxopts::value<bool>()->default_value("false"))
        ("compilation-log-file", "Save the compilation log to the specified file (regardless of whether it's printed)", cxxopts::value<string>())
        ("generate-line-info", "Add source line information to the intermediate representation code (PTX)", cxxopts::value<bool>()->default_value("true"))
        ("b,block-dimensions", "Set grid block dimensions in threads  (OpenCL: local work size); a comma-separated list", cxxopts::value<std::vector<unsigned>>() )
        ("g,grid-dimensions", "Set grid dimensions in blocks; a comma-separated list", cxxopts::value<std::vector<unsigned>>() )
        ("o,overall-grid-dimensions", "Set grid dimensions in threads (OpenCL: global work size); a comma-separated list", cxxopts::value<std::vector<unsigned>>() )
        ("S,dynamic-shared-memory-size", "Force specific amount of dynamic shared memory", cxxopts::value<unsigned>() )
        ("W,overwrite", "Overwrite the files for buffer and/or PTX output if they already exists", cxxopts::value<bool>()->default_value("false"))
        ("i,include", "Include a specific file into the kernels' translation unit", cxxopts::value<std::vector<string>>())
        ("I,include-path", "Add a directory to the search paths for header files included by the kernel (can be used repeatedly)", cxxopts::value<std::vector<string>>())
        ("s,kernel-source", "Path to CUDA source file with the kernel function to compile; may be absolute or relative to the sources dir", cxxopts::value<string>())
        ("k,kernel-function", "Name of function within the source file to compile and run as a kernel (if different than the key)", cxxopts::value<string>())
        ("K,kernel-key", "The key identifying the kernel among all registered runnable kernels", cxxopts::value<string>())
        ("L,list-kernels", "List the (keys of the) kernels which may be run with this program")
        ("z,zero-output-buffers", "Set the contents of output(-only) buffers to all-zeros", cxxopts::value<bool>()->default_value("false"))
        ("t,time-execution", "Use CUDA/OpenCL events to time the execution of each run of the kernel", cxxopts::value<bool>()->default_value("false"))
        ("language-standard", "Set the language standard to use for CUDA compilation (options: c++11, c++14, c++17)", cxxopts::value<string>())
        ("input-buffer-dir", "Base location for locating input buffers", cxxopts::value<string>()->default_value( filesystem::current_path().native() ))
        ("output-buffer-dir", "Base location for writing output buffers", cxxopts::value<string>()->default_value( filesystem::current_path().native() ))
        ("kernel-sources-dir", "Base location for locating kernel source files", cxxopts::value<string>()->default_value( filesystem::current_path().native() ))
        ("h,help", "Print usage information")
        ;
    return options;
}

std::vector<const char*> get_required_arg_names(const kernel_adapter& ka) {
    auto sads = ka.scalar_argument_details();
    std::vector<const char*> result;
    util::transform_if(
        std::cbegin(sads), std::cend(sads), std::back_inserter(result),
        [](const auto& sad) { return sad.required; },
        [](const auto& sad) { return sad.name;});
    return result;
}


// TODO: DRY with get_required_scalar_arguments :-(
// Also check if we can't use `util::difference` with strings vs const char*'s after all
std::unordered_set<std::string> get_required_preprocessor_definition_terms(const kernel_adapter& ka) {
    auto pdds = ka.preprocessor_definition_details();
    std::unordered_set<std::string> result;
    util::transform_if(
        std::cbegin(pdds), std::cend(pdds), std::inserter(result, result.begin()),
        [](const auto& sad) { return sad.required; },
        [](const auto& sad) { return std::string{sad.name};});
    return result;
}

void ensure_necessary_terms_were_defined(const execution_context_t& context)
{
    const auto& ka = *context.kernel_adapter_;

    parameter_name_set terms_defined_by_define_options =
        get_defined_terms(context.options.preprocessor_definitions);
    parameter_name_set terms_defined_by_specific_options =
        util::keys(context.options.preprocessor_value_definitions);
    auto all_defined_terms = util::union_(terms_defined_by_define_options, terms_defined_by_specific_options);
    auto terms_required_to_be_defined = get_required_preprocessor_definition_terms(ka);
    auto required_but_undefined = util::difference(terms_required_to_be_defined, all_defined_terms);
    if (not required_but_undefined.empty()) {
        std::ostringstream oss;
        oss << required_but_undefined;
        die("The following preprocessor definitions must be specified, but have not been: {}", oss.str());
    }
}

cxxopts::Options create_command_line_options_for_kernel(const char* program_name, execution_context_t& context)
{
    const auto& ka = *(context.kernel_adapter_.get());
    std::string kernel_name = ka.key();
    spdlog::debug("Creating a command-line options structured for kernel {}", kernel_name);
    cxxopts::Options options = basic_cmdline_options(program_name);
        // We're adding them parse and then ignore; and also possibly for printing usage information

    options.allow_unrecognised_options();
        // This is useful for when you play with removing some of a kernel's parameters or compile-time definitions,
        // so that the same command-line would still work even though it may have some unused options.
        // TODO: Consider reporting the unrecognized options, at least in the log.

    // This splits up the buffers into sections in the options display, each with a "section title"

    static constexpr const auto all_directions = { parameter_direction_t::input, parameter_direction_t::output, parameter_direction_t::inout};
    for(parameter_direction_t dir : all_directions) {
        auto adder = options.add_options(ka.key() + std::string(" (") + parameter_direction_name(dir) + " buffers)");
        kernel_adapter::buffer_details_type dir_buffers =
            util::filter(ka.buffer_details(), [dir](const auto& bd) { return bd.direction == dir; } );
        for(const auto& buffer : dir_buffers ) {
            adder(buffer.name, buffer.description, cxxopts::value<std::string>()->default_value(buffer.name));
        }
    }
    ka.add_scalar_arguments_cmdline_options(options.add_options(ka.key() + std::string(" (scalar arguments)")));
    ka.add_preprocessor_definition_cmdline_options(options.add_options(ka.key() + std::string(" (preprocessor definitions)")));
    return options;
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

void finalize_preprocessor_definitions(execution_context_t& context)
{
    spdlog::debug("Finalizing preprocessor definitions.");
    context.finalized_preprocessor_definitions.valued = context.options.preprocessor_value_definitions;

    for (const auto& definition : context.options.preprocessor_definitions) {
        auto equals_pos = definition.find('=');
        switch(equals_pos) {
        case std::string::npos:
            context.finalized_preprocessor_definitions.valueless.insert(definition);
            continue;
        case 0:
            spdlog::error("Invalid command-line argument \"{}\": Empty defined string",  definition);
            continue;
        default:
            // If the string happens to have "=" at the end, e.g. "FOO=" -
            // it's an empty definition -  which is fine.
            auto term = definition.substr(0, equals_pos);
            auto value = definition.substr(equals_pos+1);
            context.finalized_preprocessor_definitions.valued.emplace(term, value);
        }
    }
    for (const auto& def : context.finalized_preprocessor_definitions.valued) {
        spdlog::trace("finalized value preprocessor definition: {}={}", def.first, def.second);
    }
    for (const auto& def : context.finalized_preprocessor_definitions.valueless) {
        spdlog::trace("finalized valueless preprocessor definition: {}", def);
    }
}

[[noreturn]] void print_help_and_exit(
    const cxxopts::Options &options,
    bool user_asked_for_help = true)
{
    auto &ostream = user_asked_for_help ? std::cout : std::cerr;
    ostream << options.help() << "\n";
    exit(user_asked_for_help ? EXIT_SUCCESS : EXIT_FAILURE);
}

void parse_command_line_for_kernel(int argc, char** argv, execution_context_t& context)
{
    spdlog::debug("Parsing the command line for kernel-specific options.");
    auto program_name = argv[0];
    // if (context.kernel_adapter_.get() == nullptr) { throw std::runtime_error("Null kernel adapter pointer"); }
    const auto& ka = *(context.kernel_adapter_.get());
    auto options = create_command_line_options_for_kernel(program_name, context);

    auto parse_result = non_consumptive_parse(options, argc, argv);
        // Note: parse() will not just change our "local" argc and argv, but will also alter the array of pointers
        // argv points to. If you don't want that - use a

    spdlog::debug("Kernel-inspecific command-line options parsing complete.");

    if (contains(parse_result, "help")) {
        print_help_and_exit(options);
    }

    // TODO: It's possible that the kernel's buffer names will coincide with other option names (especially
    // for the case of single-character names). When this is the case, we should disambiguate. In fact, it might
    // be a good idea to disambiguate to begin with by adding prefixes: input_ output_, inout_, scalar_arg_
    //
    // TODO: Lots of repetition here... avoid it

    for(const auto& buffer_name : buffer_names(ka, parameter_direction_t::input, parameter_direction_t::inout) ) {
        if (contains(parse_result, buffer_name)) {
            context.buffers.filenames.inputs[buffer_name] = parse_result[buffer_name].as<std::string>();
        }
        else {
            context.buffers.filenames.inputs[buffer_name] = buffer_name;
            spdlog::debug("Filename for input buffer {} not specified; defaulting to using its name.", buffer_name);
        }
        spdlog::trace("Filename for input buffer {}: {}", buffer_name, context.buffers.filenames.inputs[buffer_name]);
    }
    if (context.options.write_output_buffers_to_files) {
        for(const auto& buffer_name : ka.buffer_names(parameter_direction_t::output)  ) {
            auto output_filename = [&]() {
                if (contains(parse_result, buffer_name)) {
                    return parse_result[buffer_name].as<std::string>();
                } else {
                    // TODO: Is this a reasonable convention for the output filename?
                    spdlog::debug("Filename for output buffer {0} not specified; defaulting to: \"{0}.out\".", buffer_name);
                    return buffer_name + ".out";
                }
            }();
            if (filesystem::exists(output_filename)) {
                if (not context.options.overwrite_allowed) {
                    throw std::invalid_argument("Writing the contents of output buffer "
                        + buffer_name + " would overwrite an existing file: " + output_filename);
                }
                spdlog::info("Output buffer {} will overwrite {}", buffer_name, output_filename);
            }
            // Note that if the output file gets created while the kernel runs, we might miss this fact
            // when trying to write to it.
            spdlog::trace("Filename for output buffer {}: {}", buffer_name, output_filename);
            context.buffers.filenames.outputs[buffer_name] = output_filename;
        }
        for(const auto& buffer_name : ka.buffer_names(parameter_direction_t::inout)  ) {
            // TODO: Consider support other schemes for naming output versions of inout buffers
            context.buffers.filenames.outputs[buffer_name] =
                context.options.buffer_base_paths.output / (buffer_name + ".out");
            spdlog::trace("Using output file {} for buffer {}", context.buffers.filenames.outputs[buffer_name], buffer_name);
        }
    }
    auto required_scalar_names = get_required_arg_names(ka);
    for(const auto& arg_name : required_scalar_names ) {
        contains(parse_result, arg_name)
            or die("Scalar argument '{}' must be specified, but wasn't.\n\n", arg_name);
        // TODO: Consider not parsing anything at this stage, and just marshaling all the scalar arguments together.
        spdlog::trace("Parsing scalar argument {}", arg_name);
        auto& arg_value = parse_result[arg_name].as<std::string>();
        context.scalar_input_arguments.raw[arg_name] = arg_value;
        context.scalar_input_arguments.typed[arg_name] =
             ka.parse_cmdline_scalar_argument(arg_name, arg_value);
        spdlog::trace("Successfully parsed scalar argument {}", arg_name);
    }

    auto required_defs = get_required_preprocessor_definition_terms(ka);
    for(const auto& def_name : required_defs ) {
        if (not contains(parse_result, def_name)) {
            // we'll check this later; maybe it was otherwise specified
            spdlog::trace("Preprocessor term {} not passed using a specific option; "
                "hopefully it has been manually-defined.");
            continue;
        }
        // TODO: Consider not parsing anything at this stage, and just marshaling all the scalar arguments together.
        // context.scalar_input_arguments.raw[def_name] = parse_result[def_name].as<std::string>();
        const auto& arg_value = parse_result[def_name].as<std::string>();
        context.options.preprocessor_value_definitions[def_name] = arg_value;
        spdlog::trace("Got preprocessor argument {}={} through specific option", def_name, arg_value);
    }

    ensure_necessary_terms_were_defined(context);

    finalize_preprocessor_definitions(context);
}

void ensure_gpu_device_validity(
    execution_ecosystem_t ecosystem,
    optional<unsigned> platform_id,
    int device_id,
    bool need_ptx)
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

kernel_inspecific_cmdline_options_t parse_command_line_initially(int argc, char** argv)
{
    auto program_name = argv[0];
    cxxopts::Options options = basic_cmdline_options(program_name);
    options.allow_unrecognised_options();

    // Note that the following will be printed based only on the compiled-in
    // default log level
    spdlog::debug("Parsing the command line for non-kernel-specific options.");
    auto parse_result = non_consumptive_parse(options, argc, argv);

    kernel_inspecific_cmdline_options_t parsed_options;

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

    auto log_level_name = parse_result["log-level"].as<std::string>();
    auto log_level = spdlog::level::from_str(log_level_name);
    if (spdlog::level_is_at_least(spdlog::level::debug)) {
        spdlog::log(spdlog::level::debug, "Setting log level to {}", log_level_name);
    }
    spdlog::set_level(log_level);

    auto log_flush_threshold_name = parse_result["log-flush-threshold"].as<std::string>();
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

    std::string source_file_path;

    if (got.source_file_path) {
        source_file_path = parse_result["kernel-source"].as<std::string>();
    }

    if (got.function_name) {
        parsed_options.kernel.function_name = parse_result["kernel-function"].as<std::string>();
        if (not util::is_valid_identifier(parsed_options.kernel.function_name)) {
            throw std::invalid_argument("Function name must be non-empty.");
        }
    }
    if (got.key) {
        parsed_options.kernel.key = parse_result["kernel-key"].as<std::string>();
        if (parsed_options.kernel.key.empty()) {
            throw std::invalid_argument("Kernel key may not be empty.");
        }
    }

    std::string clipped_key = [&]() {
        if (got.key) {
            auto pos_of_last_invalid = parsed_options.kernel.key.find_last_of("/-;.[]{}(),");
            return parsed_options.kernel.key.substr(
                (pos_of_last_invalid == std::string::npos ? 0 : pos_of_last_invalid + 1), std::string::npos);
        }
        return std::string{};
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
            std::string("No source file specified, and inferred source file path does not exist") +
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
    parsed_options.always_print_compilation_log = parse_result["print-compilation-log"].as<bool>();
    parsed_options.write_compilation_log = parse_result["write-compilation-log"].as<bool>();
    if (parsed_options.write_compilation_log) {
        // Note: DRY with PTX file
        if (contains(parse_result, "compilation-log-file")) {
            parsed_options.compilation_log_file = parse_result["compilation-log-file"].as<string>();
            if (filesystem::exists(parsed_options.compilation_log_file)) {
                if (not parsed_options.overwrite_allowed) {
                    throw std::invalid_argument("Specified compilation log file "
                        + parsed_options.compilation_log_file.native() + " exists, and overwrite is not allowed.");
                }
                // Note that there could theoretically be a race condition in which the file gets created
                // between our checking for its existence and our wanting to write to it after compilation.
            }
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
    parsed_options.compile_in_debug_mode = parse_result["debug-mode"].as<bool>();
    parsed_options.zero_output_buffers = parse_result["zero-output-buffers"].as<bool>();
    parsed_options.time_with_events = parse_result["time-execution"].as<bool>();

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

//    parsed_options.compare_outputs_against_expected = parse_results["compare-outputs"].as<std::string>();

    if (parse_result.count("define") > 0) {
        const auto& parsed_defines = parse_result["define"].as<std::vector<string>>();
        std::copy(parsed_defines.cbegin(), parsed_defines.cend(),
            std::inserter(parsed_options.preprocessor_definitions, parsed_options.preprocessor_definitions.begin()));
            // TODO: This line is kind of brittle, because the RHS type is dictated
            // by cxxopts while the LHS is a choice of ours which is supposedly
            // independent of it. Perhaps we should initialize parsed_options.include_paths
            // with a begin/end constructor pair?
        for (const auto& def : parsed_options.preprocessor_definitions) {
            spdlog::trace("Preprocessor definition: {}", def);
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

    if (not kernel_adapter::can_produce_subclass(std::string(parsed_options.kernel.key))) {
        die("No kernel adapter is registered for key {}", parsed_options.kernel.key);
    }

    return parsed_options;
}

/**
 * Writes output buffers, generated by the kernel, to the files specified at the
 * command-line - one file per buffer.
 */
void write_buffers_to_files(execution_context_t& context)
{
    spdlog::info("Writing output buffers to files.");
    // Unfortunately, decent ranged-for iteration on maps is only possible with C++17
    for(const auto& pair : context.buffers.host_side.outputs) {
        const auto& buffer_name = pair.first;
        const auto& buffer = pair.second;
        auto write_destination = maybe_prepend_base_dir(
               context.options.buffer_base_paths.output,
               context.buffers.filenames.outputs[buffer_name]);
        write_buffer_to_file(buffer_name, buffer, write_destination, context.options.overwrite_allowed);
    }
}

// TODO: Yes, make execution_context_t a proper class... and be less lax with the initialization
execution_context_t initialize_execution_context(kernel_inspecific_cmdline_options_t parsed_options)
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
        kernel_adapter::produce_subclass(std::string(parsed_options.kernel.key));

    collect_include_paths(execution_context);

    return execution_context;
}

void copy_buffer_to_device(
    const execution_context_t& context,
    const string&              buffer_name,
    const device_buffer_type&  device_side_buffer,
    const host_buffer_type&    host_side_buffer)
{
    if (context.ecosystem == execution_ecosystem_t::cuda) {
        spdlog::debug("Copying buffer {} (size {} bytes) from host-side copy at {} to device side copy at {}",
            buffer_name, host_side_buffer.size(), (void *) host_side_buffer.data(),
            (void *) device_side_buffer.cuda.data());
        cuda::context::current::scoped_override_t scoped_context_override{ *context.cuda.context };
        cuda::memory::copy(device_side_buffer.cuda.data(), host_side_buffer.data(), host_side_buffer.size());
    } else { // OpenCL
        const constexpr auto blocking { CL_TRUE };
        context.opencl.queue.enqueueWriteBuffer(device_side_buffer.opencl, blocking, 0, host_side_buffer.size(),
            host_side_buffer.data());
    }
}

void copy_buffer_on_device(
    execution_ecosystem_t ecosystem,
    cl::CommandQueue* queue,
    const device_buffer_type&  destination,
    const device_buffer_type&  origin)
{
    if (ecosystem == execution_ecosystem_t::cuda) {
        cuda::memory::copy(destination.cuda.data(), origin.cuda.data(), destination.cuda.size());
    } else { // OpenCL
        size_t size;
        origin.opencl.getInfo(CL_MEM_SIZE, &size);
        queue->enqueueCopyBuffer(origin.opencl, destination.opencl, 0, 0, size);
    }
}


void copy_input_buffers_to_device(const execution_context_t& context)
{
    spdlog::debug("Copying inputs to device.");
    for(const auto& input_pair : context.buffers.host_side.inputs) {
        const auto& name = input_pair.first;
        const auto& host_side_buffer = input_pair.second;
        const auto& device_side_buffer = context.buffers.device_side.inputs.at(name);
        copy_buffer_to_device(context, name, device_side_buffer, host_side_buffer);

    }

    spdlog::debug("Copying in-out buffers to a 'pristine' copy on the device (which will not be altered).");
    for(const auto& buffer_name : context.kernel_adapter_->buffer_names(parameter_direction_t::inout)  ) {
        auto& host_side_buffer = context.buffers.host_side.inputs.at(buffer_name);
        const auto& device_side_buffer = context.buffers.device_side.inputs.at(buffer_name);
        copy_buffer_to_device(context, buffer_name, device_side_buffer, host_side_buffer);
    }
}

void copy_buffer_to_host(
    execution_ecosystem_t ecosystem,
    //const execution_context_t& context,
    cl::CommandQueue*          opencl_queue,
    const device_buffer_type&  device_side_buffer,
    host_buffer_type&          host_side_buffer)
{
    if (ecosystem == execution_ecosystem_t::cuda) {
        cuda::memory::copy(host_side_buffer.data(), device_side_buffer.cuda.data(), host_side_buffer.size());
    } else {
        // OpenCL
        const constexpr auto blocking { CL_TRUE };
        constexpr const auto no_offset { 0 };
        opencl_queue->enqueueReadBuffer(device_side_buffer.opencl, blocking, no_offset, host_side_buffer.size(), host_side_buffer.data());
    }
}

// Note: must take the context as non-const, since it has vector members, and vectors
// are value-types, not reference-types, i.e. copying into those vectors changes
// the context.
void copy_outputs_from_device(execution_context_t& context)
{
    spdlog::debug("Copying outputs back to host memory.");
    for(auto& output_pair : context.buffers.host_side.outputs) {
        const auto& name = output_pair.first;
        auto& host_side_buffer = output_pair.second;
        const auto& device_side_buffer = context.buffers.device_side.outputs.at(name);
        spdlog::trace("Copying device output buffer to host output buffer for {}", name);
        copy_buffer_to_host(
            context.ecosystem,
            &context.opencl.queue,
            device_side_buffer,
            host_side_buffer);
    }
    context.cuda.context->synchronize();
}

device_buffer_type create_device_side_buffer(
    const std::string& name,
    std::size_t size,
    execution_ecosystem_t ecosystem,
    const optional<cuda::context_t>& cuda_context,
    optional<cl::Context> opencl_context,
    const host_buffers_map&)
{
    device_buffer_type result;
    if (ecosystem == execution_ecosystem_t::cuda) {
        auto region = cuda::memory::device::allocate(*cuda_context, size);
        poor_mans_span sp { static_cast<byte_type*>(region.data()), region.size() };
        spdlog::trace("Created buffer at address {} with size {} for kernel parameter {}", (void*) sp.data(), sp.size(), name);
        result.cuda = sp;
    }
    else { // OpenCL
        cl::Buffer buffer { opencl_context.value(), CL_MEM_READ_WRITE, size };
            // TODO: Consider separating in, out and in/out buffer w.r.t. OpenCL creating, to be able to pass
            // other flags.
        spdlog::trace("Created an OpenCL read/write buffer with size {} for kernel parameter {}", size, name);
        result.opencl = std::move(buffer);
    }
    return result;
}

device_buffers_map create_device_side_buffers(
    execution_ecosystem_t            ecosystem,
    const optional<cuda::context_t>& cuda_context,
    optional<cl::Context>            opencl_context,
    const host_buffers_map&          host_side_buffers)
{
    device_buffers_map result;
    // TODO: Use map() from functional

    std::transform(
        host_side_buffers.cbegin(),
        host_side_buffers.cend(),
        std::inserter(result, result.end()),
        [&](const auto& p) {
            const auto& name = p.first;
            const auto& size = p.second.size();
            spdlog::debug("Creating device buffer of size {} for kernel parameter {}.", size, name);
            auto buffer = create_device_side_buffer(
                name, size,
                ecosystem,
                cuda_context,
                opencl_context,
                host_side_buffers);
            return device_buffers_map::value_type { name, std::move(buffer) };
        } );
    return result;
}

void zero_output_buffer(
    execution_ecosystem_t     ecosystem,
    const device_buffer_type  buffer,
    optional<cuda::stream_t>  cuda_stream,
    const cl::CommandQueue*   opencl_queue,
    const std::string &       buffer_name)
{
    spdlog::trace("Zeroing output buffer '{}'", buffer_name);
    if (ecosystem == execution_ecosystem_t::cuda) {
        cuda_stream->enqueue.memzero(buffer.cuda.data(), buffer.cuda.size());
    } else {
        // OpenCL
        const constexpr unsigned char zero_pattern { 0 };
        const constexpr size_t no_offset { 0 };
        size_t size;
        buffer.opencl.getInfo(CL_MEM_SIZE, &size);
        opencl_queue->enqueueFillBuffer(buffer.opencl, zero_pattern, no_offset, size);
    }
}

void zero_output_buffers(execution_context_t& context)
{
    const auto& ka = *context.kernel_adapter_;
    auto output_only_buffers = ka.buffer_names(parameter_direction_t::out);
    if (output_only_buffers.empty()) {
        spdlog::debug("There are no output-only buffers to fill with zeros.");
        return;
    }
    spdlog::debug("Zeroing output-only buffers.");
    for(const auto& buffer_name : output_only_buffers) {
        const auto& buffer = context.buffers.device_side.outputs.at(buffer_name);
        zero_output_buffer(context.ecosystem, buffer, context.cuda.stream, &context.opencl.queue, buffer_name);
    }
    spdlog::debug("Output-only buffers filled with zeros.");
}

void create_device_side_buffers(execution_context_t& context)
{
    spdlog::debug("Creating device buffers.");
    context.buffers.device_side.inputs = create_device_side_buffers(
        context.ecosystem,
        context.cuda.context,
        context.opencl.context,
        context.buffers.host_side.inputs);
    spdlog::debug("Input device buffers created.");
    context.buffers.device_side.outputs = create_device_side_buffers(
        context.ecosystem,
        context.cuda.context,
        context.opencl.context,
        context.buffers.host_side.outputs);
            // ... and remember the behavior regarding in-out buffers: For each in-out buffers, a buffer
            // is crea0ted in _both_ previous function calls
    spdlog::debug("Output device buffers created.");
}

// Note: Will create buffers also for each inout buffers
void create_host_side_output_buffers(execution_context_t& context)
{
    spdlog::debug("Creating host-side output buffers");
    auto output_buffer_sizes = context.kernel_adapter_->output_buffer_sizes(
        context.buffers.host_side.inputs,
        context.scalar_input_arguments.typed,
        context.finalized_preprocessor_definitions.valueless,
        context.finalized_preprocessor_definitions.valued);

    // TODO: Double-check that all output and inout buffers have entries in the map we've received.

    std::transform(
        output_buffer_sizes.begin(),
        output_buffer_sizes.end(),
        std::inserter(context.buffers.host_side.outputs, context.buffers.host_side.outputs.end()),
        [](const auto& pair) {
            const auto& name = pair.first;
            const auto& size = pair.second;
            return std::make_pair(name, host_buffer_type(size));
        }
    );
}

void read_buffers_from_files(execution_context_t& context)
{
    spdlog::debug("Reading input buffers.");
    auto buffer_names_to_read_from_files = buffer_names(
        *context.kernel_adapter_,
        parameter_direction_t::input,
        parameter_direction_t::inout);
    context.buffers.host_side.inputs =
        read_buffers_from_files(
            buffer_names_to_read_from_files,
            context.buffers.filenames.inputs,
            context.options.buffer_base_paths.input);
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

    if (context.options.write_compilation_log and
        context.options.compilation_log_file.empty())
    {
        context.options.compilation_log_file =
            context.options.kernel.function_name + ".log";
    }
}

bool build_kernel(execution_context_t& context)
{
    finalize_kernel_function_name(context);
    const auto& source_file = context.options.kernel.source_file;
    spdlog::debug("Reading the kernel from {}", source_file.native());
    auto kernel_source_buffer = read_file_as_null_terminated_string(source_file);
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
            context.finalized_preprocessor_definitions.valueless,
            context.finalized_preprocessor_definitions.valued);
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
            context.finalized_preprocessor_definitions.valueless,
            context.finalized_preprocessor_definitions.valued);
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

// Note: We could actually do some verification
// before building the kernel and before reading
// from any file - although just for the scalars.
void verify_input_arguments(execution_context_t& context)
{
    spdlog::debug("Verifying input arguments (buffers and scalars)");
    auto& ka = *context.kernel_adapter_;

    auto in_and_inout_names = buffer_names(ka, parameter_direction_t::input, parameter_direction_t::inout);
    auto obtained_in_buffers = util::keys(context.buffers.host_side.inputs);
    if (obtained_in_buffers != in_and_inout_names)
    {
        std::ostringstream ss;
        auto names_of_missing_buffers = util::difference(in_and_inout_names, obtained_in_buffers);
        for (auto buffer_name : names_of_missing_buffers) { ss << buffer_name << " "; }
        die("Missing input/inout buffers: {}", ss.str());
    }

    std::stringstream ss;
    const auto& available_args = util::keys(context.scalar_input_arguments.raw);
    ss << available_args;
    spdlog::trace("Available scalar arguments: {}", ss.str()); ss.str("");
    const auto required_args = get_required_arg_names(ka);
    ss << required_args;
    spdlog::trace("Required scalar arguments: {}", ss.str()); ss.str("");

    for(const auto& required : required_args) {
        util::contains(available_args, required)
            or die("Required scalar argument {} not provided", required);
    }

    ka.input_sizes_are_valid(context) or die("Inputs are invalid, cannot execute kernel");

    if (not context.kernel_adapter_->extra_validity_checks(context)) {
        // TODO: Have the kernel adapter report an error instead of just a boolean;
        // but we don't want it to know about spdlog, so it should probably
        // return a runtime_error (?)
        die("The combination of input arguments (scalars and buffers) and preprocessor definitions is invalid.");
    }
    spdlog::info("Verified all inputs (scalars and/or buffers, including any inout)");
}

void generate_additional_scalar_arguments(execution_context_t& context)
{
    auto generated_scalars = context.kernel_adapter_->generate_additional_scalar_arguments(context);
    context.scalar_input_arguments.typed.insert(generated_scalars.begin(), generated_scalars.end());
}

void reset_working_copy_of_inout_buffers(execution_context_t& context)
{
    auto& ka = *context.kernel_adapter_;
    auto inout_buffer_names = ka.buffer_names(parameter_direction_t::inout);
    if (inout_buffer_names.empty()) {
        return;
    }
    spdlog::debug("Initializing the 'work-copies' of the in-out buffers with the contents of the read-only device-side copies.");
    for(const auto& inout_buffer_name : inout_buffer_names) {
        const auto& pristine_copy = context.buffers.device_side.inputs.at(inout_buffer_name);
        const auto& work_copy = context.buffers.device_side.outputs.at(inout_buffer_name);
        spdlog::debug("Initializing {}...", inout_buffer_name);
        copy_buffer_on_device(context.ecosystem,
            context.ecosystem == execution_ecosystem_t::opencl ? &context.opencl.queue : nullptr,
            work_copy, pristine_copy);

    }
    context.cuda.context->synchronize();
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

void maybe_print_and_write_log(bool compilation_succeeded, execution_context_t& context)
{
    bool empty_log = context.compilation_log and
        std::all_of(context.compilation_log.value().cbegin(),context.compilation_log.value().cend(),isspace) == true;

    //        auto end = compilation_log.end() - ((not compilation_log.empty() and compilation_log.back() == '\0') ? 1 : 0);
//        bool print_the_log = compilation_succeeded or

    spdlog::level::level_enum level = compilation_succeeded ? spdlog::level::debug : spdlog::level::err;

    if (context.options.always_print_compilation_log or not compilation_succeeded) {
        if (not context.compilation_log or empty_log) {
            spdlog::log(level, "No compilation log produced.");
        }
        else {
            spdlog::log(level, "Kernel compilation log:");
            std::cout << context.compilation_log.value();
            if (context.compilation_log.value().end()[-1] != '\n') {
                std::cout << '\n';
            }
        }
    }
    if (context.options.write_compilation_log and context.compilation_log) {
        auto log { context.compilation_log.value() };
        write_data_to_file(
            "compilation log for", context.options.kernel.key,
            // TODO: Get rid of this, use a proper span and const span...
            poor_mans_span{ const_cast<byte_type*>(log.data()), log.size() },
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
        poor_mans_span{ const_cast<byte_type *>(ptx.data()), ptx.length() },
        context.options.ptx_output_file,
        context.options.overwrite_allowed,
        spdlog::level::info);
}

int main(int argc, char** argv)
{
    spdlog::set_level(spdlog::level::info);
    spdlog::cfg::load_env_levels(); // support setting the logging verbosity with an environment variable

    auto kernel_inspecific_cmdline_options = parse_command_line_initially(argc, argv);

    execution_context_t context = initialize_execution_context(kernel_inspecific_cmdline_options);
    parse_command_line_for_kernel(argc, argv, context);

    auto build_succeeded = build_kernel(context);
    auto log = context.compilation_log.value();
    maybe_print_and_write_log(build_succeeded, context);
    build_succeeded or die();

    maybe_write_intermediate_representation(context);

    if (context.options.compile_only) { return EXIT_SUCCESS; }

    read_buffers_from_files(context);
    // TODO: Consider verifying before reading the buffers, but obtaining the sizes
    // for the verification
    verify_input_arguments(context);
    create_host_side_output_buffers(context);
    create_device_side_buffers(context);
    generate_additional_scalar_arguments(context);
    copy_input_buffers_to_device(context);

    finalize_kernel_arguments(context);
    configure_launch(context);

    for(run_index_t ri = 0; ri < context.options.num_runs; ri++) {
        perform_single_run(context, ri);
    }
    if (context.options.write_output_buffers_to_files) {
        copy_outputs_from_device(context);
        write_buffers_to_files(context);
    }

    spdlog::info("All done.");
}
