#ifndef KERNEL_ADAPTER_HPP_
#define KERNEL_ADAPTER_HPP_

#include "execution_context.hpp"
#include "parsers.hpp"

#include "util/miscellany.hpp"
#include "util/functional.hpp"
#include "util/factory_producible.hpp"
#include "util/optional_and_any.hpp"
#include "util/static_block.hpp"
    // This file itself does not use static blocks, but individual kernel adapters may
    // want to use them for registering themselves in the factory.

#include <cxx-prettyprint/prettyprint.hpp>

#include <common_types.hpp>

// A convenience overload for specific kernel adapters to be able
// to complain about dimensions_t's they get.
std::ostream& operator<<(std::ostream& os, cuda::grid::dimensions_t dims);

// Note: The returned size is in bytes, not elements
using size_calculator_type = std::size_t (*)(
    const host_buffers_t& input_buffers,
    const scalar_arguments_map& scalar_arguments,
    const preprocessor_definitions_t& valueless_preprocessor_definitions,
    const valued_preprocessor_definitions_t& value_preprocessor_definitions,
    const optional_launch_config_components_t& forced_lc_components);

static constexpr const size_calculator_type no_size_calc = nullptr;

using scalar_pusher_type = void (*)(
    marshalled_arguments_type& argument_ptrs_and_maybe_sizes,
    const execution_context_t& context,
    const char* buffer_argument_name);

static constexpr const scalar_pusher_type no_pusher = nullptr;


namespace kernel_adapters {

using key = std::string;
    // I have also considered, but rejected for now, the option of
    // struct key { string variant; string name; };

} //namespace kernel_adapters

/**
 * This class (or rather, its concrete subclasses) encapsulates
 * all the apriori information and logic specifically regarding a single
 * kernel - and hopefully nothing else. The rest of the kernel runner's
 * code knows nothing about any specific kernel, and uses the methods here
 * to obtain this information, uniformly for all kernels. Any logic
 * _not_ dependent on the kernel should not be in this class - not in
 * the child classes, but not even in this abstract base class.
 *
 * @note This class' methods...
 *
 *   - Do not allocate, de-allocate or own any large buffers
 *   - Do not perform any significant computation
 *   - Do not trigger memory copy to/from CUDA devices, nor kernel execution
 *     on CUDA devices, etc.
 *   - May make CUDA API calls to determine information about CUDA devices.
 */
class kernel_adapter : util::mixins::factory_producible<kernel_adapters::key, kernel_adapter> {
public:
    using key_type = kernel_adapters::key;
    using mixin_type = util::mixins::factory_producible<key_type, kernel_adapter>;
    using mixin_type::can_produce_subclass;
    using mixin_type::produce_subclass;
    using mixin_type::get_subclass_factory;
    using mixin_type::register_in_factory;

protected:
    // This will make it easier for subclasses to implement the parameter_details function
    static constexpr const auto input  = parameter_direction_t::input;
    static constexpr const auto output = parameter_direction_t::output;
    static constexpr const auto buffer = kernel_parameters::kind_t::buffer;
    static constexpr const auto scalar = kernel_parameters::kind_t::scalar;
    static constexpr const auto inout  = parameter_direction_t::inout;
    static constexpr const auto scratch  = parameter_direction_t::scratch;
    static constexpr const bool is_required = kernel_parameters::is_required;
    static constexpr const bool isnt_required = kernel_parameters::isnt_required;
    static constexpr const bool is_not_required = kernel_parameters::isnt_required;

public: // constructors & destructor
    kernel_adapter() = default;
    kernel_adapter(const kernel_adapter&) = default;
    virtual ~kernel_adapter() = default;
    kernel_adapter& operator=(kernel_adapter&) = default;
    kernel_adapter& operator=(kernel_adapter&&) = default;

    struct single_parameter_details {
        const char* name;
        std::vector<std::string> aliases_;
        kernel_parameters::kind_t kind;
        parser_type parser;
        size_calculator_type size_calculator;
        scalar_pusher_type pusher;
        parameter_direction_t direction; // always input for scalars
        bool required;

        const std::vector<std::string>& get_aliases() const noexcept { return aliases_; }
        single_parameter_details aliases(std::initializer_list<const char*> extra_aliases) const;
        single_parameter_details alias(const char* extra_alias) const;
        single_parameter_details alias(const std::string& extra_alias) const;
        bool has_alias(const std::string& alias) const;
    };

    struct single_preprocessor_definition_details {
        const char* name;
        bool required;
        const char *default_value;
    };

    // TODO: This should really be a span (and then we wouldn't
    // need to use const-refs to it)
    using parameter_details_type = std::vector<single_parameter_details>;
    using preprocessor_definition_details_type = std::vector<single_preprocessor_definition_details>;

public:
    /**
     * @brief The key for each adapter has multiple uses: It's used to look it up dynamically
     * and create an instance of it; it's used as a default path suffix for the kernel file;
     * it's used to identify which kernel is being run, to the user; it may be used for output
     * file generation; etc.
     */
    virtual std::string key() const = 0;

    // Q: Why is the value of this function not the same as the key?
    // A: Because multiple variants of the same kernel may use the same kernel function name,
    // e.g. in CUDA and in OpenCL, with different kinds of optimizations etc.
    virtual std::string kernel_function_name() const = 0;

    // Note: Inheriting classes must define a key_type key_ static member -
    // or else they cannot be registered in the factory.

    virtual const parameter_details_type & parameter_details() const = 0;
    virtual parameter_details_type scalar_parameter_details() const;
    virtual parameter_details_type buffer_details() const;
    // TODO: Could use an optional-ref return type here
    /**
     * @brief Obtains the set of preprocessor definitions which may affect the kernel's  compilation,
     * including an indication of which of them _must_ be specified for compilation to succeed.
     * @return  Either the set of affecting preprocessor definition terms, or the empty optional value in
     * case that set is not known to the adapter
     */
    virtual preprocessor_definition_details_type preprocessor_definition_details() const { return {}; }

protected:
    template <typename Scalar>
    static void pusher(
        marshalled_arguments_type& argument_ptrs_and_maybe_sizes,
        const execution_context_t& context,
        const char* scalar_parameter_name);

public:
    virtual any parse_cmdline_scalar_argument(
        const single_parameter_details& parameter_details,
        const std::string& value_str) const
    {
        return parameter_details.parser(value_str);
    }

    virtual void generate_additional_preprocessor_definitions(execution_context_t&) const { }
    virtual scalar_arguments_map generate_additional_scalar_arguments(execution_context_t&) const { return {}; }
    virtual bool extra_validity_checks(const execution_context_t&) const { return true; }
    virtual optional<std::string> default_language_standard(execution_ecosystem_t) const { return {}; }

public:

    /**
     * Marshals an array of pointers which can be used for a CUDA/OpenCL-driver-runnable kernel's
     * arguments.
     *
     * @param context A fully-populated test execution context, containing all relevant buffers
     * and scalar arguments.
     * @return the marshaled array of pointers, which may be passed to cuLaunchKernel or
     * clEnqueueNDRangeKernel. For CUDA, it is nullptr-terminated; for OpenCL, we also fill
     * an array of argument sizes.
     *
     * @note This method will get invoked after we've already used the preprocessor
     * definitions to compile the kernels. It may therefore assume they are all present and valid
     * (well, valid enough to compile).
     *
     * @TODO I think we can probably arrange it so that the specific adapters only need to specify
     * the sequence of names, and this function can take care of all the rest - seeing how
     * launching gets the arguments in a type-erased fashion.
     *
     */
    marshalled_arguments_type marshal_kernel_arguments(const execution_context_t& context) const;
    virtual optional_launch_config_components_t deduce_launch_config(const execution_context_t& context) const;
    optional_launch_config_components_t make_launch_config(const execution_context_t& context) const;
    static std::initializer_list<std::string> no_aliases() { return {}; }

protected:
    // Convenience functions for construction parameter details within an
    // implementation of the @ref parameter_details method

    template <typename T>
    static single_parameter_details scalar_details(
        const char*                        name,
        bool                               required = is_required,
        std::initializer_list<std::string> name_aliases = no_aliases())
    {
        return single_parameter_details {name, name_aliases, scalar, parser<T>, no_size_calc, pusher<T>, input, required};
    }

    static single_parameter_details buffer_details(
        const char*            name,
        parameter_direction_t  direction,
        size_calculator_type   size_calculator = no_size_calc,
        bool                   required = is_required,
        std::initializer_list<std::string> name_aliases = no_aliases());
}; // kernel_adapter

name_set buffer_names(const kernel_adapter& kernel_adapter, parameter_direction_t direction);

template <typename Predicate, typename = std::enable_if_t<not std::is_same<Predicate, parameter_direction_t>::value, void> >
name_set buffer_names(const kernel_adapter& kernel_adapter, Predicate pred)
{
    return util::transform_if<name_set>(kernel_adapter.parameter_details(),
        [&pred](const auto& spd) {
            return spd.kind == kernel_parameters::kind_t::buffer and pred(spd);
        },
        [](const auto& spd) { return spd.name; }
    );
}

namespace kernel_adapters {

template <typename U>
static void register_in_factory()
{
    bool dont_ignore_repeat_registrations { false };
    kernel_adapter::register_in_factory<U>(U::key_, dont_ignore_repeat_registrations);
}

// TODO:
// 1. Perhaps we should wrap the raw argument vector with methods for pushing back?
//    and arrange it so that when its used, e.g. for casting into a void**, we also
//    append the final nullptr?
// 2. Consider placing the argument_ptrs_and_maybe_sizes vector in the test context; not sure why
//    it should be outside of it.
void push_back_buffer(
    marshalled_arguments_type& argument_ptrs_and_maybe_sizes,
    const execution_context_t& context,
    parameter_direction_t dir,
    const char* buffer_parameter_name);

template <typename Scalar>
inline void push_back_scalar(
    marshalled_arguments_type& argument_ptrs,
    const execution_context_t& context,
    const char* scalar_parameter_name)
{
    argument_ptrs.pointers.push_back(& get_scalar_argument<Scalar>(context, scalar_parameter_name));
    if (context.ecosystem == execution_ecosystem_t::opencl) {
        argument_ptrs.sizes.push_back(sizeof(Scalar));
    }
}

} // namespace kernel_adapters

template <typename Scalar>
inline void kernel_adapter::pusher(
    marshalled_arguments_type& argument_ptrs_and_maybe_sizes,
    const execution_context_t& context,
    const char* scalar_parameter_name)
{
    return kernel_adapters::push_back_scalar<Scalar>(argument_ptrs_and_maybe_sizes, context, scalar_parameter_name);
}

bool is_input (kernel_adapter::single_parameter_details spd);
bool is_output(kernel_adapter::single_parameter_details spd);

std::size_t apply_size_calc(const size_calculator_type& calc, const execution_context_t& context);

// Boilerplate macros for subclasses of kernel_adapter.
// Each of these needs to be invoked once in any subclass
// definition

// The name of the kernel function in the source file
#define KA_KERNEL_FUNCTION_NAME(kfn) \
    constexpr static const char* kernel_function_name_ { kfn }; \
    std::string kernel_function_name() const override { return kernel_function_name_; }

// The key to be passed to the kernel-runner executable for using
// this kernel adapter. Must be unique.
#define KA_KERNEL_KEY(kk) \
    constexpr static const char* key_ { kk }; \
    std::string key() const override { return key_; }

#endif /* KERNEL_ADAPTER_HPP_ */
