#ifndef UTIL_CXXOPTS_EXTRA_HPP_
#define UTIL_CXXOPTS_EXTRA_HPP_

#include <cxxopts/cxxopts.hpp>
#include <cxx-prettyprint/prettyprint.hpp>

#include <vector>

inline bool contains(cxxopts::ParseResult& parse_result, const char* option_name)
{
    return (parse_result.count(option_name) > 0);
}
inline bool contains(cxxopts::ParseResult& parse_result, const std::string& option_name)
{
    return (parse_result.count(option_name) > 0);
}

inline cxxopts::ParseResult non_consumptive_parse(cxxopts::Options& opts, int argc, char** argv)
{
    std::vector<char*> new_argv{argv, argv + argc};
    char** new_argv_data_copy = new_argv.data();
    return opts.parse(argc, new_argv_data_copy);
}

struct string_option_spec {
    const char* name;
    const char* description;
    const char* default_value; // nullptr for no default
};

inline void add_string_option(cxxopts::OptionAdder& adder, string_option_spec option_spec)
{
    adder(option_spec.name, option_spec.description,
        option_spec.default_value ?
        cxxopts::value<std::string>()->default_value(option_spec.default_value) :
        cxxopts::value<std::string>());
}

template <typename StringOptionSpecContainer>
inline void add_string_options(
    cxxopts::Options& opts,
    std::string option_group_name,
    StringOptionSpecContainer&& string_options)
{
    auto adder = opts.add_options(option_group_name);
    for(const string_option_spec& option_spec : string_options) {
        add_string_option(adder, option_spec);
    }
}

#endif // UTIL_CXXOPTS_EXTRA_HPP_
