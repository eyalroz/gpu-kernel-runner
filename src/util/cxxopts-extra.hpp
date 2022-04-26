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

#endif // UTIL_CXXOPTS_EXTRA_HPP_
