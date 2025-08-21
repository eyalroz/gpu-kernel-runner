#include "parsed_cmdline_options.hpp"

#include <cxxopts/cxxopts.hpp>

parsed_cmdline_options_t parse_command_line(int argc, char** argv);

inline bool early_exit_needed(parsed_cmdline_options_t const& parsed)
{
    return parsed.early_exit_action.operator bool();
}