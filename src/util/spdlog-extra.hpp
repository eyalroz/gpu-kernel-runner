#ifndef UTIL_SPDLOG_EXTRA_HPP_
#define UTIL_SPDLOG_EXTRA_HPP_

// #include <common.hpp>

#include <spdlog/spdlog.h>

namespace spdlog {

inline bool level_is_at_least(spdlog::level::level_enum l) {
    return spdlog::default_logger_raw()->level() >= l;
}

} // namespace spdlog

#endif // UTIL_SPDLOG_EXTRA_HPP_

