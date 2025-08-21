#ifndef UTIL_SPDLOG_EXTRA_HPP_
#define UTIL_SPDLOG_EXTRA_HPP_

#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>

#include <unordered_set>
#include <string>
#include <cstdlib>

namespace spdlog {

inline bool level_is_at_least(spdlog::level::level_enum l) {
    return spdlog::default_logger_raw()->level() >= l;
}

} // namespace spdlog

template <typename T> class fmt::formatter<std::unordered_set<T>> {
public:
    constexpr auto parse (format_parse_context& ctx) { return ctx.begin(); }
    template <typename Context>
    constexpr auto format (std::unordered_set<T> const& set, Context& ctx) const {
        bool first = true;
        format_to(ctx.out(), "{{");
        for(const T& e : set) {
            format_to(ctx.out(), first ? "{}" : ", {}", e);
            first = false;
        }
        return format_to(ctx.out(), "}}");
    }
};

template <typename T> class fmt::formatter<std::vector<T>> {
public:
    constexpr auto parse (format_parse_context& ctx) { return ctx.begin(); }
    template <typename Context>
    constexpr auto format (std::vector<T> const& set, Context& ctx) const {
        bool first = true;
        format_to(ctx.out(), "(");
        for(const T& e : set) {
            format_to(ctx.out(), first ? "{}" : ", {}", e);
            first = false;
        }
        return format_to(ctx.out(), ")");
    }
};

template <typename... Ts>
[[noreturn]] inline bool die(std::string message_format_string = "", Ts&&... args)
{
    if(not message_format_string.empty()) {
        spdlog::critical(message_format_string, std::forward<Ts>(args)...);
    }
    exit(EXIT_FAILURE);
}

inline void set_log_level(std::string log_level_name)
{
    auto log_level = spdlog::level::from_str(log_level_name);
    if (spdlog::level_is_at_least(spdlog::level::debug)) {
        spdlog::log(spdlog::level::debug, "Setting log level to {}", log_level_name);
    }
    spdlog::set_level(log_level);
}

#endif // UTIL_SPDLOG_EXTRA_HPP_

