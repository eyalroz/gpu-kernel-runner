#ifndef UTIL_SPDLOG_EXTRA_HPP_
#define UTIL_SPDLOG_EXTRA_HPP_

// #include <common.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>
#include <unordered_set>

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

#endif // UTIL_SPDLOG_EXTRA_HPP_

