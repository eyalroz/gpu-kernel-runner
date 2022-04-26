#ifndef UTIL_FUNCTIONAL_HPP_
#define UTIL_FUNCTIONAL_HPP_

#include <algorithm>
#include <numeric>
#include <utility>
#include <iterator>

namespace util {

template<typename Container, typename F>
Container map(const Container& container, F mapper)
{
    Container mapped;
    std::transform(container.cbegin(), container.cend(), std::inserter(mapped, mapped.begin()), mapper);
    return mapped;
}

// This applies the functional operator "map" to the C++ data structure Map (e.g. std::map or std::unordered_map)
template<template<class, class> typename Map, typename Key, typename Value , typename F>
auto map_values(const Map<Key,Value>& map, const F& value_mapper)
{
    using mapped_value_type = decltype(value_mapper(std::declval<Value>()));
    Map<Key, mapped_value_type> mapped;
    std::transform(map.cbegin(), map.cend(), std::inserter(mapped, mapped.begin()),
        [&value_mapper](const auto& pair) {
            const auto& key = pair.first;
            const auto& value = pair.second;
            return {key, value_mapper(value)};
        }
    );
    return mapped;
}

template<typename Container, typename F>
Container filter(const Container& container, F predicate)
{
    Container filtered;
    std::copy_if(container.cbegin(), container.cend(), std::inserter(filtered, filtered.begin()), predicate);
    return filtered;
}

template<typename Container, typename T, typename BinaryOp>
T fold(const Container& container, const T& initial_value, BinaryOp folder)
{
    return std::accumulate(container.cbegin(), container.cend(), initial_value, folder);
}

} // namespace util

#endif // UTIL_FUNCTIONAL_HPP_
