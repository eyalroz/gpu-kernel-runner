#ifndef UTIL_FUNCTIONAL_HPP_
#define UTIL_FUNCTIONAL_HPP_

#include <algorithm>
#include <numeric>
#include <utility>
#include <iterator>

namespace util {

template<typename DestinationContainer, typename SourceContainer, typename F>
DestinationContainer transform(const SourceContainer& container, F func)
{
    DestinationContainer transformed;
    std::transform(container.cbegin(), container.cend(), std::inserter(transformed, transformed.begin()), func);
    return transformed;
}

// This applies the functional operator "map" to the C++ data structure Map (e.g. std::map or std::unordered_map)
template<template<class, class> class Map, typename Key, typename Value , typename F>
auto map_values(const Map<Key,Value>& map, const F& value_mapper)
{
    using mapped_value_type = decltype(value_mapper(std::declval<Value>()));
    return transform<Map<Key, mapped_value_type>>(
        map,
        [&value_mapper](const auto& pair) -> std::pair<Key,mapped_value_type> {
            const auto& key = pair.first;
            const auto& value = pair.second;
            return {key, value_mapper(value)};
        } );
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

template <typename InputIterator, typename OutputIterator, typename Predicate, typename UnaryOperator>
OutputIterator transform_if(
    InputIterator first1, InputIterator last1, OutputIterator d_first, Predicate pred, UnaryOperator op)
{
    for(; first1 != last1; ++first1) {
        const auto& e = *first1;
        if (not pred(e)) {
            continue;
        }
        *d_first = op(e);
        d_first++;
    }
    return d_first;
}

} // namespace util

#endif // UTIL_FUNCTIONAL_HPP_
