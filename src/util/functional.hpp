#ifndef UTIL_FUNCTIONAL_HPP_
#define UTIL_FUNCTIONAL_HPP_

#include <algorithm>
#include <numeric>
#include <utility>
#include <iterator>

namespace util {

template<typename SourceContainer, typename DestinationContainer>
DestinationContainer copy(SourceContainer& src, DestinationContainer& dest)
{
    std::copy(src.cbegin(), src.cend(), std::inserter(dest, dest.begin()));
    return dest;
}

template<typename SourceContainer, typename DestinationContainer>
DestinationContainer append(SourceContainer&& src, DestinationContainer& dest)
{
    std::copy(src.cbegin(), src.cend(), std::back_inserter(dest));
    return dest;
}

template<typename SourceContainer, typename DestinationContainer, typename Predicate>
DestinationContainer copy_if(SourceContainer& src, DestinationContainer& dest, Predicate pred)
{
    std::copy_if(src.cbegin(), src.cend(), std::inserter(dest, dest.begin()), pred);
    return dest;
}

template<typename DestinationContainer, typename SourceContainer, typename F>
DestinationContainer transform(const SourceContainer& container, F func)
{
    DestinationContainer transformed;
    std::transform(container.cbegin(), container.cend(), std::inserter(transformed, transformed.begin()), func);
    return transformed;
}


namespace projected {

template <typename Container, typename Projector>
auto min_element(Container&& container, Projector projector) -> decltype(container.cbegin())
{
    using value_type = typename Container::value_type;
    auto comparator = [&projector](value_type const& lhs, value_type const& rhs) {
        return projector(lhs) < projector(rhs);
    };
    return std::min_element(std::cbegin(container), std::cend(container), comparator);
}


template <typename Container, typename Projector>
auto max_element(Container&& container, Projector projector) -> decltype(container.cbegin())
{
    using value_type = typename Container::value_type;
    auto comparator = [&projector](value_type const& lhs, value_type const& rhs) {
        return projector(lhs) < projector(rhs);
    };
    return std::max_element(std::cbegin(container), std::cend(container), comparator);
}

} // namespace projected

template <typename Container>
constexpr auto min_element(Container&& container) -> decltype(std::begin(container))
{
    return std::min_element(std::begin(container), std::end(container));
}

template <typename Container, typename Comparator>
constexpr auto min_element(Container&& container, Comparator comp) -> decltype(std::begin(container))
{
    return std::min_element(std::begin(container), std::end(container), comp);
}

template <typename Container>
constexpr auto max_element(Container&& container) -> decltype(std::begin(container))
{
    return std::max_element(std::begin(container), std::end(container));
}

template <typename Container, typename Comparator>
constexpr auto max_element(Container&& container, Comparator comp) -> decltype(std::begin(container))
{
    return std::max_element(std::begin(container), std::end(container), comp);
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

template<template<class, class> class Map, typename Key, typename Value , typename F>
auto map_keys(const Map<Key,Value>& map, const F& key_mapper)
{
    using mapped_key_type = decltype(key_mapper(std::declval<Key>()));
    static_assert(std::is_same<mapped_key_type, std::string>::value, "not the same");
    return transform<Map<mapped_key_type, Value>>(
        map,
        [&key_mapper](const auto& pair) -> std::pair<mapped_key_type, Value> {
            const auto& key = pair.first;
            const auto& value = pair.second;
            return {key_mapper(key), value};
        } );
}

template<typename Container, typename F>
Container filter(const Container& container, F predicate)
{
    Container filtered;
    std::copy_if(container.cbegin(), container.cend(), std::inserter(filtered, filtered.end()), predicate);
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

template <typename OutContainer, typename InContainer, typename Predicate, typename UnaryOperator>
OutContainer transform_if(InContainer in_container, Predicate pred, UnaryOperator op)
{
    OutContainer result;
    transform_if(std::cbegin(in_container), std::cend(in_container),
        std::inserter(result, std::end(result)), std::forward<Predicate>(pred), std::forward<UnaryOperator>(op));
    return result;
}

} // namespace util

#endif // UTIL_FUNCTIONAL_HPP_
