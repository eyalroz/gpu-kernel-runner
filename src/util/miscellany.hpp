#ifndef UTIL_MISCELLANY_HPP_
#define UTIL_MISCELLANY_HPP_

#include <unordered_set>
#include <unordered_map>
#include <map>
#include <set>
#include <algorithm>
#include <string>
#include <cctype>
#include <util/optional_and_any.hpp>
#include "functional.hpp"

namespace util {

inline std::string to_lowercase(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(),
        [](unsigned char c){ return std::tolower(c); });
    return str;
}

namespace detail {

template <typename T, typename Key, typename = void>
struct has_find_method : std::false_type{};

template <typename T, typename Key>
struct has_find_method<T, Key, decltype(void(std::declval<T>().find(std::declval<const Key&>())))> : std::true_type {};

template <typename Container, typename Key>
auto find(
    [[maybe_unused]] std::true_type has_find_method,
    Container& container,
    const Key& x)
{
    return container.find(x);
}

template <typename Container, typename Key>
auto find(
    [[maybe_unused]] std::false_type doesnt_have_find_method,
    Container& container,
    const Key& x)
{
    return std::find(std::begin(container), std::end(container), x);
}

} // namespace detail

template <typename Container, typename Key>
auto find(Container& container, const Key& x)
{
    using container_has_find = std::integral_constant<bool, detail::has_find_method<Container, Key>::value>;
    return detail::find(container_has_find{}, container, x);
}

template <typename Container, typename Predicate>
typename Container::iterator find_if(Container& container, const Predicate& pred)
{
    return  std::find_if(std::begin(container), std::end(container), pred);
}

template <typename Container, typename Key>
bool contains(const Container& container, const Key& key)
{
    return find(container, key) != std::cend(container);
}

template <typename Container, typename Key>
bool remove_first(Container& container, const Key& key)
{
    auto it = find(container, key);
    auto found = (it != container.end());
    if (found) {
        container.erase(it);
    }
    return found;
}

template <typename Container, typename Key>
void remove(Container& container, const Key& key)
{
    std::remove(container.begin(), container.end(), key);
}
// This is a bit ugly, but we don't have ranges
template <typename Map>
std::unordered_set<typename Map::key_type>
keys(const Map& map)
{
    return util::transform<std::unordered_set<typename Map::key_type>>(
        map, [](const auto& pair){ return pair.first;} );
}

template <typename Map>
std::unordered_set<typename Map::key_type>
keys(Map&& map)
{
    const Map copy { map };
    return keys(copy);
}

// This is a bit ugly, but we don't have ranges
template <typename Map>
auto values(const Map& map)
{
    return util::transform<std::unordered_set<typename Map::value_type>>(map,
        [](const typename Map::value_type &pair){return pair.second;} );
}

template <typename Container>
Container union_(const Container& set_1, const Container& set_2)
{
    Container result;
    std::set_union(
        std::cbegin(set_1), std::cend(set_1),
        std::cbegin(set_2), std::cend(set_2),
        std::inserter(result, std::begin(result))
    );
    return result;
}

template <typename Map>
Map incremental_map_union(const Map& map_1, const Map& map_2)
{
    Map result;

    util::copy(map_1, result);
    util::copy_if(map_2, result,
      [&](const auto& kv_pair) {
          const auto& key = kv_pair.first;
          // set_1 has element with the same key
          return not contains(map_1, key);
      });
    return result;
}
template <typename LHSContainer, typename RHSContainer>
LHSContainer intersection(const LHSContainer& lhs, const RHSContainer& rhs)
{
    // TODO: This is an inefficient implementation; we could use a counting map for amortized-linear-time complexity
    LHSContainer result;
    std::copy_if(lhs.begin(), lhs.end(), std::inserter(result, result.end()),
        [&rhs] (const auto& needle) {
        return std::find(std::cbegin(rhs), std::cend(rhs), needle) != rhs.end();
    });
    return result;
}

template <typename Container>
Container difference(const Container& lhs, const Container& rhs)
{
/*
    // TODO: This is an inefficient implementation; we could use a counting map for amortized-linear-time complexity
    Container result;
    std::copy_if(lhs.begin(), lhs.end(), std::inserter(result, result.end()),
        [&rhs] (const auto& needle) { return rhs.find(needle) == rhs.end(); });
    return result;
*/
    Container result;
    std::set_difference(
        std::cbegin(lhs), std::cend(rhs),
        std::cbegin(lhs), std::cend(rhs),
        std::inserter(result, std::begin(result))
    );
    return result;
}

template<class InputIt, typename Delimiter>
constexpr std::ostream& implode(InputIt start, InputIt end, std::ostream& os, const Delimiter& delimiter)
{
    for(auto it = start; it < end; it++) {
        os << *it;
        if ((it+1) < end) {
            os << delimiter;
        }
    }
    return os;
}

template<class Container, typename Delimiter>
constexpr std::ostream& implode(const Container& container, std::ostream& os, const Delimiter& delimiter)
{
    return implode(std::begin(container), std::end(container), os, delimiter);
}

inline optional<std::string> get_env ( const char* key )
{
    const char* ev_val = getenv(key);
    if (ev_val == nullptr) {
        return nullopt;
    }
    return std::string(ev_val);
}

inline optional<std::string> get_env ( const std::string& key )
{
    return get_env(key.c_str());
}

inline bool case_insensitive_equals(const std::string& lhs, const std::string& rhs)
{
    return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
        [](char a, char b) {
            return tolower(a) == tolower(b);
        });
}

template <typename I, typename I2 = I>
constexpr I round_up(I x, I2 y) noexcept
{
    static_assert(std::is_signed<I>::value or std::is_unsigned<I2>::value,
        "Attempt to implicitly introduce signedness; make an explicit cast");
    return (x % y == 0) ? x : x + (y - x%y);
}

template <typename I, typename I2>
constexpr I div_rounding_up(I x, const I2 modulus) noexcept
{
    static_assert(std::is_signed<I>::value or std::is_unsigned<I2>::value,
                  "Attempt to implicitly introduce signedness; make an explicit cast");
    return (x + I{modulus} - I{1} ) / I{modulus};
}


template <typename I, typename I2 = I>
constexpr I round_down_to_power_of_2(I x, I2 power_of_2) noexcept
{
    return (x & ~(I{power_of_2} - 1));
}

template <typename I>
constexpr bool divides(I divisor, const I dividend) noexcept
{
    return dividend % divisor == 0;
}


/**
 * @note careful, this may overflow!
 */
template <typename I, typename I2 = I>
constexpr I round_up_to_power_of_2(I x, I2 power_of_2) noexcept
{
    return round_down_to_power_of_2 (x + I{power_of_2} - 1, power_of_2);
}

namespace detail {
template <typename T>
struct irange { T min, max; };
} // namespace detail

template <typename T, typename U>
inline bool in_range(T x, std::initializer_list<U> l)
{
    static_assert(std::is_convertible<T, U>::value, "The value must be convertible to the same type as the range extrema.");
    if (l.size() != 2) {
        throw std::invalid_argument("in_range passed something other than a min-and-max initialization list.");
    }
    auto x_ = static_cast<U>(x);
    return (x_ >= *l.begin()) and (x_ <= *(l.begin()+1));
}

//template <typename T>
//inline bool in_range(T x, T min, T max) { return in_range(x, {min, max}); }

inline bool is_valid_identifier(const std::string& str)
{
    if (str.length() == 0) { return false; }

    {
        char first_char = *str.cbegin();
        if (!(
              util::in_range(first_char, {'a', 'z'}) or
              util::in_range(first_char, {'A', 'Z'}) or
              first_char == '_'))
            return false;
    }

    // Traverse the string for the rest of the characters
    return std::all_of(str.cbegin()+1, str.cend(),
        [](auto ch) {
          return util::in_range(ch, {'a', 'z'}) or
            util::in_range(ch, {'A', 'Z'}) or
            util::in_range(ch, {'0', '9'}) or
            ch == '_';
        });
}

inline std::string newline_if_missing(const std::string& str)
{
    return (str.end()[-1] != '\n') ? "\n" : "";
}

template <class I>
int naive_num_digits(I number)
{
    int digits = (number < 0) ? 1 : 0;
        // Need a digit for the minus sign

    while (number >= 10) {
        number /= 10;
        digits++;
    }
    return digits + 1; // Note: 0 has a single digit
}

#if __cplusplus >= 201402L
template <typename S, typename T = S>
constexpr typename std::common_type<S,T>::type gcd(S u, T v) noexcept
{
// TODO: Shouldn't we first cast everything into the common type?
while (v != 0) {
    auto remainder = u % v;
    u = v;
    v = remainder;
    }
    return u;
}
#else
template <typename S, typename T = S>
constexpr KAT_FHD typename std::common_type<S,T>::type gcd(S u, T v) noexcept
{
        return (v == 0) ? u : gcd<std::common_type<S,T>::type>(v, u % v);
}
#endif

template <typename S, typename T = S>
constexpr typename std::common_type<S,T>::type lcm(S u, T v) noexcept
{
using result_type = typename std::common_type<S,T>::type;
return (result_type{u} / gcd(u,v)) * v;
}

} // namespace util

#endif // UTIL_MISCELLANY_HPP_
