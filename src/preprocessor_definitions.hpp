#ifndef PREPROCESSOR_DEFINITIONS_HPP_
#define PREPROCESSOR_DEFINITIONS_HPP_

#include "util/from_string.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <string>

using valued_preprocessor_definitions_t = std::unordered_map<std::string, std::string>;
using preprocessor_definitions_t = std::unordered_set<std::string>;
using name_set = std::unordered_set<std::string>;

inline name_set get_defined_terms(const preprocessor_definitions_t definitions)
{
    return util::transform<name_set>(
        definitions,
        [](const std::string& definition) {
            auto pos = definition.find('=');
            switch(pos) {
                case std::string::npos:
                    return definition;
                    break;
                case 0:
                    throw std::invalid_argument("Invalid command-line argument " + definition + ": Empty defined string");
            }
            // If the string happens to have "=" at the end, e.g. "FOO=" -
            // it's an empty definition -  which is fine.
            return definition.substr(0, pos);
        }
    );
}

/**
 * Imbue a string defined for the preprocessor with a type, to obtain
 * the value as it will (supposedly) be used in the source code.
 *
 * @param map Some preprocessor definitions with values
 * @param defined_term The term for which a preprocessor define was made (the
 * `WHATEVER` in `-DWHATEVER=SOME_VALUE`)
 * @return the parsed/type-imbued defined value
 * @throws std::runtime_error if the term hasn't been defined
 */
template <typename T>
optional<T> safe_get_defined_value(const valued_preprocessor_definitions_t& map, const std::string& defined_term)
{
    auto find_result = map.find(defined_term);
    if (find_result == std::cend(map)) {
        return nullopt;
    }
    return util::from_string<T>(find_result->second);
}

/**
 *  Retrieve and cast a value provided as a KEY=VALUE preprocsser definition, given the KEY string.
 *
 *  @note the cast is safe, using @ref `util::from_string`; but is not infallible.
 *  @throws A runtime_error on parsing failure
 */
template <typename T>
T get_defined_value(const valued_preprocessor_definitions_t& map, const std::string& defined_term)
{
    auto result = safe_get_defined_value<T>(map, defined_term);
    if (not result) {
        throw std::runtime_error("Could not find key " + defined_term + " in the valued preprocessor definitions map");
    }
    return result.value();
}

#endif /* PREPROCESSOR_DEFINITIONS_HPP_ */
