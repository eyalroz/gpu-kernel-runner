#ifndef PREPROCESSOR_DEFINITIONS_HPP_
#define PREPROCESSOR_DEFINITIONS_HPP_

#include <util/from_string.hpp>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <string>

using preprocessor_value_definitions_t = std::unordered_map<std::string, std::string>;
using preprocessor_definitions_t = std::unordered_set<std::string>;
using parameter_name_set = std::unordered_set<std::string>;

inline parameter_name_set
get_defined_terms(const preprocessor_definitions_t definitions)
{
    parameter_name_set results;
    std::transform(
        definitions.cbegin(), definitions.cend(),
        std::inserter(results, results.begin()),
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
    return results;
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
T get_defined_value(const preprocessor_value_definitions_t& map, const std::string& defined_term)
{
    auto find_result = map.find(defined_term);
    if (find_result == std::cend(map)) {
        throw std::runtime_error("Could not find key " + defined_term + " in the valued preprocessor definitions map");
    }
    return util::from_string<T>(find_result->second);
}

#endif /* PREPROCESSOR_DEFINITIONS_HPP_ */
