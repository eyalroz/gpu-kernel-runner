#ifndef UTIL_FROM_STRING_HPP_
#define UTIL_FROM_STRING_HPP_

#include <string>
#include <stdexcept>

namespace util {

// Poor man's boost::lexical_cast... :-(
template <typename T> inline T from_string(const std::string& str);

template <> inline int                from_string<int               >(const std::string& str) { return std::stoi(str);   }
template <> inline long               from_string<long              >(const std::string& str) { return std::stol(str);   }
template <> inline unsigned long      from_string<unsigned long     >(const std::string& str) { return std::stoul(str);  }
template <> inline long long          from_string<long long         >(const std::string& str) { return std::stoll(str);  }
template <> inline unsigned long long from_string<unsigned long long>(const std::string& str) { return std::stoull(str); }
template <> inline float              from_string<float             >(const std::string& str) { return std::stof(str);   }
template <> inline double             from_string<double            >(const std::string& str) { return std::stod(str);   }
template <> inline long double        from_string<long double       >(const std::string& str) { return std::stold(str);  }

template <> inline unsigned           from_string<unsigned          >(const std::string& str) { return static_cast<unsigned>(std::stoul(str));  }
template <> inline bool               from_string<bool              >(const std::string& str)
{
	if (str == "true" or str == "TRUE" or str == "True" or str == "1") { return true; }
	if (str == "false" or str == "FALSE" or str == "False" or str == "0") { return false; }
	throw std::invalid_argument("Cannot directly parse string as a boolean value");
}
template <> inline std::string        from_string<std::string       >(const std::string& str) { return str;              }

} // namespace util

#endif // UTIL_FROM_STRING_HPP_
