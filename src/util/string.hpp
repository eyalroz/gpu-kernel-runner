#ifndef UTIL_STRING_HPP_
#define UTIL_STRING_HPP_

#include <string>
#include <vector>
#include <algorithm>

namespace util {

inline bool case_insensitive_equals(const std::string& lhs, const std::string& rhs)
{
	return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
		[](char a, char b) {
			return tolower(a) == tolower(b);
		});
}

inline std::string newline_if_missing(const std::string& str)
{
	return str.empty() ? "\n" : (str.end()[-1] != '\n') ? "\n" : "";
}

/// Make all Latin characters in a string lowercase (in-place)
inline std::string to_lowercase(std::string str) {
	std::transform(str.begin(), str.end(), str.begin(),
		[](unsigned char c){ return std::tolower(c); });
	return str;
}

/// Make all Latin characters in a string uppercase (in-place)
inline std::string to_uppercase(std::string str) {
	std::transform(str.begin(), str.end(), str.begin(),
		[](unsigned char c){ return std::toupper(c); });
	return str;
}
/**
 * Splits a string into a sequence of strings between occurrences
 * of a chosen delimiter.
 *
 * @note
 * 1. the resulting strings do not contain any of the delimiters; those are discarded
 * 2. a delimiter at the beginning of the string, as well as  multiple consecutive
 *    delimiters, do not produce empty strings.
 * 3. This should probably have returned a dynarray rather than a vector
 *
 * @note As per @url https://stackoverflow.com/a/60782724/1593077
 */
inline std::vector<std::string> split(const std::string& str, char delim)
{
	std::vector<std::string> strings;
	size_t start;
	size_t end = 0;
	while ((start = str.find_first_not_of(delim, end)) != std::string::npos) {
		end = str.find(delim, start);
		strings.push_back(str.substr(start, end - start));
	}
	return strings;
}

/**
 * Removes (in-place) space characters from the beginning of a string
 *
 * @note as per @url https://stackoverflow.com/a/217605/1593077
 */
inline void ltrim(std::string &s) {
	s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
		return !std::isspace(ch);
	}));
}

/**
 * Removes (in-place) space characters from the end of a string
 *
 * @note as per @url https://stackoverflow.com/a/217605/1593077
 */
inline void rtrim(std::string &s) {
	s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
		return !std::isspace(ch);
	}).base(), s.end());
}

/// Removes (in-place) space characters from both ends of a string
inline void trim(std::string &s) {
	rtrim(s);
	ltrim(s);
}

} // namespace util

#endif // UTIL_STRING_HPP_
