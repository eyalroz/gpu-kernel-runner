/**
 * @file factory.hpp

 * Implementation of the factory pattern, based on suggestions here:
 *
 * http://stackoverflow.com/q/5120768/1593077
 *
 * and on the suggestions for corrections here:
 *
 * http://stackoverflow.com/a/34948111/1593077
 *
 * Copyright (c) 2016-2019, Eyal Rozenberg and CWI Amsterdam
 * Copyright (c) 2019-2020, Eyal Rozenberg
 *
 * Licensed under the CC-SA-BY license v3.0
 * https://creativecommons.org/licenses/by-sa/3.0/legalcode
 */
#pragma once
#ifndef UTIL_FACTORY_HPP_
#define UTIL_FACTORY_HPP_

#include <unordered_map>
#include <exception>
#ifndef AVOID_STREAMING_KEY_CONTENTS
#include <type_traits>
#include <utility>
#include <iostream>
#include <sstream>
#endif

namespace util {

#ifndef AVOID_STREAMING_KEY_CONTENTS
template<typename S, typename T>
class is_streamable
{
    template<typename SS, typename TT>
    static auto test(int)
    -> decltype( std::declval<SS&>() << std::declval<TT>(), std::true_type() );

    template<typename, typename>
    static auto test(...) -> std::false_type;

public:
    static const bool value = decltype(test<S,T>(0))::value;
};

namespace detail_ {

template <typename K>
std::string maybe_stringify(std::integral_constant<bool, true>, const K& key)
{
    thread_local std::ostringstream oss;
    oss << ": " << key;
	return oss.str();
}

template <typename K>
std::string maybe_stringify(std::integral_constant<bool, false>, const K& key)
{
	return "";
}

template <typename K>
std::string maybe_stringify(const K& key)
{
	return maybe_stringify<K>(std::integral_constant<bool, is_streamable<std::ostream, K>::value>{}, key);
}
  
} // namespace detail_

#endif // AVOID_STREAMING_KEY_CONTENTS

// Note: ProductionArgs are not necessarily the same as the parameters
// of any of the constructors
template<typename Key, typename T, typename... ConstructionArgs>
class factory {
public:
	using instantiator_type = T* (*)(ConstructionArgs...);

public:
    T* produce(const Key& subclass_key, ConstructionArgs... args) const
    {
        auto it = subclass_instantiators.find(subclass_key);
        if (it == subclass_instantiators.end()) {
            throw std::invalid_argument("No class with the specified key is registered in the factory.");
        }
        auto instantiator = it->second;
        return instantiator(std::forward<ConstructionArgs>(args)...);
    }

    bool can_produce(const Key& subclass_key) const {
        return subclass_instantiators.find(subclass_key) != subclass_instantiators.end();
    }

protected:
	template<typename U>
	static T* create_instance(ConstructionArgs... args)
	{
		return new U(std::forward<ConstructionArgs>(args)...);
	}
	using instantiator_map = std::unordered_map<Key,instantiator_type>;

	instantiator_map subclass_instantiators;

protected:
	// Returns true if the call actually registered the class
	template<typename U>
	bool maybe_register_class(const Key& key)
	{
		static_assert(std::is_base_of<T, U>::value,
			"This factory cannot register a class which is is not actually "
			"derived from the factory's associated class");
		if (can_produce(key)) {
			return false;
		}
		subclass_instantiators.emplace(key, &create_instance<U>);
		return true;
	}

public:
	template<typename U>
	void register_class(const Key& key, bool ignore_repeat_registration = true)
	{
		auto wasnt_registered = maybe_register_class<U>(key);
		if (not wasnt_registered and not ignore_repeat_registration) {
            std::string message {"Repeat registration of the same subclass in this factory" };
            message += detail_::maybe_stringify(key);
			throw std::logic_error(message);
			
		}
		return;
	}
};

/**
 * This is for when you want to be able to list factory contents.
 */
template<typename Key, typename T, typename... ConstructionArgs>
class exposed_factory : public factory<Key, T, ConstructionArgs...> {

    using parent = factory<Key, T, ConstructionArgs...>;
    using instantiator_type = typename parent::instantiator_type;
    using instantiator_map = typename parent::instantiator_map;

public:
    const instantiator_map& instantiators() const {
        return parent::subclass_instantiators;
    }
};

} // namespace util

#endif // UTIL_FACTORY_HPP_ 
