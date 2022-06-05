/**
 * @file factory_producible.hpp
 *
 * Copyright (c) 2017-2019, Eyal Rozenberg and CWI Amsterdam
 * Copyright (c) 2019-2020, Eyal Rozenberg
 *
 * Licensed under the BSD 3-clause license
 * https://opensource.org/licenses/BSD-3-Clause
 */
#ifndef UTIL_FACTORY_PRODUCIBLE_HPP_
#define UTIL_FACTORY_PRODUCIBLE_HPP_

#include <typeinfo>
 
#include "factory.hpp"
#include "type_name.hpp"
#include <string>
#include <exception>
#include <memory>

namespace util {

namespace mixins {

using std::logic_error;

/**
 * Note: This is a mix-in class, and you can't actually instantiate it!
 * I mean, maybe you sort of can, but it won't work unless you
 * implement the main "pseudo-virtual" method
 */
template <typename Key, typename Base, typename... ConstructionArgs>
class factory_producible {
public:

	template <typename U>
	static void register_in_factory(const Key& key, bool ignore_repeat_registration = true) {
		get_subclass_factory_().template register_class<U>(key, ignore_repeat_registration);
	}

protected:
	using subclass_factory_type = util::exposed_factory<Key, Base, ConstructionArgs...>;

	/**
	 * This method is sort of an attempt to avoid the static initialization
	 * fiasco; if you call it, you're guaranteed that the initialization
	 * of subclass_factory happens before you get it.
	 *
	 * @return the class' static factory for producing subclasses - initialized
	 */

	static subclass_factory_type& get_subclass_factory_() {
		static subclass_factory_type subclass_factory;
		return subclass_factory;
	}


public:
	static const subclass_factory_type& get_subclass_factory() {
		return get_subclass_factory_();
	}

	// This is not implemented generically for the mixin class, which
	// makes it a sort of a virtual static method - but virtual only in
	// the sense of the template arguments.
	static Key resolve_subclass_key(ConstructionArgs... args);


	static std::unique_ptr<Base> produce_subclass(const Key& subclass_key, ConstructionArgs... args) {
		if (not get_subclass_factory().can_produce(subclass_key)) {
			throw std::invalid_argument(std::string("No subclass of the base type ")
				+ util::type_name<Base>() + " is registered with key \""
				+ std::string(subclass_key) + "\"");
		}
		return std::unique_ptr<Base> { get_subclass_factory().produce(subclass_key, args...) };
	}

	static std::unique_ptr<Base> produce_subclass(ConstructionArgs... args) {
		Key subclass_key = resolve_subclass_key(args...);
		return produce_subclass(subclass_key, args...);
	}

    static bool can_produce_subclass(const Key& subclass_key, ConstructionArgs...) {
        return get_subclass_factory().can_produce(subclass_key);
    }

	static bool can_produce_subclass(ConstructionArgs... args) {
	    auto subclass_key = resolve_subclass_key(args...);
		return can_produce_subclass(subclass_key, args...);
	}

	factory_producible(const factory_producible& other) = default;
	factory_producible() = default;
	~factory_producible() = default;
};

} // namespace mixins
} // namespace util

#endif // UTIL_FACTORY_PRODUCIBLE_HPP_
