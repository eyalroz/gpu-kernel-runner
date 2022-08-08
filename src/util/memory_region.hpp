#ifndef GPU_KERNEL_RUNNER_UTIL_SPAN_HPP_
#define GPU_KERNEL_RUNNER_UTIL_SPAN_HPP_

#include <stdlib.h> // for std::size_t

namespace util {

#if __cplusplus >= 201712L
using byte_type = std::byte;
#else
using byte_type = char;
#endif

struct const_memory_region {
    using value_type = const byte_type;
    using size_type = std::size_t;

    value_type* data_;
    size_type   size_;

    value_type * const & data() const { return data_; }
    value_type * & data() { return data_; }
    const size_type& size() const { return size_; }
    size_type& size() { return size_; }
};

struct memory_region {
    using value_type = byte_type;
    using size_type = std::size_t;

    value_type* data_;
    size_type   size_;

    value_type * const & data() const { return data_; }
    value_type * & data() { return data_; }
    const size_type& size() const { return size_; }
    size_type& size() { return size_; }

    operator const_memory_region() const { return { data_, size_ }; }
};

template <typename ContiguousContainer>
inline auto as_region(ContiguousContainer& container)
{
    using region_type = typename std::conditional_t<
        std::is_const<std::remove_reference_t<decltype(*(container.data()))>>::value,
        const_memory_region, memory_region>;

    return region_type { container.data(), container.size() };
}

} // namespace util


#endif // GPU_KERNEL_RUNNER_UTIL_SPAN_HPP_