#ifndef KERNELS_CONSTEXPR_MATH_CUH_
#define KERNELS_CONSTEXPR_MATH_CUH_

#include <limits>

namespace constexpr_ {

namespace detail
{
    double constexpr sqrt_newton_raphson_iteration(double x, double curr, double prev)
    {
        return (curr == prev) ? curr
            : sqrt_newton_raphson_iteration(x, 0.5 * (curr + x / curr), curr);
    }
}

template <typename F>
F constexpr sqrt(F x, F initial_guess = 0)
{
    return x >= 0 && x < std::numeric_limits<F>::infinity()
        ? detail::sqrt_newton_raphson_iteration(x, x, initial_guess)
        : std::numeric_limits<F>::quiet_NaN();
}

template <typename I, typename F>
constexpr I ceil(F num)
{
    return (static_cast<F>(static_cast<I>(num)) == num)
        ? static_cast<I>(num)
        : static_cast<I>(num) + ((num > 0) ? 1 : 0);
}

template <typename I, typename F>
constexpr I floor(F num)
{
    return (static_cast<F>(static_cast<I>(num)) == num)
        ? static_cast<I>(num)
        : static_cast<I>(num) - ((num > 0) ? 0 : 1);
}


template <typename I>
constexpr I factorial(I x)
{
    I result { 1 };
    for(int i = 2; i <= x; i++) {
        result *= i;
    }
    return result;
}

template <typename F, typename I>
constexpr I power(F x, I exponent)
{
    // Poor implementation! Inaccurate and slow
    F result = 1.0;
    for(; exponent > 0; exponent--) {
        result *= x;
    }
    return result;
}


template <typename F>
F constexpr sin(F x);

template <>
double constexpr sin(double x)
{
    static constexpr const auto max_taylor_expansion_power = 13;
    double result = 0;
    for(int i = 1; 2*i -1 <= max_taylor_expansion_power; i ++) {
        auto exponent = 2*i - 1;
        auto sign = i % 2 ? 1.0 : - 1.0;
        result += sign * power<double>(x, exponent) / factorial(exponent);
    }
    return result;
}

template <typename I>
constexpr I div_rounding_up(I dividend, I divisor)
{
    // TODO: May not be correct for negative arguments (check definition)
    return dividend / divisor + (dividend % divisor == 0) ? 0 : 1;
}

template <typename I, typename I2 = I>
constexpr I round_up(I x, I2 y) noexcept
{
    return (x % y == 0) ? x : x + (y - x%y);
}

} // namespace constexpr_

#endif /* KERNELS_CONSTEXPR_MATH_CUH_ */
