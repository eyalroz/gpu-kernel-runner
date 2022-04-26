#ifndef KERNELS_MISCELLANY_CUH_
#define KERNELS_MISCELLANY_CUH_

// Note: I'm trying to avoid use of standard-library facilities, since these are a bit difficult to access with NVRTC.

enum : unsigned {warp_size = 32 }; // Like CUDA's warpSize, but exists at compile-time

inline bool logical_xor(bool x, bool y) { return x != y; }
inline bool logical_xor(bool x, bool y, bool z) {
    return
        ((x == y) and not z) or
        ((x == z) and not y) or
        ((y == z) and not x);
}

int operator==(const uint3& lhs, const uint3& rhs)
{
    return
        lhs.x == rhs.x and
        lhs.y == rhs.y and
        lhs.z == rhs.z;
}

template <typename F>
void do_once_per_grid(F f)
{
    if (blockIdx == uint3{ 0, 0, 0 } and threadIdx == uint3{ 0, 0, 0 }) {
        f();
    }
}

template <typename F>
void do_in_first_block(F f)
{
    if (blockIdx == uint3{ 0, 0, 0 }) {
        f();
    }
}

template <typename F>
void do_in_last_block(F f)
{
    if (blockIdx == uint3{ gridDim.x-1, gridDim.y-1, gridDim.z-1 }) {
        f();
    }
}

template <typename T, typename B>
constexpr bool between(T v, B lb, B ub)
{
    return v >= lb and v < ub;
}

template <typename T2>
inline constexpr T2 flip(T2 v) { return T2 { v.x, v.y }; }

/**
 * Returns the fractional part of a floating-point number.
 *
 * @param x A non-zero, non-infinity, non-NaN floating-point value
 * @return Implemented as per the suggestion at
 * https://stackoverflow.com/a/65596601/1593077
 *
 */
template <typename F>
inline F fractional_part(F x)
{
    return x - truncf(x);
}

template <typename F>
void kahan_add(F& sum, F& compensation, F new_element)
{
    F y = new_element - compensation;
    F t = sum + y;
    compensation = (t - sum) - y;
    sum = t;
}

template <typename F, typename Size>
F kahan_sum(const F* data, Size num_elements)
{
    F result(0);
    F compensation(0);
    for(Size i = 0; i < num_elements; i++) {
        kahan_sum(result, compensation, data[i]);
    }
    return result;
}

// This is something you would use iterators for - so,
// an example of avoiding standard library facilities
template <typename F, typename Size>
F strided_kahan_sum(const F* data, Size num_elements, Size stride)
{
    F result(0);
    F compensation(0);
    for(Size i = 0; i < num_elements; i++) {
        kahan_add<F>(result, compensation, data[i * stride]);
    }
    return result;
}

void trap() { asm("trap;"); }

namespace warp {

template<typename T, typename F>
T reduce(T value, F f)
{
    auto partial_result { value };
    for (int shuffle_mask = warp_size/2; shuffle_mask > 0; shuffle_mask >>= 1) {
        constexpr const unsigned full_warp_mask = 0xFFFFFFFFu;
        partial_result = f(partial_result, __shfl_xor_sync(full_warp_mask, partial_result, shuffle_mask));
    }
    return partial_result;
}

// This assumes the active threads are consecutive from 0
// The first lane holds a valid result
template<typename T, typename F>
T reduce_for_incomplete_warp(T value, F f)
{
    auto partial_result { value };
    unsigned active = __activemask();
    unsigned lane_index = threadIdx.x % warp_size;
    for (int delta = warp_size/2; delta > 0; delta >>= 1) {
        auto shuffle_result = __shfl_down_sync(active, partial_result, delta);
        auto source_lane_mask = 1 << (lane_index + delta);
        auto source_lane_is_active = source_lane_mask & active;
        if (source_lane_is_active) {
            partial_result = f(partial_result, shuffle_result);
        }
    }
    return partial_result;
}

} // namespace warp

namespace block {

template <typename T, bool Synchronize = true>
void share_per_warp_data(
    T                 datum,
    T*  __restrict__  where_to_make_available,
    unsigned          writing_lane_id)
{
    if (threadIdx.x % warp_size == writing_lane_id) {
        where_to_make_available[threadIdx.x / warp_size] = datum;
    }
    if (Synchronize) __syncthreads();
}

template<unsigned BlockSize, typename T, typename F>
T reduce(T value, F f, T neutral_value) // first warp hold result
{
    static __shared__ T warp_reductions[warp_size];
    auto intra_warp_result = warp::reduce<T,F>(value, f);

    constexpr const unsigned first_lane_writes = 0;
    block::share_per_warp_data(intra_warp_result, warp_reductions, first_lane_writes);

    __syncthreads(); // Perhaps we can do with something weaker here?

    // shared memory now holds all intra-warp reduction results


    if (threadIdx.x >= warp_size) { return; }

    // read from shared memory only if that warp actually existed; also,
    // this assumes the block is made up of full warps.
    constexpr const unsigned warps_per_block = BlockSize / warp_size;
    auto other_warp_result  = (threadIdx.x < warps_per_block) ?
        warp_reductions[threadIdx.x] : neutral_value;

    return warp::reduce<T,F>(other_warp_result, f);
}


// Assumes the first lane in each warp provides a warp value
// The first block thread holds the result
template<unsigned BlockSize, typename T, typename F>
T reduce_from_warp_values(T warp_reduction, F f)
{
    constexpr const unsigned warps_per_block =
        BlockSize / warp_size + (BlockSize % warp_size ? 1 : 0);
    static __shared__ T warp_reductions[warps_per_block];

    constexpr const unsigned first_lane_writes = 0;
    block::share_per_warp_data(warp_reduction, warp_reductions, first_lane_writes);

    __syncthreads(); // Perhaps we can do with something weaker here?

    // shared memory now holds all intra-warp reduction results

    if (threadIdx.x > 0) { return; }

    // TODO: Could have used some shuffles here in case more of the first warp is active
    T result = warp_reduction;
    for(unsigned other_warp_index = 1; other_warp_index < warps_per_block;  other_warp_index++)
        result = f(result, warp_reductions[other_warp_index]);
    return result;
}

} // namespace block

void report_dimensions() {
    do_once_per_grid([]() {
        printf("Grid dimensions are %3u x %3u x %3u\n", gridDim.x, gridDim.y, gridDim.z);
        printf("Block dimensions are %3u x %3u x %3u\n", gridDim.x, gridDim.y, gridDim.z);
    });
}

inline bool operator==(dim3 lhs, dim3 rhs) noexcept
{
    return  (lhs.x == rhs.x)
        and (lhs.y == lhs.y)
        and (lhs.z == rhs.z);
}

inline bool operator!=(dim3 lhs, dim3 rhs) noexcept
{
    return (lhs.x != rhs.x)
        or (lhs.y != lhs.y)
        or (lhs.z != rhs.z);
}

#define CONCATENATE(s1, s2) s1##s2
#define EXPAND_THEN_CONCATENATE(s1, s2) CONCATENATE(s1, s2)
#ifdef __COUNTER__
#define UNIQUE_IDENTIFIER(prefix) EXPAND_THEN_CONCATENATE(prefix, __COUNTER__)
#else
#define UNIQUE_IDENTIFIER(prefix) EXPAND_THEN_CONCATENATE(prefix, __LINE__)
#endif // __COUNTER__

#define UNUSED_GLOBAL(g) \
constexpr decltype(g) UNIQUE_IDENTIFIER(g) () { return g; }

#endif // KERNELS_MISCELLANY_CUH_
