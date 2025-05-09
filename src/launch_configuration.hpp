#ifndef LAUNCH_CONFIGURATION_HPP_
#define LAUNCH_CONFIGURATION_HPP_

#include "util/filesystem.hpp"
#include "util/optional_and_any.hpp"
#include "util/miscellany.hpp"
#include "util/warning_suppression.hpp"

#include <cuda/api.hpp>

#include "opencl-related/types.hpp"

#include <string>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <memory>
#include <utility>
#include <tuple>

struct optional_launch_config_components_t;

union launch_configuration_type {
    cuda::launch_configuration_t cuda;
    raw_opencl_launch_config opencl;

    launch_configuration_type()  { }
//    launch_configuration_type() { std::memset(this, 0, sizeof(*this)); }
DISABLE_WARNING_PUSH
DISABLE_WARNING_CLASS_MEMACCESS
    launch_configuration_type(const launch_configuration_type& other) { std::memcpy(this, &other, sizeof(other)); }
DISABLE_WARNING_POP

    launch_configuration_type(const raw_opencl_launch_config& config) : opencl(config) { }
    launch_configuration_type(const cuda::launch_configuration_t& config) : cuda(config) { }
//    launch_configuration_type(const optional_launch_config_components_t& config);

    ~launch_configuration_type()  {};

    launch_configuration_type& operator=(const launch_configuration_type& other) = default;
    launch_configuration_type& operator=(launch_configuration_type&& other) = default;
    launch_configuration_type& operator=(const raw_opencl_launch_config& other) { opencl = other; return *this; }
    launch_configuration_type& operator=(raw_opencl_launch_config&& other) { opencl = other; return *this; }
    launch_configuration_type& operator=(const cuda::launch_configuration_t& other) { cuda = other; return *this; }
    launch_configuration_type& operator=(cuda::launch_configuration_t&& other) { cuda = other; return *this; }


};



// Note: According to this: https://stackoverflow.com/questions/8990454
// floats can be serialized and de-serialized with perfect accuracy if we
// use 9 decimal digits (17 for double values)

using launch_config_dimensions_t = std::array<std::size_t, 3>;

struct optional_launch_config_components_t {
    static constexpr const dim3 maximum_possible_grid_dimensions { 0x7FFFFFFF, 0x1111, 0x1111 }; // = 2^31 - 1, 2^16 - 1, 2^16 - 1
    using dims_type = launch_config_dimensions_t;

    optional<dims_type>   grid_dimensions;
    // We're assuming only one of the following two is set.
    optional<dims_type>   block_dimensions;
    optional<dims_type>   overall_grid_dimensions;

    optional<cuda::memory::shared::size_t> dynamic_shared_memory_size;
    // TODO: Add an optional boolean for cooperativity

    bool all_values_present(execution_ecosystem_t ecosystem) const noexcept {
        return
            (bool)block_dimensions and
            ((bool)grid_dimensions or (bool)overall_grid_dimensions) and
            (ecosystem == execution_ecosystem_t::opencl or (bool)dynamic_shared_memory_size);
    }

    void deduce_overall_dimensions()
    {
        if (overall_grid_dimensions) { return; }
        if (not block_dimensions or not grid_dimensions) {
            throw std::invalid_argument{
                "Launch config components does not have sufficient information "
                "for completing the overall grid dimensions computation"};
        }
        auto& bd = block_dimensions.value();
        auto& gd = grid_dimensions.value();
        overall_grid_dimensions = std::array<std::size_t, 3>{
            bd[0]*gd[0],
            bd[1]*gd[1],
            bd[2]*gd[2]
        }; // Note: Not checking for overflow (but that shouldn't matter)
    }

    bool full_blocks() const
    {
        using util::divides;
        auto& gd = block_dimensions.value();
        auto& ogd = overall_grid_dimensions.value();
        return ( divides(gd[0], ogd[0]) and
                 divides(gd[1], ogd[1]) and
                 divides(gd[2], ogd[2]) );
    }

    void deduce_grid_dimensions()
    {
        if (grid_dimensions) { return; }
        if (not overall_grid_dimensions or not block_dimensions) {
            throw std::invalid_argument{
                "Launch config components does not have sufficient information "
                "for completing the grid dimensions computation"};
        }
        auto& bd = block_dimensions.value();
        auto& ogd = overall_grid_dimensions.value();
        grid_dimensions = std::array<std::size_t, 3>{
            util::div_rounding_up(ogd[0], bd[0]),
            util::div_rounding_up(ogd[1], bd[1]),
            util::div_rounding_up(ogd[2], bd[2]),
        }; // Note: Not checking for overflow (but that shouldn't matter)
    }

    bool is_sufficient() const
    {
        return dynamic_shared_memory_size and block_dimensions and
            (grid_dimensions or overall_grid_dimensions);
    }

    void deduce_missing()
    {
        if (not grid_dimensions) { deduce_grid_dimensions(); }
        else { deduce_overall_dimensions(); }
    }

    void set_block_dims(size_t x, size_t y = 1, size_t z = 1)
    {
        block_dimensions.emplace(std::array<size_t, 3>{x, y, z});
    }

    void set_grid_dims(size_t x, size_t y = 1, size_t z = 1)
    {
        grid_dimensions.emplace(std::array<size_t, 3>{x, y, z});
    }

    void set_overall_dims(size_t x, size_t y = 1, size_t z = 1)
    {
        overall_grid_dimensions.emplace(std::array<size_t, 3>{x, y, z});
    }

    void set_block_dims(dim3 dims)   { set_block_dims  (dims.x, dims.y, dims.z); }
    void set_block_dims(int3 dims)   { set_block_dims  (dims.x, dims.y, dims.z); }
    void set_grid_dims(dim3 dims)    { set_grid_dims   (dims.x, dims.y, dims.z); }
    void set_grid_dims(int3 dims)    { set_grid_dims   (dims.x, dims.y, dims.z); }
    void set_overall_dims(dim3 dims) { set_overall_dims(dims.x, dims.y, dims.z); }

    operator cuda::launch_configuration_t() const noexcept(false) {
        if ((bool)grid_dimensions) {
            auto gd = grid_dimensions.value();
            auto bd = block_dimensions.value();
            return {
                { (cuda::grid::dimension_t) gd[0],
                  (cuda::grid::dimension_t) gd[1],
                  (cuda::grid::dimension_t) gd[2] },
                { (cuda::grid::block_dimension_t) bd[0],
                  (cuda::grid::block_dimension_t) bd[1],
                  (cuda::grid::block_dimension_t) bd[2] },
                dynamic_shared_memory_size.value_or(0)
            };
        }
        else {
            using util::divides;
            using util::div_rounding_up;
            auto ogd = overall_grid_dimensions.value();
            auto bd = block_dimensions.value();
            std::array<std::size_t, 3> gd = {
                div_rounding_up(ogd[0],bd[0]),
                div_rounding_up(ogd[1],bd[1]),
                div_rounding_up(ogd[2],bd[2]) };
            if (gd[0] >= maximum_possible_grid_dimensions.x or
                gd[1] >= maximum_possible_grid_dimensions.y or
                gd[2] >= maximum_possible_grid_dimensions.z )
            {
                throw std::invalid_argument("Grid exceeds the maximum theoretical possible size of a CUDA grid.");
                    // TODO: We should also check the grid dimensions against the device parameters
            }
            return {
                { (cuda::grid::block_dimension_t) gd[0],
                  (cuda::grid::block_dimension_t) gd[1],
                  (cuda::grid::block_dimension_t) gd[2] },
                { (cuda::grid::block_dimension_t) bd[0],
                  (cuda::grid::block_dimension_t) bd[1],
                  (cuda::grid::block_dimension_t) bd[2] },
                dynamic_shared_memory_size.value_or(0)
            };
        }
    }
    operator raw_opencl_launch_config() const noexcept(false) {
        // Note: We assume that excess fields in the dimensions structures
        // are set to 1 rather than 0

        if (dynamic_shared_memory_size.value_or(0) > 0) {
            throw std::runtime_error("Can't force non-argument-specific dynamic shared memory for an OpenCL kernel");
        }
        if(not block_dimensions) {
            throw std::runtime_error("Block dimensions not specified");
        }

        // TODO: Should we ensure the overall dims correspond well to the block dims?
        // Let's assume we shouldn't do that here.

        const auto& bd = block_dimensions.value();
        const auto& gd = grid_dimensions.value();

        return {
            { bd[0], bd[1], bd[2] },
            { gd[0], gd[1], gd[2] }
        };
    }
};

inline bool has_zeros(const optional_launch_config_components_t::dims_type& dims_component) {
    return util::contains(dims_component, 0);
}

inline bool has_zeros(optional<optional_launch_config_components_t::dims_type> maybe_dims) {
    return maybe_dims and has_zeros(maybe_dims.value());
}

/// Was the kernel adapter "lazy", neglecting to initialize one of the dimensions?
/// that dimension may be zero-initialized...
inline bool has_zeros(optional_launch_config_components_t lc_components)
{
    return
        has_zeros(lc_components.block_dimensions) or
        has_zeros(lc_components.grid_dimensions) or
        has_zeros(lc_components.overall_grid_dimensions);
}

inline launch_configuration_type realize_launch_config(
    const optional_launch_config_components_t& components,
    execution_ecosystem_t                    ecosystem)
{
    return (ecosystem == execution_ecosystem_t::cuda) ?
        launch_configuration_type{(cuda::launch_configuration_t) components } :
        launch_configuration_type{ (raw_opencl_launch_config) components };
}

#endif /* LAUNCH_CONFIGURATION_HPP_ */
