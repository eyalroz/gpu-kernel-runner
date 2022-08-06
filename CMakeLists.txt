cmake_minimum_required(VERSION 3.19)
project(gpu-kernel-runner
	DESCRIPTION "Dynamically-compiled GPU kernel runner"
	VERSION 0.1.0
	HOMEPAGE_URL "https://github.com/eyalroz/gpu-kernel-runner"
	LANGUAGES CXX CUDA)

###############
##  Modules  ##
###############

# Standard CMake modules
# (none)
# Custom modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

include(CompileWithWarnings)

############################
##  Package dependencies  ##
############################

# There seem to be some cl.hpp changes in some CUDA 11 versions;
# need to test with that before supporting it. 10.2 has not been tested but should hopefully be fine
find_package(CUDAToolkit 10.1 REQUIRED)
find_package(spdlog 1.6.0 REQUIRED)
#find_package(cuda-api-wrappers 0.3.3 REQUIRED)
find_package(cuda-api-wrappers 0.5.3 REQUIRED)

###############
##  OPTIONS  ##
###############

set(EXTRA_ADAPTER_SOURCE_DIRS CACHE STRING "A semicolon-separated list of directories of additional self-registering kernel adapter .cpp files")

#message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

###############
##  Targets  ##
###############

add_executable(kernel-runner
	src/kernel-runner.cpp
	src/buffer_io.cpp
	src/util/cxxopts-extra.hpp
	src/util/optional_and_any.hpp
	src/nvrtc-related/execution.hpp
	src/nvrtc-related/build.hpp
	src/nvrtc-related/standard_header_substitutes.hpp
	src/opencl-related/build.hpp
	src/opencl-related/execution.hpp
	src/opencl-related/miscellany.hpp
	src/opencl-related/types.hpp
	src/opencl-related/ugly_error_handling.hpp
	src/util/cxxopts-extra.hpp
	src/util/factory_producible.hpp
	src/util/filesystem.hpp
	src/util/functional.hpp
	src/util/miscellany.hpp
	src/util/optional_and_any.hpp
	src/util/spdlog-extra.hpp
	src/util/static_block.hpp
	src/parsers.hpp)

set(adapter_source_dirs src/kernel_adapters)
list(APPEND adapter_source_dirs ${EXTRA_ADAPTER_SOURCE_DIRS})
foreach(adapter_source_dir ${adapter_source_dirs})
	message(STATUS "Using adapters from directory ${adapter_source_dir}")
	file(GLOB_RECURSE adapter_sources "${adapter_source_dir}/*.cpp")
#	foreach(adapter_source ${adapter_sources})
#		message(STATUS "Building adapter ${adapter_source}")
#	endforeach()
	target_sources(kernel-runner PRIVATE ${adapter_sources})
endforeach()

# Note: There's a bit of redundancy in the following loop;
# but it makes more sense if you want to separate the kernel
# adapters into a library and have it depend on less stuff.

set(compiled-targets
	kernel-runner
)

foreach(TARGET ${compiled-targets})
	set_target_properties(
		${TARGET}
		PROPERTIES
		CXX_STANDARD 14
			# Not C++17, because we want to be SLES 15-compatible :-(
			# ... but note that our CUDA kernel _can_ theoretically
			# be C++17 (we'll just have to be careful when passing
			# instances of C++17-modified classes to kernels.
		CXX_STANDARD_REQUIRED YES
		CXX_EXTENSIONS NO
	    CUDA_STANDARD 14
	    CUDA_STANDARD_REQUIRED O
	)
	# Give some special treatment to libraries we didn't write
	target_include_directories(${TARGET} SYSTEM PRIVATE third-party)
	target_include_directories(${TARGET} PRIVATE src src/cuda-samples-helpers)
	target_warning_options(${TARGET})
endforeach(TARGET)

target_link_libraries(
	kernel-runner
	PRIVATE
	spdlog::spdlog
	stdc++fs # For std::filesystem
	CUDA::cudart
	CUDA::cuda_driver
	CUDA::nvrtc
	${CUDA_LIBRARIES}
	cuda-api-wrappers::runtime-and-driver
	CUDA::OpenCL
	)

target_compile_definitions(kernel-runner PRIVATE CUDA_INCLUDE_DIR="${CUDAToolkit_INCLUDE_DIRS}")

####################
##  Installation  ##
####################

# (not supported for now)
#
#include(GNUInstallDirs)

#install(
#	TARGETS kernel-runner
#	EXPORT kernel-runner_export
#	INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
#)

#install(
#	DIRECTORY src/kat
#	DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
#	FILES_MATCHING REGEX "\\.(h|hpp|cuh)$"
#)

#install(
#	EXPORT kernel-runner_export
#	DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/kernel-runner"
#	NAMESPACE "kernel-runner::"
#	FILE kernel-runner-config.cmake
#)

#include(CMakePackageConfigHelpers)

#write_basic_package_version_file(
#	"kernel-runner-config-version.cmake"
#	VERSION ${PROJECT_VERSION}
#	COMPATIBILITY SameMinorVersion
#)

#install(
#  FILES "${CMAKE_CURRENT_BINARY_DIR}/kernel-runner-config-version.cmake"
#  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/kernel-runner"
#)


