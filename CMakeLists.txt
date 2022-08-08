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
find_package(cuda-api-wrappers 0.5.3 REQUIRED)
find_package(Python REQUIRED)

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
	src/util/buffer_io.cpp
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

add_library(kernel-adapters OBJECT)
target_link_libraries(kernel-runner PRIVATE kernel-adapters)
set(adapter-definition-dirs src/kernel_adapters)
list(APPEND adapter-definition-dirs ${EXTRA_ADAPTER_SOURCE_DIRS})
foreach(adapter-source_dir ${adapter-definition-dirs})
	message(STATUS "Using adapters from directory ${adapter-source_dir}")
	file(GLOB_RECURSE adapter-definitions "${adapter-source_dir}/*.hpp")
	file(GLOB_RECURSE adapter-sources "${adapter-source_dir}/*.cpp")
	foreach(adapter-definition ${adapter-definitions})
		target_sources(kernel-adapters PRIVATE "${adapter-definition}")
	endforeach()
	foreach(adapter-source ${adapter-sources})
		target_sources(kernel-adapters PRIVATE "${adapter-source}")
	endforeach()
	list(LENGTH adapter-sources num-sources-in-dir)
	math(EXPR num-adapters-with-source "${num-adapters-with-source} + ${num-sources-in-dir}" OUTPUT_FORMAT DECIMAL)
	set(adapter-definitions-with-sources ${adapter-sources})
	list(TRANSFORM adapter-definitions-with-sources REPLACE "\\.cpp\$" ".hpp")
	list(REMOVE_ITEM adapter-definitions ${adapter-definitions-with-sources})
	list(LENGTH adapter-definitions num-generated-in-dir)
	math(EXPR num-generated-adapters "${num-generated-adapters} + ${num-generated-in-dir}" OUTPUT_FORMAT DECIMAL)
	foreach(adapter-definition ${adapter-definitions})
		file(READ ${adapter-definition} adapter-definition-contents)
		if(NOT "${adapter-definition-contents}" MATCHES "[ \t]*(struct|class)[ \t]+.*kernel_adapter.*{")
			math(EXPR num-generated-adapters "${num-generated-adapters} - 1" OUTPUT_FORMAT DECIMAL)
			continue()
		endif()
		get_filename_component(adapter-name ${adapter-definition} NAME_WE)
		set(adapter-source "src/${adapter-name}.cpp") # Note this will be generated within the build directory
		if (Python_FOUND)
			add_custom_command(
				OUTPUT ${adapter-source}
				MAIN_DEPENDENCY ${adapter-definition}
				COMMAND Python::Interpreter ${PROJECT_SOURCE_DIR}/scripts/generate-adapter-cpp.py "${adapter-definition}" "${adapter-source}"
				COMMENT "Generating source file for adapter ${adapter-name}"
				VERBATIM
			)
		else()
			add_custom_command(
				OUTPUT ${adapter-source}
				MAIN_DEPENDENCY ${adapter-definition}
				COMMAND ${PROJECT_SOURCE_DIR}/scripts/generate-adapter-cpp.sh "${adapter-definition}" "${adapter-source}"
				COMMENT "Generating source file for adapter ${adapter-name}"
				VERBATIM
			)
		endif()

		target_sources(kernel-adapters PRIVATE "${adapter-source}")
	endforeach()
endforeach()

math(EXPR num-adapters "${num-adapters-with-source} + ${num-generated-adapters}")
if (num-adapters EQUAL 0)
	message(FATAL_ERROR "No kernel adapters found")
endif()
message(STATUS "Adapter instantiation sources: ${num-adapters-with-source} pre-existing, ${num-generated-adapters} to be auto-generated")

# Note: There's a bit of redundancy in the following loop;
# but it makes more sense if you want to separate the kernel
# adapters into a library and have it depend on less stuff.

set(compiled-targets kernel-adapters kernel-runner)

foreach(tgt ${compiled-targets})
	set_target_properties(
		${tgt}
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
	target_include_directories(${tgt} SYSTEM PRIVATE third-party)
	target_include_directories(${tgt} PRIVATE src src/cuda-samples-helpers)
	target_warning_options(${tgt})
	target_compile_definitions(${tgt} PRIVATE CUDA_INCLUDE_DIR="${CUDAToolkit_INCLUDE_DIRS}")
	target_link_libraries(
		${tgt}
		PRIVATE
		spdlog::spdlog
		cuda-api-wrappers::runtime-and-driver
		CUDA::OpenCL
		)
endforeach(tgt)

target_link_libraries(
	kernel-runner
	PRIVATE
	spdlog::spdlog
	stdc++fs # For std::filesystem / std::experimental::filesystem
	CUDA::cudart
	CUDA::cuda_driver
	CUDA::nvrtc
	cuda-api-wrappers::runtime-and-driver
	CUDA::OpenCL
	)


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


