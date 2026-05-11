cmake_minimum_required(VERSION 3.19)
project(gpu-kernel-runner
	DESCRIPTION "Dynamically-compiled GPU kernel runner"
	VERSION 0.1.0
	HOMEPAGE_URL "https://github.com/eyalroz/gpu-kernel-runner"
	LANGUAGES CXX)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
endif()

###############
##  Modules  ##
###############

# Standard CMake modules
# (none)
# Custom modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

include(CompileWithWarnings)

function(poor_mans_remove_prefix str prefix_length out_var)
  # We assume all parameters are valid, and that length is non-negative and doesn't exceed the sring length
  string(LENGTH "${str}" __str_len)
  math(EXPR __remaining_len "${__str_len} - ${prefix_length}")
  # Extract substring from index = length, length = remaining
  string(SUBSTRING "${str}" ${prefix_length} ${__remaining_len} __result)
  set(${out_var} "${__result}" PARENT_SCOPE)
endfunction()

############################
##  Package dependencies  ##
############################

# There seem to be some cl.hpp changes in some CUDA 11 versions;
# need to test with that before supporting it. 10.2 has not been tested but should hopefully be fine
find_package(CUDAToolkit 10.1 REQUIRED)
find_package(spdlog 1.17.0 REQUIRED)
find_package(cuda-api-wrappers 0.8.1 REQUIRED)
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
	src/kernel_adapter.cpp
	src/buffer_ops.cpp
	src/cmdline_arguments.cpp
	src/util/filesystem.cpp
	src/util/buffer_io.cpp
	src/opencl-related/build.cpp
	src/opencl-related/execution.cpp
	src/opencl-related/ugly_error_handling.cpp
	src/opencl-related/miscellany.cpp
	src/nvrtc-related/build.cpp
	src/nvrtc-related/execution.cpp
	src/nvrtc-related/standard_header_substitutes.cpp
	src/nvrtc-related/miscellany.cpp
	)

add_library(kernel-adapters OBJECT)
target_link_libraries(kernel-runner PRIVATE kernel-adapters)
set(adapter-definition-root-dirs src/kernel_adapters)
list(APPEND adapter-definition-root-dirs ${EXTRA_ADAPTER_SOURCE_DIRS})
foreach(adapter-definitions-base-dir ${adapter-definition-root-dirs})
	cmake_path(ABSOLUTE_PATH adapter-definitions-base-dir  NORMALIZE)
	cmake_path(GET adapter-definitions-base-dir ROOT_NAME root-to-strip) 
	string (LENGTH "${root-to-strip}" root-len)
	message(STATUS "Using adapters defined under ${adapter-definitions-base-dir}")
	file(GLOB_RECURSE adapter-definitions "${adapter-definitions-base-dir}/*.hpp")
	file(GLOB_RECURSE adapter-sources "${adapter-definitions-base-dir}/*.cpp")
	foreach(adapter-definition ${adapter-definitions})
		target_sources(kernel-adapters PRIVATE "${adapter-definition}")
	endforeach()
	foreach(adapter-source ${adapter-sources})
		target_sources(kernel-adapters PRIVATE "${adapter-source}")
	endforeach()
	list(LENGTH adapter-sources num-sources-in-dir)
	math(EXPR num-adapters-with-source "${num-adapters-with-source} + ${num-sources-in-dir}" OUTPUT_FORMAT DECIMAL)
	list(TRANSFORM adapter-sources REPLACE "\\.cpp\$" ".hpp" OUTPUT_VARIABLE adapter-definitions-with-sources)
	set(adapter-definitions-requiring-generation ${adapter-definitions}) # ... this will make sense after the next line
	list(REMOVE_ITEM adapter-definitions-requiring-generation ${adapter-definitions-with-sources})
	list(LENGTH adapter-definitions-requiring-generation num-generated-in-dir)
	math(EXPR num-generated-adapters "${num-generated-adapters} + ${num-generated-in-dir}" OUTPUT_FORMAT DECIMAL)
	foreach(adapter-definition ${adapter-definitions-requiring-generation})
		file(READ ${adapter-definition} adapter-definition-contents)
		if(NOT "${adapter-definition-contents}" MATCHES "[ \t]*(struct|class)[ \t]+.*kernel_adapter.*{")
			math(EXPR num-generated-adapters "${num-generated-adapters} - 1" OUTPUT_FORMAT DECIMAL)
			continue()
		endif()
		get_filename_component(adapter-name ${adapter-definition} NAME_WE)
		get_filename_component(adapter-definition-dir_ ${adapter-definition} DIRECTORY)
		cmake_path(NORMAL_PATH adapter-definition-dir_)
		poor_mans_remove_prefix("${adapter-definition-dir_}" "${root-len}" adapter-definition-dir)
		cmake_path(SET adapter-source "src/${adapter-definition-dir}/${adapter-name}.cpp" NORMALIZE)
		# ... but note that this source file will be generated under the build directory
		if (Python_FOUND)
			add_custom_command(
				OUTPUT ${adapter-source}
				MAIN_DEPENDENCY ${adapter-definition}
				DEPENDS ${PROJECT_SOURCE_DIR}/scripts/generate-adapter-cpp.py
				COMMAND Python::Interpreter ${PROJECT_SOURCE_DIR}/scripts/generate-adapter-cpp.py "${adapter-definition}" "${adapter-source}"
				COMMENT "Generating source file for adapter ${adapter-name}"
				VERBATIM
			)
		else()
			add_custom_command(
				OUTPUT ${adapter-source}
				MAIN_DEPENDENCY ${adapter-definition}
				DEPENDS ${PROJECT_SOURCE_DIR}/scripts/generate-adapter-cpp.sh
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
	if (MSVC)
		set_target_properties(${tgt} PROPERTIES CXX_STANDARD 17)
		target_compile_definitions(${tgt} PRIVATE
			"_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING"
			"_CRT_SECURE_NO_WARNINGS"
			"_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING"
			)
		target_compile_options(${tgt} PRIVATE
			"/Zc:__cplusplus" # ensure the __cplusplus macro is defined
			"/permissive-" # Make MSVC's preprocessor behave more properly
			"/utf-8" # Interpret text as UTF-8
			"/EHs" # Exception stack unwinding
		)
		target_link_options(${tgt} PUBLIC "/NODEFAULTLIB:LIBCMT" "/NODEFAULTLIB:LIBCMT.lib")
	endif()
	# Give some special treatment to libraries we didn't write
	target_include_directories(${tgt} SYSTEM PRIVATE third-party)
	target_include_directories(${tgt} PRIVATE src)
	target_warning_options(${tgt})
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
	CUDA::cudart
	CUDA::cuda_driver
	CUDA::nvrtc
	cuda-api-wrappers::runtime-and-driver
	CUDA::OpenCL
	)

if (NOT MSVC)
	target_link_libraries(kernel-runner PRIVATE stdc++fs)
endif()

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


