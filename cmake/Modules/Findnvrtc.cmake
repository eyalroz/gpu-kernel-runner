# Copyright (c)    2015 Patrick Diehl
# Copyright (c)    2020-2022 GE Healthcare Inc.
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt
##############################################################################
# - Try to find the Cuda NVRTC library
# Once done this will define
# nvrtc_FOUND - System has the NVRTC library
# LIBNVRTC_LIBRARY_DIR - The NVRTC library dir
# CUDA_NVRTC_LIBRARY - The NVRTC lib
##############################################################################
find_package(PkgConfig)

find_library(NVRTC_LIBRARY 
	NAMES libnvrtc nvrtc 
	PATHS "${CUDA_TOOLKIT_ROOT_DIR}" "${LIBNVRTC_LIBRARY_DIR}"
	PATH_SUFFIXES "lib64" "common/lib64" "common/lib" "lib"
	HINTS 
#	/usr/lib64 
#	/usr/local/cuda/lib64
	DOC "Location of the NVIDIA CUDA Run-Time Compilation (NVRTC) library"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(nvrtc
	REQUIRED_VARS NVRTC_LIBRARY # NVRTC_INCLUDE_DIR
	HANDLE_COMPONENTS
# VERSION_VAR ???
)

mark_as_advanced(CUDA_NVRTC_LIBRARY)

if(nvrtc_FOUND AND NOT TARGET nvrtc::nvrtc)
	add_library(nvrtc::nvrtc INTERFACE IMPORTED)
	set_property(TARGET nvrtc::nvrtc PROPERTY INTERFACE_LINK_LIBRARIES "${NVRTC_LIBRARY}")
endif()
