#!/bin/bash
#
# Takes the header file defining a kernel_adapter subclass, and
# generates a source file which, at static-initialization time,
# registers the class in the kernel-runner's kernel adapter factory
# singleton (so it can then be instantiated by code which has not
# compiled against the adapter subclass header).

function die {
        echo $1 >&2   # error message to stderr
        exit ${2:-1}  # default exit code is -1 but you can specify something else
}

(( $# >= 1 && $# <= 2 )) || die "Usage: $0 KERNEL_ADAPTER_HEADER [ OUTPUT_SOURCE_FILE ]"

kernel_header_path="$1"
[[ -r "$kernel_header_path" ]] || die "Can't read header file: ${kernel_header_path}"
if (( $# == 2 )); then
	mkdir -p "$(dirname $2)"
	exec 1>"$2"
fi
kernel_header_filename="${kernel_header_path##*/}"
echo -e "\n// Auto-generated from ${kernel_header_filename}\n\n#include \"$kernel_header_path\"\n"
egrep  '^\s*(namespace|}\s*//\s*namespace|(class|struct)\s*[^ ]*)' "$kernel_header_path" \
	| sed -r 's/(class|struct)\s+([^ ]*)\s+.*$/static_block {\n    register_in_factory<\2>();\n}/;'
