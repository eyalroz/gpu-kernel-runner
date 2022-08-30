#!/usr/bin/env python
#
# Takes the header file defining a kernel_adapter subclass, and
# generates a source file which, at static-initialization time,
# registers the class in the kernel-runner's kernel adapter factory
# singleton (so it can then be instantiated by code which has not
# compiled against the adapter subclass header).

import sys;
import re;
import os;

assert len(sys.argv) >= 2, "No input file specified"
assert len(sys.argv) <= 3, "Too many command-line arguments"

if len(sys.argv) == 3:
    output_dir = os.path.dirname(sys.argv[2])
    os.makedirs(output_dir, mode=0o755, exist_ok=True)
    sys.stdout = open(sys.argv[2], 'w')

print("#include \"{}\"".format(sys.argv[1]))
for line in open(sys.argv[1], 'r').readlines():
    namespace_delimiter_match = re.match(r'^\s*(namespace)|(}\s*//\s*namespace)', line)
    if (namespace_delimiter_match):
        sys.stdout.write(line)
    adapter_definition_match = re.match(r'^\s*(class|struct)\s*([^ ]+).*public\s+kernel_adapter', line)
    if (adapter_definition_match):
        print(
            "static_block {\n" +
            "    kernel_adapters::register_in_factory<" + adapter_definition_match.group(2) + ">();\n" +
            "}")
        continue
    #
#
