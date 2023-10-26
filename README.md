# GPU Kernel Runner

A harness for stand-alone execution of single GPU kernels, for timing, debugging and profiling.

<br>

| Table of contents|
|:----------------|
| <sub>[Example: Executing a simple kernel to get its output](#example) <br> [Motivation](#motivation)<br>[Command-line interface](#cmdline) <br>[How do I get the runner to run my own kernel?](#running-your-own-kernel) <br> [Feedback, bugs, questions etc.](#feedback) </sub>|

----

## <a name="example">Example: Executing a simple kernel to get its output</a>

Consider the following kernel (bundled with this repository):

```
__global__ void vectorAdd(
        unsigned char       * __restrict  C,
        unsigned char const * __restrict  A,
        unsigned char const * __restrict  B,
        size_t length)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < length) {
        C[i] = A[i] + B[i] + A_LITTLE_EXTRA;
    }
}
```
and suppose that you've also created two files: 

* `input_a`, containing the three characters `abc`;
* `input_b`, containing 3 octets, each with values `03`.

Now, if you run:
```
kernel-runner \
    --execution-ecosystem cuda \
    --kernel-key bundled_with_runner/vector_add \
    --kernel-source vector_add.cu \
    --block-dimensions 256,1,1 \
    --grid-dimensions 1,1,1 \
    --arg A=input_a --arg length=3 --arg B=input_b \
    -DA_LITTLE_EXTRA=2
```
then you'll get a file named `C.out`, containing `fgh`... which is indeed the correct output of the kernel: The sequence `abc`, plus 3 for each character due the values in `input_B`, plus 2 for each character from the preprocessor definition of `A_LITTLE_EXTRA`. 

You can do the same with an equivalent OpenCL kernel, also bundled with this repository; just specify `opencl` instead of `cuda` as the execution ecosystem, 
nd use the `vector_add.cl` kernel source file.

There is a bit of "cheating" here: The kernel runner doesn't magically parse your kernel source to determine what arguments are required. You need to have added some boilerplate code for your kernel into the runner:  Listing the kernel name, parameter names, whether they're input or output etc.

## <a name="motivation">Motivation</a>

When we develop GPU kernels, or try to optimize existing ones, they are often intended for the middle of a large application:

* A lot of work (and time) is expended before our kernel of interest gets run
* The kernel is fed inputs - scalar and buffers - which are created as intermediate data of the larger program, and is neither saved to disk nor printed to logs.
* ... alternatively, the kernel may be invoked so many times, that it would not make sense to save or print all of that data.
* The kernel may be compiled dynamically, and the compilation parameters may also not be saved for later scrutiny

This makes the kernel unwieldy in development and testing - if not outright impossible. So, either you live with it, which is difficult and frustrating, or what some of us do occasionally is write a separate program which only runs the kernel, which is a lot of hassle.

This repository is intended to take away all of that hassle away: It has the machinery you need to run _any_ kernel - CUDA or OpenCL - independently. You just need to provide with the kernel's direct inputs and outputs (as buffers); launch grid parameters; and dynamic compilation options (since we have to JIT the kernel).


## <a name="cmdline">Command-line interface</a>

```
A runner for dynamically-compiled CUDA kernels
Usage:
  kernel-runner [OPTION...]

  -l, --log arg                 Set logging level (default: warning)
      --log-flush-threshold arg
                                Set the threshold level at and above which
                                the log is flushed on each message
                                (default: info)
  -w, --save-output             Write output buffers to files (default:
                                true)
  -n, --repetitions arg         Number of times to run the compiled kernel
                                (default: 1)
  -e, --execution-ecosystem arg
      --opencl                  Use OpenCL
      --cuda                    Use CUDA
                                Execution ecosystem (CUDA or Opencl)
  -p, --platform-id arg         Use the OpenCL platform with the specified
                                index
  -a, --argument arg            Set one of the kernel's argument, keyed by
                                name, with a serialized value for a scalar
                                (e.g. foo=123) or a path to the contents of
                                a buffer (e.g. bar=/path/to/data.bin)
  -A, --no-default-compilation-options
                                Avoid setting any compilation options not
                                explicitly requested by the user
      --output-buffer-size arg  Set one of the output buffers' sizes, keyed
                                by name, in bytes (e.g. myresult=1048576)
  -d, --device arg              Device index (default: 0)
  -D, --define arg              Set a preprocessor definition for NVRTC
                                (can be used repeatedly; specify either
                                DEFINITION or DEFINITION=VALUE)
  -c, --compile-only            Compile the kernel, but don't actually run
                                it
  -G, --device-debug-mode       Have the NVRTC compile the kernel in debug
                                mode (no optimizations)
  -P, --write-ptx               Write the intermediate representation code
                                (PTX) resulting from the kernel
                                compilation, to a file
      --ptx-output-file arg     File to which to write the kernel's
                                intermediate representation
      --print-compilation-log   Print the compilation log to the standard
                                output
      --write-compilation-log arg
                                Path of a file into which to write the
                                compilation log (regardless of whether it's
                                printed to standard output) (default: "")
      --print-execution-durations
                                Print the execution duration, in
                                nanoseconds, of each kernel invocation to
                                the standard output
      --write-execution-durations arg
                                Path of a file into which to write the
                                execution durations, in nanoseconds, for
                                each kernel invocation (regardless of
                                whether they're printed to standard output)
                                (default: "")
      --generate-line-info      Add source line information to the
                                intermediate representation code (PTX)
  -b, --block-dimensions arg    Set grid block dimensions in threads
                                (OpenCL: local work size); a
                                comma-separated list
  -g, --grid-dimensions arg     Set grid dimensions in blocks; a
                                comma-separated list
  -o, --overall-grid-dimensions arg
                                Set grid dimensions in threads (OpenCL:
                                global work size); a comma-separated list
  -O, --append-compilation-option arg
                                Append an arbitrary extra compilation
                                option
  -S, --dynamic-shared-memory-size arg
                                Force specific amount of dynamic shared
                                memory
  -W, --overwrite-output-files  Overwrite the files for buffer and/or PTX
                                output if they already exists
  -i, --include arg             Include a specific file into the kernels'
                                translation unit
  -I, --include-path arg        Add a directory to the search paths for
                                header files included by the kernel (can be
                                used repeatedly)
  -s, --source-file arg         Path to CUDA source file with the kernel
                                function to compile; may be absolute or
                                relative to the sources dir
  -k, --kernel-function arg     Name of function within the source file to
                                compile and run as a kernel (if different
                                than the key)
  -K, --kernel-key arg          The key identifying the kernel among all
                                registered runnable kernels
  -L, --list-adapters           List the (keys of the) kernels which may be
                                run with this program
  -z, --zero-outputs            Set the contents of output(-only) buffers
                                to all-zeros
      --language-standard arg   Set the language standard to use for CUDA
                                compilation (options: c++11, c++14, c++17)
      --input-buffer-directory arg
                                Base location for locating input buffers
                                (default:
                                /home/lh156516/src/gpu-kernel-runner)
      --output-buffer-directory arg
                                Base location for writing output buffers
                                (default:
                                /home/lh156516/src/gpu-kernel-runner)
      --kernel-sources-dir arg  Base location for locating kernel source
                                files (default:
                                /home/lh156516/src/gpu-kernel-runner)
  -h, --help                    Print usage information
```

## <a name="running-your-own-kernel">How do I get the runner to run my own kernels?</a>

So, you've written a kernel. In order for the GPU kernel runner to run it, the runner needs to know about it. Internally, the runner knows kernels through "kernel adapter" classes, instantiated into a factory. Luckily, you don't have to be familiar with this mechanism in order to use it. What you _do_ need is a:

1. A kernel adapter class definition, in a header file
2. Building the kernel runner so as to recognize and use your kernel adapter header file

### <a name="build-with-extra-adapters">Telling the build about your adapters</a>

The CMake for this repository has a variables named `EXTRA_ADAPTER_SOURCE_DIRS`- you can set it when invoking CMake to configure your build, e.g.:
```
cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -DEXTRA_ADAPTER_SOURCE_DIRS="/path/to/my_adapters/;/another/path/to/more_adapters" \
    -B /path/to/build_dir/
```
... so that the build configuration can find your adapters and ensure they are instantiated.

### <a name="kernel-adapter-template">A kernel adapter template</a>

To create a kernel adapter for your kernel, it's easiest to start with the following empty template and replace the `[[[ ... ]]]` parts with what's relevant for your own kernel:
```
#include <kernel_adapter.hpp>

class [[[ UNIQUE CLASS NAME HERE ]]] : public kernel_adapter {
public:
    KA_KERNEL_FUNCTION_NAME("[[[ (POSSIBLY-NON-UNIQUE) FUNCTION NAME HERE ]]]")
    KA_KERNEL_KEY("[[[ UNIQUE KERNEL KEY STRING HERE ]]]")

    const parameter_details_type& parameter_details() const override
    {
        static const parameter_details_type pd = {
            [[[ DETAIL LINES FOR EACH KERNEL PARAMETER ]]]
            
            // Example detail lines:
            //
            //  scalar_details<int>("x"),
            //  buffer_details("my_results", output),
            //  buffer_details("my_data", input),
        };
        return pd;
    }
};
```
For a concrete example, see the adapter file [`vector_add.hpp`](https://github.com/eyalroz/gpu-kernel-runner/blob/main/src/kernel_adapters/vector_add.hpp).

The `kernel_adapter` class actually has other methods one could overwrite in order to make the kernel easier to invoke, for:

* Deducing launch grid configuration
* Specifying required preprocessor definitions
* Generation of some arguments from other arguments (e.g. length from buffer size)

but none of these are essential.

## <a name="feedback"> Feedback, bugs, questions etc.

* If you use this kernel runner in an interesting project, consider [dropping me a line](mailto:eyalroz1@gmx.com) and telling me about it - both the positive and any negative part of your experience.
* Found a bug? A function/feature that's missing? A poor choice of design or of wording?-Please file an [issue](https://github.com/eyalroz/gpu-kernel-runner/issues/).
* Have a question? If you believe it's generally relevant, also [file an issue](https://github.com/eyalroz/gpu-kernel-runner/issues/), and clearly state that it's a question.
* Want to suggest significant additional functionality, which you believe would be of general interest? Either file an [issue](https://github.com/eyalroz/gpu-kernel-runner/issues/) or [write me](mailto:eyalroz1@gmx.com).

