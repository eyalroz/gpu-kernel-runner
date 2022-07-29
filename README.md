# GPU Kernel Runner

A harness for stand-alone execution of single GPU kernels, for timing, debugging and profiling.

<br>

| Table of contents|
|:----------------|
| [Example: Executing a simple kernel to get its output](#example) <br> [Motivation](#motivation)<br>[Command-line interface](#cmdline) <br> [Feedback, bugs, questions etc.](#feedback)|

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
* `input_b`, containing 3 octets, each with values `003`.

Now, if you run:
```
kernel-runner --cuda \
    --kernel-key bundled_with_runner/vector_add \
    --kernel-source vector_add.cu \
    --block-dimensions=256,1,1 \
    --grid-dimensions=1,1,1 \
    -A input_a -B input_b -D A_LITTLE_EXTRA=2
```
then you'll get a file named `C.out`, containing `fgh`. And that is, indeed, `abc`, plus 3 for each letter, from the values of `input_B`, plus 2 for each letter from the preprocessor definition. 

You can also do the same with an OpenCL kernel (but specify `--opencl` instead of `--cuda`). Such a kernel is bundled with this repository, so do try it.

There is a bit of "cheating" here: The kernel runner doesn't magically parse your kernel source to determine what parameters you need. You need to have added some boilerplate code for your kernel into the runner:  Listing the kernel name, parameter names, whether they're input or output etc.


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
  ./kernel-runner [OPTION...]

  -l, --log-level arg           Set logging level (default: warning)
      --log-flush-threshold arg
                                Set the threshold level at and above which
                                the log is flushed on each message (default:
                                info)
  -w, --write-output            Write output buffers to files (default: true)
  -n, --num-runs arg            Number of times to run the compiled kernel
                                (default: 1)
      --opencl                  Use OpenCL
      --cuda                    Use CUDA
  -p, --platform-id arg         Use the OpenCL platform with the specified
                                index
  -d, --device arg              Device index (default: 0)
  -D, --define arg              Set a preprocessor definition for NVRTC (can
                                be used repeatedly; specify either DEFINITION
                                or DEFINITION=VALUE)
  -c, --compile-only            Compile the kernel, but don't actually run it
  -G, --debug-mode              Have the NVRTC compile the kernel in debug
                                mode (no optimizations)
  -P, --write-ptx               Write the intermediate representation code
                                (PTX) resulting from the kernel compilation, to
                                a file
      --ptx-output-file arg     File to which to write the kernel's
                                intermediate representation
      --print-compilation-log   Print the compilation log to the standard
                                output
      --write-compilation-log   Write the compilation log to a file
      --compilation-log-file arg
                                Save the compilation log to the specified
                                file (regardless of whether it's printed)
      --generate-line-info      Add source line information to the
                                intermediate representation code (PTX) (default: true)
  -b, --block-dimensions arg    Set grid block dimensions in threads
                                (OpenCL: local work size); a comma-separated list
  -g, --grid-dimensions arg     Set grid dimensions in blocks; a
                                comma-separated list
  -o, --overall-grid-dimensions arg
                                Set grid dimensions in threads (OpenCL:
                                global work size); a comma-separated list
  -O, --append-compilation-option
                                Append an arbitrary extra compilation option
  -S, --dynamic-shared-memory-size arg
                                Force specific amount of dynamic shared
                                memory
  -W, --overwrite               Overwrite the files for buffer and/or PTX
                                output if they already exists
  -i, --include arg             Include a specific file into the kernels'
                                translation unit
  -I, --include-path arg        Add a directory to the search paths for
                                header files included by the kernel (can be used
                                repeatedly)
  -s, --kernel-source arg       Path to CUDA source file with the kernel
                                function to compile; may be absolute or relative
                                to the sources dir
  -k, --kernel-function arg     Name of function within the source file to
                                compile and run as a kernel (if different than
                                the key)
  -K, --kernel-key arg          The key identifying the kernel among all
                                registered runnable kernels
  -L, --list-kernels            List the (keys of the) kernels which may be
                                run with this program
  -z, --zero-output-buffers     Set the contents of output(-only) buffers to
                                all-zeros
  -t, --time-execution          Use CUDA/OpenCL events to time the execution
                                of each run of the kernel
      --language-standard arg   Set the language standard to use for CUDA
                                compilation (options: c++11, c++14, c++17)
      --input-buffer-dir arg    Base location for locating input buffers
                                (default: $PWD)
      --output-buffer-dir arg   Base location for writing output buffers
                                (default: $PWD)
      --kernel-sources-dir arg  Base location for locating kernel source
                                files (default: $PWD)
  -h, --help                    Print usage information
```
Additionally, for a given kernel, you can specify its parameters. For example, if the kernel's signature is `__global__ foo(int bar, float* baz)`, you can also specify:
```
  --foo arg  Description of foo
  --bar arg  Description of bar (default: foo)
```
(the buffer `bar` will, by default, be loaded from the file named `bar` in the present working directory.) Scalar parameters do not typically have defaults.


## <a name="feedback"> Feedback, bugs, questions etc.

* If you use this kernel runner in an interesting project, consider [dropping me a line](mailto:eyalroz1@gmx.com) and telling me about it - both the positive and any negative part of your experience.
* Found a bug? A function/feature that's missing? A poor choice of design or of wording?-Please file an [issue](https://github.com/eyalroz/gpu-kernel-runner/issues/).
* Have a question? If you believe it's generally relevant, also [file an issue](https://github.com/eyalroz/gpu-kernel-runner/issues/), and clearly state that it's a question.
* Want to suggest significant additional functionality, which you believe would be of general interest? Either file an [issue](https://github.com/eyalroz/gpu-kernel-runner/issues/) or [write me](mailto:eyalroz1@gmx.com).

