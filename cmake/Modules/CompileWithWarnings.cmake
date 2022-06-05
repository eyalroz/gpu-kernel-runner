macro(target_warning_options the_tgt)
	target_compile_options(${the_tgt} PRIVATE
		$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
			-Wall -Wextra -pedantic -Wshadow -Wdouble-promotion -Wformat=2 -Wconversion -Wno-missing-field-initializers>
        $<$<CXX_COMPILER_ID:Clang>: -Wno-c++17-extensions>
		$<$<CXX_COMPILER_ID:GNU>:
			-Wformat-overflow -Wformat-truncation -Wconversion -Wextra -Wno-missing-field-initializers>
		$<$<CXX_COMPILER_ID:MSVC>: /W4>
		$<$<CXX_COMPILER_ID:Intel>: -w3 -wd1418,2259>
)
# For clang, we would have liked to use -Wno-c++17-extensions, but that probably requires a newer version
endmacro()

