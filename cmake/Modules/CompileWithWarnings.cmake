if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(WARNING_FLAGS "-Wall -Wextra -Wpedantic -Wno-missing-field-initializers")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(WARNING_FLAGS "-Wall -Wextra -Wpedantic -Wno-missing-field-initializers")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(WARNING_FLAGS "-w3 -wd1418,2259")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(WARNING_FLAGS "/W4")
else ()
    message(WARNING "Unknown compiler - cannot set warning flags")
endif()

if(WARNING_FLAGS)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS}")
endif()

macro(target_warning_options tgt)
target_compile_options(tgt PRIVATE
     $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
          -Wall -Wextra -pedantic -Wshadow -Wdouble-promotion -Wformat=2 -Wsign-conversion -Wformat-overflow -Wformat-truncation -Wundef -fno-common -Wconversion -Wextra>
           # Can add more errors here, I guess
     $<$<CXX_COMPILER_ID:MSVC>:
          /W4>
     $<$<CXX_COMPILER_ID:Intel>:
          -w3 -wd1418,2259>
)
endmacro()

