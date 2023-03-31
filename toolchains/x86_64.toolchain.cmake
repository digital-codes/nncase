if(ENABLE_X86SIMD)
    add_definitions(-DX86_64_SIMD_ON)
    if (MSVC)
        add_compile_options(/arch:AVX)
        add_compile_options(/arch:AVX2)
    else()
        add_compile_options(-mfma -msse -msse2 -msse3 -mssse3 -msse4 -msse4a -msse4.1 -msse4.2 -mavx -mavx2)
    endif()
endif()
