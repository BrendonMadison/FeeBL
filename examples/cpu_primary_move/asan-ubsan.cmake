set(CMAKE_CXX_FLAGS
    "-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer"
    CACHE STRING "CPU primary movement ASan/UBSan flags" FORCE)
set(CMAKE_EXE_LINKER_FLAGS
    "-fsanitize=address,undefined"
    CACHE STRING "CPU primary movement ASan/UBSan linker flags" FORCE)
