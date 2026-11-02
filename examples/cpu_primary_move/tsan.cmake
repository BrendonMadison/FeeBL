set(CMAKE_CXX_FLAGS
    "-O1 -g -fsanitize=thread -fno-omit-frame-pointer"
    CACHE STRING "CPU primary movement TSan flags" FORCE)
set(CMAKE_EXE_LINKER_FLAGS
    "-fsanitize=thread"
    CACHE STRING "CPU primary movement TSan linker flags" FORCE)
