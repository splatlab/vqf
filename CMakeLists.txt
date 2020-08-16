project("ququ")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED TRUE)

include_directories(include pmdk/src/include .)

add_executable(main src/main.cc src/ququ_filter.cc)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fgnu-tm -mavx512bw -mavx512f -frename-registers -march=native")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fgnu-tm -mavx512bw -mavx512f -frename-registers -march=native")

target_link_libraries(main pthread ssl crypto m itm pmemobj pmem)

