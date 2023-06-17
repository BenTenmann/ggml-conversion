cmake_minimum_required(VERSION 3.24)

set(PROJECT_NAME torch_jit)
project(${PROJECT_NAME})

include_directories(ggml/include)

set(CMAKE_C_STANDARD   11)
set(CMAKE_CXX_STANDARD 11)

add_subdirectory(ggml)

add_executable(${PROJECT_NAME} main.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ggml/include ggml/examples)
target_link_libraries(${PROJECT_NAME} PRIVATE ggml)
