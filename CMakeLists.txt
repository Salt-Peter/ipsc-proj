cmake_minimum_required(VERSION 3.13)
project(ipsc_proj)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)

add_executable(ipsc_proj
        src/main.cpp)
