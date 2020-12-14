cmake_minimum_required(VERSION 3.16)

project(a0) # set the project name
add_executable(a0 WIN32 main.cpp d3dx12.h pch.h debug.h shader.h)
target_precompile_headers(a0 PRIVATE pch.h)
add_definitions(-DUNICODE -D_UNICODE)