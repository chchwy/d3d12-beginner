cmake_minimum_required(VERSION 3.16)

project(main) # set the project name
add_executable(main main.cpp d3dx12.h pch.h debug.h shader.h) # add the executable

target_precompile_headers(main PRIVATE pch.h)