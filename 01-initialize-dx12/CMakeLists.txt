cmake_minimum_required(VERSION 3.16)

project(main) # set the project name
add_executable(main main.cpp pch.h) # add the executable

target_precompile_headers(main PRIVATE pch.h)