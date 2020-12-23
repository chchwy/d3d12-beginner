cmake_minimum_required(VERSION 3.16)

project(D3D12_00 VERSION 1.0 LANGUAGES CXX)

add_definitions(-DUNICODE -D_UNICODE)
add_executable(D1212_00 WIN32 main.cpp pch.h)

target_precompile_headers(D1212_00 PRIVATE pch.h)

