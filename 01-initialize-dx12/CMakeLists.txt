cmake_minimum_required(VERSION 3.16)

project(D3D12_01 VERSION 1.0 LANGUAGES CXX)

add_definitions(-DUNICODE -D_UNICODE)
add_executable(D3D12_01 WIN32 main.cpp d3dx12.h pch.h debug.h)

target_precompile_headers(D3D12_01 PRIVATE pch.h)