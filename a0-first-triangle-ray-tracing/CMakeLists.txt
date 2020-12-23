cmake_minimum_required(VERSION 3.16)

project(D3D12_A0 VERSION 1.0 LANGUAGES CXX)

add_executable(D3D12_A0 WIN32 main.cpp d3dx12.h pch.h debug.h shader.h)
add_definitions(-DUNICODE -D_UNICODE)

target_precompile_headers(D3D12_A0 PRIVATE pch.h)
