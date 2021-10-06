cmake_minimum_required(VERSION 3.16)

project(D3D12_03)

add_definitions(-DUNICODE -D_UNICODE)
add_executable(D3D12_03 WIN32 "main.cpp" "d3dx12.h" "pch.h" "debug.h" "timer.h")

target_precompile_headers(D3D12_03 PRIVATE pch.h)

add_custom_target(
    copy_shader_files
    ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/shaders.hlsl ${CMAKE_BINARY_DIR}
)
add_dependencies(D3D12_03 copy_shader_files)
