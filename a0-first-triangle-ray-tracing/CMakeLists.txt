cmake_minimum_required(VERSION 3.16)

project(D3D12_A0 VERSION 1.0 LANGUAGES CXX)

add_definitions(-DUNICODE -D_UNICODE)
add_executable(D3D12_A0 WIN32 main.cpp d3dx12.h pch.h debug.h shader.h)

target_precompile_headers(D3D12_A0 PRIVATE pch.h)


FILE(GLOB ShaderFiles CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.hlsl)

add_custom_target(
    copy_shader_files
    ${CMAKE_COMMAND} -E copy ${ShaderFiles} ${CMAKE_BINARY_DIR}
)
add_dependencies(D3D12_A0 copy_shader_files)
