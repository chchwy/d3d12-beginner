CONFIG -= qt

TEMPLATE = app
TARGET = DX12_a0_First_Triangle_RayTracing

HEADERS = \
	d3dx12.h \
	debug.h \
	nv_helpers_dx12/TopLevelASGenerator.h \
	nv_helpers_dx12/BottomLevelASGenerator.h

SOURCES = \
	main.cpp \
	nv_helpers_dx12/TopLevelASGenerator.cpp \
	nv_helpers_dx12/BottomLevelASGenerator.cpp

PRECOMPILED_HEADER = pch.h