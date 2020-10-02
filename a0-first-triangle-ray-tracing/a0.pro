CONFIG -= qt

TEMPLATE = app
TARGET = a0-first-triangle-ray-tracing

HEADERS = \
	d3dx12.h \
	debug.h \
	shader.h \
	nv_helpers_dx12/ShaderBindingTableGenerator.h

SOURCES = \
	main.cpp \
	nv_helpers_dx12/ShaderBindingTableGenerator.cpp

PRECOMPILED_HEADER = pch.h

OBJECTS_DIR = obj
DESTDIR = bin