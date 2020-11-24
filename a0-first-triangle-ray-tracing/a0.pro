CONFIG -= qt

TEMPLATE = app
TARGET = a0-first-triangle-ray-tracing

HEADERS = \
	d3dx12.h \
	debug.h \
	shader.h \

SOURCES = \
	main.cpp \

PRECOMPILED_HEADER = pch.h

OBJECTS_DIR = obj
DESTDIR = bin