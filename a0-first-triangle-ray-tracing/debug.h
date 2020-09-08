
#pragma once
#include <sstream>

#define HR(h) MyThrowIfFailed((h), __FILE__, __LINE__, __FUNCTION__);

std::string exceptionMessage(HRESULT hr, const char* file, const int line, const char* func)
{
    std::string message = std::system_category().message(hr);

    std::stringstream sout;
    sout << "==========================================\n";
    sout << "[HRESULT] " << std::hex << hr << std::endl;
    sout << "[FILE] " << file << std::endl;
    sout << "[LINE] " << line << std::endl;
    sout << "[FUNC] " << func << std::endl;
    sout << "[ERROR] " << message << std::endl;
    sout << "==========================================\n";
    std::string s = sout.str();

    OutputDebugStringA(s.c_str());
    return s;
}

inline void MyThrowIfFailed(HRESULT hr, const char* file, const int line, const char* func)
{
    if (FAILED(hr))
    {
        std::string message = exceptionMessage(hr, file, line, func);
        throw std::exception(message.c_str());
    }
}