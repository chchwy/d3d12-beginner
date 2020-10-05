
#pragma once
#include <sstream>

std::string exceptionMessage(HRESULT hr)
{
    const std::string message = std::system_category().message(hr);

    std::stringstream sout;
    sout << "==========================================\n";
    sout << "[HRESULT] " << std::hex << hr << std::endl;
    sout << "[ERROR] " << message << std::endl;
    sout << "==========================================\n";
    std::string s = sout.str();

    OutputDebugStringA(s.c_str());
    return s;
}

FORCEINLINE void HR(HRESULT hr)
{
    if (FAILED(hr))
    {
        std::string message = exceptionMessage(hr);
        throw std::exception(message.c_str());
    }
}