
#include <fstream>

// Compile a HLSL file into a DXIL library

IDxcBlob* CompileShaderLibrary2(LPCWSTR fileName)
{
    static IDxcCompiler* dxcCompiler = nullptr;
    static IDxcLibrary* dxcLibary = nullptr;
    static IDxcIncludeHandler* dxcIncludeHandler;

    HRESULT hr;

    // Initialize the DXC compiler and compiler helper
    if (!dxcCompiler)
    {
        HR(DxcCreateInstance(CLSID_DxcCompiler, __uuidof(IDxcCompiler), (void**)&dxcCompiler));
        HR(DxcCreateInstance(CLSID_DxcLibrary, __uuidof(IDxcLibrary), (void**)&dxcLibary));
        HR(dxcLibary->CreateIncludeHandler(&dxcIncludeHandler));
    }

    // Open and read the file
    std::ifstream shaderFile(fileName);
    if (!shaderFile.good())
    {
        throw std::logic_error("Cannot find shader file");
    }
    std::stringstream strStream;
    strStream << shaderFile.rdbuf();
    std::string sShader = strStream.str();

    // Create blob from the string
    IDxcBlobEncoding* pTextBlob = nullptr;
    HR(dxcLibary->CreateBlobWithEncodingFromPinned((LPBYTE)sShader.c_str(), (uint32_t)sShader.size(), 0, &pTextBlob));

    // Compile
    IDxcOperationResult* pResult = nullptr;
    HR(dxcCompiler->Compile(pTextBlob,
                            fileName, 
                            L"",
                            L"lib_6_3",
                            nullptr, 0, nullptr, 0,
                            dxcIncludeHandler,
                            &pResult));

    // Verify the result

    HR(pResult->GetStatus(&hr));
    if (FAILED(hr))
    {
        IDxcBlobEncoding* error;
        hr = pResult->GetErrorBuffer(&error);
        if (FAILED(hr))
        {
            throw std::logic_error("Failed to get shader compiler error");
        }

        // Convert error blob to a string
        std::vector<char> infoLog(error->GetBufferSize() + 1);
        memcpy(infoLog.data(), error->GetBufferPointer(), error->GetBufferSize());
        infoLog[error->GetBufferSize()] = 0;

        std::string errorMsg = "Shader Compiler Error:\n";
        errorMsg.append(infoLog.data());

        OutputDebugStringA(errorMsg.c_str());
        throw std::logic_error("Failed compile shader");
    }

    IDxcBlob* pBlob = nullptr;
    HR(pResult->GetResult(&pBlob));
    return pBlob;
}