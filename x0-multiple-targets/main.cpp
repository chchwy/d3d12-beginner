#include "pch.h"
#include "debug.h"

#pragma comment(lib,"d3dcompiler.lib")
#pragma comment(lib, "D3D12.lib")
#pragma comment(lib, "dxgi.lib")

#define WINDOW_TITLE L"Debug Multiple RenderTarget"

using Microsoft::WRL::ComPtr;
using DirectX::XMFLOAT3;
using DirectX::XMFLOAT4;

const static UINT CLIENT_WIDTH = 1024;
const static UINT CLIENT_HEIGHT = 768;
const static DXGI_FORMAT BACK_BUFFER_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;
const static UINT NUM_BACK_BUFFER = 2;

#define ENABLE_DEBUG_LAYER true

struct Vertex
{
    XMFLOAT3 position;
    XMFLOAT4 color;
};

struct DX12
{
    ComPtr<IDXGIFactory4> dxgiFactory;
    ComPtr<ID3D12Device5> device;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12CommandQueue> commandQueue;
    ComPtr<ID3D12GraphicsCommandList> commandList;
    ComPtr<ID3D12CommandAllocator> commandAllocator;
    ComPtr<IDXGISwapChain> swapChain;

    ComPtr<ID3D12RootSignature> rootSignature;
    ComPtr<ID3D12PipelineState> pipelineState;

    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    ComPtr<ID3D12DescriptorHeap> dsvHeap;

    ComPtr<ID3D12Resource> swapChainBuffers[2];
    ComPtr<ID3D12Resource> depthStencilBuffer;
    ComPtr<ID3D12Resource> debug01;
    D3D12_CPU_DESCRIPTOR_HANDLE debug01RTV;

    D3D12_VIEWPORT viewport;
    D3D12_RECT scissorRect;

    int width = 0;
    int height = 0;
    int msaaQuality = 0;

    UINT rtvDescSize = 0;
    UINT dsvDescSize = 0;
    UINT cbvSrvUavDescSize = 0;
};

struct Resource
{
    ComPtr<ID3D12Resource> vertexBuffer;
    D3D12_VERTEX_BUFFER_VIEW vertexBufferView;
};

DX12 dx;
Resource rsc;
UINT currentFence = 0;
UINT currBackBuffer = 0;

void FlushCommandQueue()
{
    currentFence += 1;

    // Add an instruction to the command queue to set a new fence point.  Because we
    // are on the GPU timeline, the new fence point won't be set until the GPU finishes
    // processing all the commands prior to this Signal().
    HR(dx.commandQueue->Signal(dx.fence.Get(), currentFence));

    if (dx.fence->GetCompletedValue() < currentFence)
    {
        HANDLE eventHandle = CreateEventEx(nullptr, false, false, EVENT_ALL_ACCESS);

        dx.fence->SetEventOnCompletion(currentFence, eventHandle);

        WaitForSingleObject(eventHandle, INFINITE);
        CloseHandle(eventHandle);
    }
}

ID3D12Resource* CurrentBackBuffer()
{
    return dx.swapChainBuffers[currBackBuffer].Get();
}

D3D12_CPU_DESCRIPTOR_HANDLE CurrentBackBufferView()
{
    return CD3DX12_CPU_DESCRIPTOR_HANDLE(
        dx.rtvHeap->GetCPUDescriptorHandleForHeapStart(),
        currBackBuffer,
        dx.rtvDescSize);
}

D3D12_CPU_DESCRIPTOR_HANDLE DepthStencilView()
{
    return dx.dsvHeap->GetCPUDescriptorHandleForHeapStart();
}

void InitD3D(HWND hwnd)
{
    dx.width = CLIENT_WIDTH;
    dx.height = CLIENT_HEIGHT;

    UINT dxgiFactoryFlags = 0;

#ifdef ENABLE_DEBUG_LAYER
    ID3D12Debug* debugController;
    HRESULT enableDebug = D3D12GetDebugInterface(IID_PPV_ARGS(&debugController));
    if (SUCCEEDED(enableDebug))
    {
        debugController->EnableDebugLayer();
        dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
    }
#endif

    // Create DXGI Factory, Device and Fence
    IDXGIFactory4* dxgiFactory = nullptr;
    HR(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&dxgiFactory)));

    ID3D12Device5* device = nullptr;
    HR(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));

    ID3D12Fence* fence = nullptr;
    HR(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));

    dx.dxgiFactory = dxgiFactory;
    dx.device = device;
    dx.fence = fence;

    dx.cbvSrvUavDescSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // MSAA
    D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS msaaQualityLevels;
    msaaQualityLevels.Format = BACK_BUFFER_FORMAT;
    msaaQualityLevels.SampleCount = 4;
    msaaQualityLevels.Flags = D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE;
    msaaQualityLevels.NumQualityLevels = 0;
    dx.device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &msaaQualityLevels, sizeof(msaaQualityLevels));
    dx.msaaQuality = msaaQualityLevels.NumQualityLevels;

    // Create CommandQueue
    ID3D12CommandQueue* commandQueue;
    ID3D12CommandAllocator* commandAllocator;

    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    HR(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue)));
    HR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocator)));

    dx.commandQueue = commandQueue;
    dx.commandAllocator = commandAllocator;

    // Create SwapChain
    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferDesc.Width = CLIENT_WIDTH;
    sd.BufferDesc.Height = CLIENT_HEIGHT;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferDesc.Format = BACK_BUFFER_FORMAT;
    sd.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
    sd.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = dx.msaaQuality - 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = NUM_BACK_BUFFER;
    sd.OutputWindow = hwnd;
    sd.Windowed = true;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

    IDXGISwapChain* swapChain = nullptr;
    // First parameter said it's a device, but in Dx12 you need to pass a CommandQueue
    HR(dx.dxgiFactory->CreateSwapChain(dx.commandQueue.Get(), &sd, &swapChain));

    dx.swapChain = swapChain;

    // Create Descriptor Heaps
    D3D12_DESCRIPTOR_HEAP_DESC rtvDesc;
    rtvDesc.NumDescriptors = NUM_BACK_BUFFER + 1;
    rtvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    rtvDesc.NodeMask = 0;
    HR(device->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(&dx.rtvHeap)));

    dx.rtvDescSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    D3D12_DESCRIPTOR_HEAP_DESC dsvDesc;
    dsvDesc.NumDescriptors = 1;
    dsvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    dsvDesc.NodeMask = 0;
    HR(device->CreateDescriptorHeap(&dsvDesc, IID_PPV_ARGS(&dx.dsvHeap)));

    dx.dsvDescSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);

    // Create RTV for each back buffer
    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHeapHandle(dx.rtvHeap->GetCPUDescriptorHandleForHeapStart());
    for (int i = 0; i < NUM_BACK_BUFFER; ++i)
    {
        HR(dx.swapChain->GetBuffer(i, IID_PPV_ARGS(&dx.swapChainBuffers[i])));

        dx.device->CreateRenderTargetView(dx.swapChainBuffers[i].Get(), nullptr, rtvHeapHandle);
        rtvHeapHandle.Offset(1, dx.rtvDescSize);
    }

    currBackBuffer = 0;

    // Depth Stencil Buffer
    D3D12_RESOURCE_DESC d;
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Alignment = 0;
    d.Width = CLIENT_WIDTH;
    d.Height = CLIENT_HEIGHT;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    d.SampleDesc.Count = 1;
    d.SampleDesc.Quality = dx.msaaQuality - 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

    D3D12_CLEAR_VALUE c;
    c.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    c.DepthStencil.Depth = 1.0f;
    c.DepthStencil.Stencil = 0;

    ID3D12Resource* depthStencilBuffer = nullptr;
    CD3DX12_HEAP_PROPERTIES heapProp(D3D12_HEAP_TYPE_DEFAULT);
    HR(dx.device->CreateCommittedResource(&heapProp,
                                          D3D12_HEAP_FLAG_NONE,
                                          &d,
                                          D3D12_RESOURCE_STATE_COMMON,
                                          &c,
                                          IID_PPV_ARGS(&depthStencilBuffer)));

    dx.depthStencilBuffer = depthStencilBuffer;

    D3D12_CPU_DESCRIPTOR_HANDLE handle = dx.dsvHeap->GetCPUDescriptorHandleForHeapStart();
    dx.device->CreateDepthStencilView(dx.depthStencilBuffer.Get(), nullptr, handle); // nullptr only works when the buffer is not typeless

    // second render target
    D3D12_RESOURCE_DESC d2;
    d2.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d2.Alignment = 0;
    d2.Width = CLIENT_WIDTH;
    d2.Height = CLIENT_HEIGHT;
    d2.DepthOrArraySize = 1;
    d2.MipLevels = 1;
    d2.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    d2.SampleDesc.Count = 1;
    d2.SampleDesc.Quality = 0;
    d2.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    d2.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

    ID3D12Resource* debug01 = nullptr;
    HR(dx.device->CreateCommittedResource(&heapProp,
        D3D12_HEAP_FLAG_NONE,
        &d2,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&debug01)));
    assert(debug01);
    dx.debug01 = debug01;
    dx.debug01->SetName(L"Debug01");

    dx.device->CreateRenderTargetView(dx.debug01.Get(), nullptr, rtvHeapHandle);
    dx.debug01RTV = rtvHeapHandle;
    rtvHeapHandle.Offset(1, dx.rtvDescSize);

    dx.viewport.TopLeftX = 0;
    dx.viewport.TopLeftY = 0;
    dx.viewport.Width = FLOAT(dx.width);
    dx.viewport.Height = FLOAT(dx.height);
    dx.viewport.MinDepth = 0.0f;
    dx.viewport.MaxDepth = 1.0f;

    dx.scissorRect = { 0, 0, dx.width, dx.height };
}

void InitPipeline()
{
    // Create an empty root signature.
    CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc;
    rootSignatureDesc.Init(0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

    ComPtr<ID3DBlob> signature;
    ComPtr<ID3DBlob> error;
    HR(D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error));
    HR(dx.device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&dx.rootSignature)));

    // Create the pipeline state, which includes compiling and loading shaders.
    {
        ComPtr<ID3DBlob> vertexShader;
        ComPtr<ID3DBlob> pixelShader;

        UINT compileFlags = (ENABLE_DEBUG_LAYER) ? (D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION) : 0;
        ID3DBlob* errorBlob = nullptr;
        HRESULT h1 = D3DCompileFromFile(L"shaders.hlsl", nullptr, nullptr, "VSMain", "vs_5_0", compileFlags, 0, &vertexShader, &errorBlob);
        if (FAILED(h1))
        {
            OutputDebugStringA((char*)errorBlob->GetBufferPointer());
            errorBlob->Release();
        }
        HRESULT h2 = D3DCompileFromFile(L"shaders.hlsl", nullptr, nullptr, "PSMain", "ps_5_0", compileFlags, 0, &pixelShader, &errorBlob);
        if (FAILED(h2))
        {
            OutputDebugStringA((char*)errorBlob->GetBufferPointer());
            errorBlob->Release();
        }

        // Define the vertex input layout.
        D3D12_INPUT_ELEMENT_DESC inputElementDescs[] =
        {
            { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
        };

        // Describe and create the graphics pipeline state object (PSO).
        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.InputLayout = { inputElementDescs, _countof(inputElementDescs) };
        psoDesc.pRootSignature = dx.rootSignature.Get();
        psoDesc.VS = CD3DX12_SHADER_BYTECODE(vertexShader.Get());
        psoDesc.PS = CD3DX12_SHADER_BYTECODE(pixelShader.Get());
        psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        psoDesc.DepthStencilState.DepthEnable = FALSE;
        psoDesc.DepthStencilState.StencilEnable = FALSE;
        psoDesc.SampleMask = UINT_MAX;
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psoDesc.NumRenderTargets = 2;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        psoDesc.RTVFormats[1] = DXGI_FORMAT_R32G32B32A32_FLOAT;
        psoDesc.SampleDesc.Count = 1;
        HR(dx.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&dx.pipelineState)));
    }

    // Create the vertex buffer.
    {
        // Define the geometry for a triangle.
        Vertex triangleVertices[] =
        {
            { { 0.0f, 1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f, 1.0f } },
            { { 1.0f, -1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f, 1.0f } },
            { { -1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f, 1.0f } }
        };

        const UINT vertexBufferSize = sizeof(triangleVertices);

        // Note: using upload heaps to transfer static data like vertex buffers is not recommended.
        // Every time the GPU needs it, the upload heap will be marshalled over.
        // Please read up on Default Heap usage.
        // An upload heap is used here for code simplicity and because there are very few vertices to actually transfer.
        CD3DX12_HEAP_PROPERTIES healProperties(D3D12_HEAP_TYPE_UPLOAD);
        CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize);

        HR(dx.device->CreateCommittedResource(
            &healProperties,
            D3D12_HEAP_FLAG_NONE,
            &bufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, // init state
            nullptr, // init value
            IID_PPV_ARGS(&rsc.vertexBuffer)));

        // Copy the triangle data to the vertex buffer.
        UINT8* data = 0;
        CD3DX12_RANGE readRange(0, 0); // We do not intend to read from this resource on the CPU.
        HR(rsc.vertexBuffer->Map(0, &readRange, (void**)&data));
        {
            memcpy(data, triangleVertices, vertexBufferSize);
        }
        rsc.vertexBuffer->Unmap(0, nullptr);

        // Initialize the vertex buffer view.
        rsc.vertexBufferView.BufferLocation = rsc.vertexBuffer->GetGPUVirtualAddress();
        rsc.vertexBufferView.StrideInBytes = sizeof(Vertex);
        rsc.vertexBufferView.SizeInBytes = vertexBufferSize;
    }

    // Resource barrier for depth stencil buffer
    // Create the command list
    ID3D12GraphicsCommandList* commandList;
    HR(dx.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, dx.commandAllocator.Get(), dx.pipelineState.Get(), IID_PPV_ARGS(&commandList)));
    dx.commandList = commandList;

    // Start off in a closed state.
    // it needs to be closed before calling Reset
    commandList->Close();
    commandList->Reset(dx.commandAllocator.Get(), dx.pipelineState.Get());

    auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(dx.depthStencilBuffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    dx.commandList->ResourceBarrier(1, &barrier);

    auto barrier2 = CD3DX12_RESOURCE_BARRIER::Transition(dx.debug01.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
    dx.commandList->ResourceBarrier(1, &barrier2);

    dx.commandList->Close(); // done recording

    ID3D12CommandList* cmdLists[] = { dx.commandList.Get() };
    dx.commandQueue->ExecuteCommandLists(1, cmdLists);

    FlushCommandQueue();
}

void Render()
{
    // We can only reset when the associated command lists have finished execution on the GPU.
    HR(dx.commandAllocator->Reset());

    // A command list can be reset after it has been added to the command queue via ExecuteCommandList.
    // Reusing the command list reuses memory.
    HR(dx.commandList->Reset(dx.commandAllocator.Get(), dx.pipelineState.Get()));

    // This needs to be reset whenever the command list is reset.
    dx.commandList->SetGraphicsRootSignature(dx.rootSignature.Get());
    dx.commandList->RSSetViewports(1, &dx.viewport);
    dx.commandList->RSSetScissorRects(1, &dx.scissorRect);

    dx.commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(CurrentBackBuffer(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));

    // Clear back buffer & depth stencil buffer
    FLOAT red[]{ 67 / 255.f, 183 / 255.f, 194 / 255.f, 1.0f };
    dx.commandList->ClearRenderTargetView(CurrentBackBufferView(), red, 0, nullptr);
    //dx.commandList->ClearDepthStencilView(DepthStencilView(), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 0, 0, nullptr);

    // Specify the buffers we are going to render to.
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv[]{ CurrentBackBufferView(), dx.debug01RTV };
    dx.commandList->OMSetRenderTargets(2, rtv, true, &DepthStencilView());

    // Draw the triangle
    dx.commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    dx.commandList->IASetVertexBuffers(0, 1, &rsc.vertexBufferView);
    dx.commandList->DrawInstanced(3, 1, 0, 0);

    // Indicate a state transition on the resource usage.
    dx.commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(CurrentBackBuffer(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));

    HR(dx.commandList->Close()); // Done recording commands.

    // Execute command list
    ID3D12CommandList* cmdsLists[] = { dx.commandList.Get() };
    dx.commandQueue->ExecuteCommandLists(_countof(cmdsLists), cmdsLists);

    HR(dx.swapChain->Present(0, 0));
    currBackBuffer = (currBackBuffer + 1) % NUM_BACK_BUFFER;

    // Wait until frame commands are complete.
    // This waiting is inefficient and is done for simplicity.
    // Later we will show how to organize our rendering code so we do not have to wait per frame.
    FlushCommandQueue();
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam)
{
    LRESULT result = 0;
    switch (msg)
    {
    case WM_KEYDOWN:
    {
        if (wparam == VK_ESCAPE) DestroyWindow(hwnd);
        break;
    }
    case WM_DESTROY:
    {
        PostQuitMessage(0);
        break;
    }
    default:
        result = DefWindowProcW(hwnd, msg, wparam, lparam);
    }
    return result;
}

HWND CreateMainWindow(HINSTANCE hInstance, int* errorCode)
{
    WNDCLASSEXW winClass = {};
    winClass.cbSize = sizeof(WNDCLASSEXW);
    winClass.style = CS_HREDRAW | CS_VREDRAW;
    winClass.lpfnWndProc = &WndProc;
    winClass.hInstance = hInstance;
    winClass.hIcon = LoadIconW(0, IDI_APPLICATION);
    winClass.hCursor = LoadCursorW(0, IDC_ARROW);
    winClass.lpszClassName = L"MyWindowClass";
    winClass.hIconSm = LoadIconW(0, IDI_APPLICATION);

    if (!RegisterClassExW(&winClass))
    {
        MessageBoxA(0, "RegisterClassEx failed", "Fatal Error", MB_OK);
        *errorCode = GetLastError();
        return HWND();
    }

    RECT initialRect { 0, 0, CLIENT_WIDTH, CLIENT_HEIGHT };
    AdjustWindowRectEx(&initialRect, WS_OVERLAPPEDWINDOW, FALSE, WS_EX_OVERLAPPEDWINDOW);
    LONG initialWidth = initialRect.right - initialRect.left;
    LONG initialHeight = initialRect.bottom - initialRect.top;

    HWND hwnd = CreateWindowExW(WS_EX_OVERLAPPEDWINDOW,
        winClass.lpszClassName,
        WINDOW_TITLE,
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT,
        initialWidth,
        initialHeight,
        0, 0, hInstance, 0);

    if (!hwnd)
    {
        MessageBoxA(0, "CreateWindowEx failed", "Fatal Error", MB_OK);
        *errorCode = GetLastError();
        return HWND();
    }
    return  hwnd;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int)
{
    // Open a window
    int errorCode = 0;
    HWND hwnd = CreateMainWindow(hInstance, &errorCode);

    if (errorCode != 0)
        return errorCode;

    InitD3D(hwnd);
    InitPipeline();

    bool isRunning = true;
    while (isRunning)
    {
        MSG message = {};
        while (PeekMessageW(&message, 0, 0, 0, PM_REMOVE))
        {
            if (message.message == WM_QUIT)
            {
                isRunning = false;
            }
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
        Render();
    }

    return 0;
}