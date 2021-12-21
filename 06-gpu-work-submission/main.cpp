#include "pch.h"
#include "debug.h"
#include "timer.h"

#pragma comment(lib,"d3dcompiler.lib")
#pragma comment(lib, "D3D12.lib")
#pragma comment(lib, "dxgi.lib")

using Microsoft::WRL::ComPtr;
using namespace DirectX;

constexpr UINT CLIENT_WIDTH = 1024;
constexpr UINT CLIENT_HEIGHT = 768;
constexpr DXGI_FORMAT BACK_BUFFER_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;
constexpr UINT NUM_BACK_BUFFER = 2;
constexpr UINT NUM_FRAME_RESOURCE = 3;

#define WINDOW_TITLE L"06. GPU work submission"
#define ALIGN(value, alignment) (((value) + ((alignment) - 1)) & ~((alignment) - 1))

#if defined(_DEBUG)
  #define ENABLE_DEBUG_LAYER true
#else
  #define ENABLE_DEBUG_LAYER false
#endif

struct Vertex
{
    XMFLOAT3 position;
    XMFLOAT4 color;
};

struct ConstantBuffer
{
    XMFLOAT4X4 worldViewProj;
};

struct DX12
{
    ComPtr<IDXGIFactory4> dxgiFactory;
    ComPtr<ID3D12Device5> device;
    ComPtr<ID3D12Fence> mainFence;
    ComPtr<ID3D12CommandQueue> commandQueue;
    ComPtr<ID3D12GraphicsCommandList> commandList;
    ComPtr<ID3D12CommandAllocator> commandAllocator;
    ComPtr<IDXGISwapChain> swapChain;

    ComPtr<ID3D12RootSignature> rootSignature;
    ComPtr<ID3D12PipelineState> pipelineState;

    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    ComPtr<ID3D12DescriptorHeap> dsvHeap;
    ComPtr<ID3D12DescriptorHeap> cbvHeap;

    ComPtr<ID3D12Resource> swapChainBuffers[2];
    ComPtr<ID3D12Resource> depthStencilBuffer;

    D3D12_VIEWPORT viewport;
    D3D12_RECT scissorRect;

    int msaaQuality = 0;

    UINT rtvDescSize = 0;
    UINT dsvDescSize = 0;
    UINT cbvSrvUavDescSize = 0;
};

struct Geometry
{
    std::string name;
    ComPtr<ID3D12Resource> vertexBuffer;
    ComPtr<ID3D12Resource> indexBuffer;
    UINT vertexStride = 0;
    UINT vertexBufferSize = 0;
    UINT indexBufferSize = 0;
    UINT indexCount = 0;

    D3D12_VERTEX_BUFFER_VIEW vertexBufferView()
    {
        D3D12_VERTEX_BUFFER_VIEW out;
        out.BufferLocation = vertexBuffer->GetGPUVirtualAddress();
        out.StrideInBytes = vertexStride;
        out.SizeInBytes = vertexBufferSize;
        return out;
    }

    D3D12_INDEX_BUFFER_VIEW indexBufferView()
    {
        D3D12_INDEX_BUFFER_VIEW out;
        out.BufferLocation = indexBuffer->GetGPUVirtualAddress();
        out.Format = DXGI_FORMAT_R16_UINT;
        out.SizeInBytes = indexBufferSize;
        return out;
    }
};

struct Camera
{
    POINT lastMousePosition;

    float theta = 1.5f * XM_PI;
    float phi = XM_PIDIV4;
    float radius = 5.0f;

    XMFLOAT4X4 world;
    XMFLOAT4X4 view;
    XMFLOAT4X4 proj;
};

struct FrameResource
{
    ComPtr<ID3D12CommandAllocator> commandAllocator;
    ComPtr<ID3D12Resource> passConstantBuffer;
    UINT64 fence = 0;
};

DX12 dx;
Geometry box;
GameTimer timer;
Camera camera;
FrameResource frameResources[NUM_FRAME_RESOURCE];

UINT currentFenceValue = 0;
UINT currentBackBufferIndex = 0;
UINT currentFrameResourceIndex = 0;

void FlushCommandQueue()
{
    currentFenceValue += 1;

    // Add an instruction to the command queue to set a new fence point.
    // Because we are on the GPU timeline,
    // the new fence point won't be set until the GPU finishes processing all the commands prior to this Signal().
    HR(dx.commandQueue->Signal(dx.mainFence.Get(), currentFenceValue));

    if (dx.mainFence->GetCompletedValue() < currentFenceValue)
    {
        HANDLE eventHandle = CreateEvent(nullptr, FALSE, FALSE, nullptr);

        dx.mainFence->SetEventOnCompletion(currentFenceValue, eventHandle);

        WaitForSingleObject(eventHandle, INFINITE);
        CloseHandle(eventHandle);
    }
}

ID3D12Resource* CurrentBackBuffer()
{
    return dx.swapChainBuffers[currentBackBufferIndex].Get();
}

D3D12_CPU_DESCRIPTOR_HANDLE CurrentBackBufferView()
{
    return CD3DX12_CPU_DESCRIPTOR_HANDLE(
        dx.rtvHeap->GetCPUDescriptorHandleForHeapStart(),
        currentBackBufferIndex,
        dx.rtvDescSize);
}

D3D12_CPU_DESCRIPTOR_HANDLE DepthStencilView()
{
    return dx.dsvHeap->GetCPUDescriptorHandleForHeapStart();
}

D3D12_CPU_DESCRIPTOR_HANDLE ConstantBufferView(int frameResourceIndex)
{
    return CD3DX12_CPU_DESCRIPTOR_HANDLE(
        dx.cbvHeap->GetCPUDescriptorHandleForHeapStart(),
        frameResourceIndex,
        dx.cbvSrvUavDescSize);
}

void InitD12(HWND hwnd)
{
    UINT dxgiFactoryFlags = 0;

    if (ENABLE_DEBUG_LAYER)
    {
        ID3D12Debug* debugController = nullptr;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
        {
            debugController->EnableDebugLayer();
            dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;

            ID3D12Debug1* debugController1 = nullptr;
            if (SUCCEEDED(debugController->QueryInterface(IID_PPV_ARGS(&debugController1))))
                debugController1->SetEnableGPUBasedValidation(true);
        }
    }

    // Create DXGI Factory, Device and Fence
    IDXGIFactory4* dxgiFactory = nullptr;
    HR(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&dxgiFactory)));

    ID3D12Device5* device = nullptr;
    HR(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));

    ID3D12Fence* fence = nullptr;
    HR(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));

    dx.dxgiFactory = dxgiFactory;
    dx.device = device;
    dx.mainFence = fence;

    // descriptor size
    dx.cbvSrvUavDescSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    dx.rtvDescSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    dx.dsvDescSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);

    // MSAA
    D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS msaaQualityLevels;
    msaaQualityLevels.Format = BACK_BUFFER_FORMAT;
    msaaQualityLevels.SampleCount = 4;
    msaaQualityLevels.Flags = D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE;
    msaaQualityLevels.NumQualityLevels = 0;
    dx.device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &msaaQualityLevels, sizeof(msaaQualityLevels));
    dx.msaaQuality = msaaQualityLevels.NumQualityLevels;

    // Create CommandQueue
    ID3D12CommandQueue* commandQueue = nullptr;


    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    HR(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue)));
    dx.commandQueue = commandQueue;

    ID3D12CommandAllocator* commandAllocator = nullptr;
    HR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocator)));
    dx.commandAllocator = commandAllocator;

    // Create the command list
    ID3D12GraphicsCommandList* commandList;
    HR(dx.device->CreateCommandList1(0, D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_FLAG_NONE, IID_PPV_ARGS(&commandList)));
    dx.commandList = commandList;
    dx.commandList->Close();  // Start off in a closed state.

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
    HR(dxgiFactory->CreateSwapChain(dx.commandQueue.Get(), &sd, &swapChain));

    dx.swapChain = swapChain;

    // Create Descriptor Heaps
    D3D12_DESCRIPTOR_HEAP_DESC rtvDesc;
    rtvDesc.NumDescriptors = NUM_BACK_BUFFER;
    rtvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    rtvDesc.NodeMask = 0;
    HR(device->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(&dx.rtvHeap)));

    D3D12_DESCRIPTOR_HEAP_DESC dsvDesc;
    dsvDesc.NumDescriptors = 1;
    dsvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    dsvDesc.NodeMask = 0;
    HR(device->CreateDescriptorHeap(&dsvDesc, IID_PPV_ARGS(&dx.dsvHeap)));

    D3D12_DESCRIPTOR_HEAP_DESC cbvDesc;
    cbvDesc.NumDescriptors = NUM_FRAME_RESOURCE;
    cbvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    cbvDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    cbvDesc.NodeMask = 0;
    HR(device->CreateDescriptorHeap(&cbvDesc, IID_PPV_ARGS(&dx.cbvHeap)));

    // Create RTV for each back buffer
    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHeapHandle(dx.rtvHeap->GetCPUDescriptorHandleForHeapStart());
    for (int i = 0; i < NUM_BACK_BUFFER; ++i)
    {
        HR(dx.swapChain->GetBuffer(i, IID_PPV_ARGS(&dx.swapChainBuffers[i])));
        device->CreateRenderTargetView(dx.swapChainBuffers[i].Get(), nullptr, rtvHeapHandle);
        rtvHeapHandle.Offset(1, dx.rtvDescSize);
    }

    currentBackBufferIndex = 0;

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
    HR(device->CreateCommittedResource(&heapProp,
                                       D3D12_HEAP_FLAG_NONE,
                                       &d,
                                       D3D12_RESOURCE_STATE_COMMON,
                                       &c,
                                       IID_PPV_ARGS(&depthStencilBuffer)));

    dx.depthStencilBuffer = depthStencilBuffer;

    D3D12_CPU_DESCRIPTOR_HANDLE handle = dx.dsvHeap->GetCPUDescriptorHandleForHeapStart();
    dx.device->CreateDepthStencilView(dx.depthStencilBuffer.Get(), nullptr, handle); // nullptr only works when the buffer is not typeless

    dx.viewport.TopLeftX = 0;
    dx.viewport.TopLeftY = 0;
    dx.viewport.Width = FLOAT(CLIENT_WIDTH);
    dx.viewport.Height = FLOAT(CLIENT_HEIGHT);
    dx.viewport.MinDepth = 0.0f;
    dx.viewport.MaxDepth = 1.0f;

    dx.scissorRect = { 0, 0, CLIENT_WIDTH, CLIENT_HEIGHT };
}

void InitGeometry()
{
    // Define the geometry for a box
    // Vertex buffer
    Vertex vertices[8] =
    {
        {{ -1, -1, -1 }, { XMFLOAT4(Colors::White) }},
        {{ -1, +1, -1 }, { XMFLOAT4(Colors::Black) }},
        {{ +1, +1, -1 }, { XMFLOAT4(Colors::Red) }},
        {{ +1, -1, -1 }, { XMFLOAT4(Colors::Green) }},
        {{ -1, -1, +1 }, { XMFLOAT4(Colors::Blue) }},
        {{ -1, +1, +1 }, { XMFLOAT4(Colors::Yellow) }},
        {{ +1, +1, +1 }, { XMFLOAT4(Colors::Cyan) }},
        {{ +1, -1, +1 }, { XMFLOAT4(Colors::Magenta) }},
    };

    // Index Buffer
    constexpr UINT numIndex = 6 * 6; // 6 vertices per face
    UINT16 indexes[numIndex] =
    {
        0, 1, 2, 0, 2, 3, // front face
        4, 6, 5, 4, 7, 6, // back face
        4, 5, 1, 4, 1, 0, // left face
        3, 2, 6, 3, 6, 7, // right face
        1, 5, 6, 1, 6, 2, // top face
        4, 0, 3, 4, 3, 7, // bottom face
    };

    box.name = "Box";
    box.vertexBufferSize = sizeof(vertices);
    box.indexBufferSize = sizeof(indexes);
    box.vertexStride = sizeof(Vertex);
    box.indexCount = ARRAYSIZE(indexes);

    // Note: using upload heaps to transfer static data like vertex buffers is not recommended.
    // Every time the GPU needs it, the upload heap will be marshalled over.
    // Please read up on Default Heap usage.
    // An upload heap is used here for code simplicity and because there are very few vertices to actually transfer.
    CD3DX12_HEAP_PROPERTIES healProperties(D3D12_HEAP_TYPE_UPLOAD);

    // Vertex Buffer
    CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(box.vertexBufferSize);

    HR(dx.device->CreateCommittedResource(
        &healProperties,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, // initial state
        nullptr, // initial value
        IID_PPV_ARGS(&box.vertexBuffer)));

    // Copy the data to the vertex buffer.
    BYTE* data = 0;
    CD3DX12_RANGE readRange(0, 0); // We do not intend to read from this resource on the CPU.
    HR(box.vertexBuffer->Map(0, &readRange, (void**)&data));
    {
        memcpy(data, vertices, box.vertexBufferSize);
    }
    box.vertexBuffer->Unmap(0, nullptr);

    // Index Buffer
    bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(box.indexBufferSize);
    HR(dx.device->CreateCommittedResource(
        &healProperties,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, // initial state
        nullptr, // initial value
        IID_PPV_ARGS(&box.indexBuffer)));

    // Copy the data to the index buffer.
    BYTE* data2 = 0;
    CD3DX12_RANGE readRange2(0, 0); // We do not intend to read from this resource on the CPU.
    HR(box.indexBuffer->Map(0, &readRange2, (void**)&data2));
    {
        memcpy(data2, indexes, box.indexBufferSize);
    }
    box.indexBuffer->Unmap(0, nullptr);
}

void InitFrameResources(ID3D12Device* device)
{
    for (int i = 0; i < NUM_FRAME_RESOURCE; ++i)
    {
        FrameResource& f = frameResources[i];

        HR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(f.commandAllocator.GetAddressOf())));

        constexpr UINT constantBufferSize = ALIGN(sizeof(ConstantBuffer), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
        CD3DX12_RESOURCE_DESC d1 = CD3DX12_RESOURCE_DESC::Buffer(constantBufferSize);
        dx.device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
            &d1, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(f.passConstantBuffer.GetAddressOf()));

        D3D12_CONSTANT_BUFFER_VIEW_DESC d2;
        d2.BufferLocation = f.passConstantBuffer->GetGPUVirtualAddress();
        d2.SizeInBytes = constantBufferSize;
        dx.device->CreateConstantBufferView(&d2, ConstantBufferView(i));
    }
}

void InitCamera()
{
    XMStoreFloat4x4(&camera.world, XMMatrixIdentity());
    XMStoreFloat4x4(&camera.view, XMMatrixIdentity());

    const float aspectRatio = float(CLIENT_WIDTH) / float(CLIENT_HEIGHT);
    XMMATRIX P = XMMatrixPerspectiveFovLH(0.25f * XM_PI, aspectRatio, 1.0f, 1000.0f);
    XMStoreFloat4x4(&camera.proj, P);
}

void UpdateCamera(FrameResource& f)
{
    // Convert Spherical to Cartesian coordinates.
    const float x = camera.radius * sinf(camera.phi) * cosf(camera.theta);
    const float z = camera.radius * sinf(camera.phi) * sinf(camera.theta);
    const float y = camera.radius * cosf(camera.phi);

    // Build the view matrix.
    XMVECTOR pos = XMVectorSet(x, y, z, 1.0f);
    XMVECTOR target = XMVectorZero();
    XMVECTOR up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);

    XMMATRIX view = XMMatrixLookAtLH(pos, target, up);
    XMStoreFloat4x4(&camera.view, view);

    XMMATRIX world = XMLoadFloat4x4(&camera.world);
    XMMATRIX proj = XMLoadFloat4x4(&camera.proj);
    XMMATRIX worldViewProj = world * view * proj;

    ConstantBuffer cb;
    XMStoreFloat4x4(&cb.worldViewProj, XMMatrixTranspose(worldViewProj));

    BYTE* data = nullptr;
    f.passConstantBuffer->Map(0, nullptr, (void**)&data);
    memcpy(data, &cb, sizeof(cb));
    f.passConstantBuffer->Unmap(0, nullptr);
}


void Update(int frameCount)
{
    currentFrameResourceIndex = (currentFrameResourceIndex + 1) % NUM_FRAME_RESOURCE;
    FrameResource f = frameResources[currentFrameResourceIndex];

    if (f.fence != 0 && dx.mainFence->GetCompletedValue() < f.fence)
    {
        HANDLE eventHandle = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        dx.mainFence->SetEventOnCompletion(f.fence, eventHandle);

        WaitForSingleObject(eventHandle, INFINITE);
        CloseHandle(eventHandle);
    }

    UpdateCamera(f);
}

void InitPipeline()
{
    // Create a root signature.
    D3D12_DESCRIPTOR_RANGE range;
    range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
    range.NumDescriptors = 1;
    range.BaseShaderRegister = 0;
    range.RegisterSpace = 0;
    range.OffsetInDescriptorsFromTableStart = 0;

    D3D12_ROOT_PARAMETER rootParameters[1];
    rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rootParameters[0].DescriptorTable.NumDescriptorRanges = 1;
    rootParameters[0].DescriptorTable.pDescriptorRanges = &range;
    CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc;
    rootSignatureDesc.Init(1, rootParameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

    ComPtr<ID3DBlob> signature;
    ComPtr<ID3DBlob> error;
    HR(D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error));
    HR(dx.device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&dx.rootSignature)));

    // Create the pipeline state, which includes compiling and loading shaders.
    {
        ComPtr<ID3DBlob> vertexShader;
        ComPtr<ID3DBlob> pixelShader;

        UINT compileFlags = (ENABLE_DEBUG_LAYER) ? (D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION) : 0;
        HR(D3DCompileFromFile(L"shaders.hlsl", nullptr, nullptr, "VSMain", "vs_5_0", compileFlags, 0, &vertexShader, nullptr));
        HR(D3DCompileFromFile(L"shaders.hlsl", nullptr, nullptr, "PSMain", "ps_5_0", compileFlags, 0, &pixelShader, nullptr));

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
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        psoDesc.SampleDesc.Count = 1;
        HR(dx.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&dx.pipelineState)));
    }

    dx.commandList->Reset(dx.commandAllocator.Get(), dx.pipelineState.Get());

    // Resource barrier for depth stencil buffer
    auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(dx.depthStencilBuffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    dx.commandList->ResourceBarrier(1, &barrier);
    dx.commandList->Close(); // done recording

    ID3D12CommandList* cmdLists[] = { dx.commandList.Get() };
    dx.commandQueue->ExecuteCommandLists(1, cmdLists);

    FlushCommandQueue();
}

void Render()
{
    FrameResource& fr = frameResources[currentFrameResourceIndex];
    // We can only reset when the associated command lists have finished execution on the GPU.
    HR(fr.commandAllocator->Reset());

    // A command list can be reset after it has been added to the command queue via ExecuteCommandList.
    // Reusing the command list reuses memory.
    HR(dx.commandList->Reset(fr.commandAllocator.Get(), dx.pipelineState.Get()));

    // This needs to be reset whenever the command list is reset.
    dx.commandList->SetGraphicsRootSignature(dx.rootSignature.Get());

    ID3D12DescriptorHeap* heaps[] { dx.cbvHeap.Get() };
    dx.commandList->SetDescriptorHeaps(1, heaps);
    dx.commandList->SetGraphicsRootDescriptorTable(0, dx.cbvHeap->GetGPUDescriptorHandleForHeapStart());
    dx.commandList->RSSetViewports(1, &dx.viewport);
    dx.commandList->RSSetScissorRects(1, &dx.scissorRect);

    dx.commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(CurrentBackBuffer(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));

    // Clear back buffer & depth stencil buffer
    FLOAT red[]{ 67 / 255.f, 183 / 255.f, 194 / 255.f, 1.0f };
    dx.commandList->ClearRenderTargetView(CurrentBackBufferView(), red, 0, nullptr);
    dx.commandList->ClearDepthStencilView(DepthStencilView(), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 0, 0, nullptr);

    // Specify the buffers we are going to render to.
    dx.commandList->OMSetRenderTargets(1, &CurrentBackBufferView(), true, &DepthStencilView());

    // Draw the triangle
    dx.commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    dx.commandList->IASetVertexBuffers(0, 1, &box.vertexBufferView());
    dx.commandList->IASetIndexBuffer(&box.indexBufferView());
    dx.commandList->DrawIndexedInstanced(box.indexCount, 1, 0, 0, 0);

    // Indicate a state transition on the resource usage.
    dx.commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(CurrentBackBuffer(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));

    HR(dx.commandList->Close()); // Done recording commands.

    // Execute command list
    ID3D12CommandList* cmdsLists[] = { dx.commandList.Get() };
    dx.commandQueue->ExecuteCommandLists(ARRAYSIZE(cmdsLists), cmdsLists);

    HR(dx.swapChain->Present(0, 0));
    currentBackBufferIndex = (currentBackBufferIndex + 1) % NUM_BACK_BUFFER;

    // Advance the fence value for the current frame
    currentFenceValue += 1;

    fr.fence = currentFenceValue;

    //the new fence value won't be set until the GPU finishes processing all the commands prior to this Signal()
    HR(dx.commandQueue->Signal(dx.mainFence.Get(), currentFenceValue));
}

float Clamp(const float x, const float low, const float high)
{
    return x < low ? low : (x > high ? high : x);
}

void OnMouseDown(WPARAM btnState, int x, int y)
{
    camera.lastMousePosition.x = x;
    camera.lastMousePosition.y = y;
}

void OnMouseUp(WPARAM btnState, int x, int y)
{
}

void OnMouseMove(WPARAM btnState, int x, int y)
{
    if ((btnState & MK_LBUTTON) != 0)
    {
        // Make each pixel correspond to a quarter of a degree.
        float dx = XMConvertToRadians(0.25f * float(x - camera.lastMousePosition.x));
        float dy = XMConvertToRadians(0.25f * float(y - camera.lastMousePosition.y));

        // Update angles based on input to orbit camera around box.
        camera.theta -= dx;
        camera.phi -= dy;

        // Restrict the angle mPhi.
        camera.phi = Clamp(camera.phi, 0.1f, XM_PI - 0.1f);
    }
    else if ((btnState & MK_RBUTTON) != 0)
    {
        // Make each pixel correspond to 0.005 unit in the scene.
        float dx = 0.005f * float(x - camera.lastMousePosition.x);
        float dy = 0.005f * float(y - camera.lastMousePosition.y);

        // Update the camera radius based on input.
        camera.radius += dx - dy;

        // Restrict the radius.
        camera.radius = Clamp(camera.radius, 3.0f, 15.0f);
    }

    camera.lastMousePosition.x = x;
    camera.lastMousePosition.y = y;
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
    case WM_LBUTTONDOWN:
    case WM_MBUTTONDOWN:
    case WM_RBUTTONDOWN:
        OnMouseDown(wparam, GET_X_LPARAM(lparam), GET_Y_LPARAM(lparam));
        break;
    case WM_LBUTTONUP:
    case WM_MBUTTONUP:
    case WM_RBUTTONUP:
        OnMouseUp(wparam, GET_X_LPARAM(lparam), GET_Y_LPARAM(lparam));
        break;
    case WM_MOUSEMOVE:
        OnMouseMove(wparam, GET_X_LPARAM(lparam), GET_Y_LPARAM(lparam));
        break;
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

    RECT initialRect{ 0, 0, CLIENT_WIDTH, CLIENT_HEIGHT };
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

void CalculateFrameStats(GameTimer& timer, HWND hwnd)
{
    // Code computes the average frames per second, and also the average time it takes to render one frame.

    static int frameCount = 0;
    static float timeElapsed = 0.0f;

    frameCount++;

    // Compute averages over one second period.
    if ((timer.TotalTime() - timeElapsed) >= 1.0f)
    {
        float fps = (float)frameCount; // fps = frameCnt / 1
        float mspf = 1000.0f / fps;

        std::wstring fpsStr = std::to_wstring(fps);
        std::wstring mspfStr = std::to_wstring(mspf);

        std::wstringstream windowText;
        windowText << WINDOW_TITLE << L"    fps: " << fpsStr << L"   mspf: " << mspfStr;

        SetWindowText(hwnd, windowText.str().c_str());

        // Reset for next average.
        frameCount = 0;
        timeElapsed += 1.0f;
    }
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int)
{
#ifdef _DEBUG
    // Enable runtime memory check for debug builds
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

    // Open a window
    int errorCode = 0;
    HWND hwnd = CreateMainWindow(hInstance, &errorCode);

    if (errorCode != 0)
        return errorCode;

    InitD12(hwnd);
    InitFrameResources(dx.device.Get());
    InitGeometry();
    InitPipeline();
    InitCamera();

    timer.Reset();
    int frame = 0;

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

        timer.Tick();
        CalculateFrameStats(timer, hwnd);

        Update(frame);
        Render();

        ++frame;
    }

    return 0;
}