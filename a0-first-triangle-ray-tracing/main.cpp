
#include "pch.h"
#include "debug.h"
#include "shader.h"

#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxcompiler.lib")
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

using Microsoft::WRL::ComPtr;
using DirectX::XMFLOAT3;
using DirectX::XMFLOAT4;
using DirectX::XMMatrixIdentity;
using DirectX::XMMatrixTranspose;

const static UINT CLIENT_WIDTH = 1024;
const static UINT CLIENT_HEIGHT = 768;
const static DXGI_FORMAT BACK_BUFFER_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;
const static UINT NUM_BACK_BUFFER = 2;

#define WINDOW_TITLE L"a0. RayTracing First Triangle"
#define ENABLE_DEBUG_LAYER

#ifndef ROUND_UP
#define ROUND_UP(v, powerOf2Alignment)                                         \
  (((v) + (powerOf2Alignment)-1) & ~((powerOf2Alignment)-1))
#endif

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
    ComPtr<ID3D12GraphicsCommandList4> commandList;
    ComPtr<ID3D12CommandAllocator> commandAllocator;
    ComPtr<IDXGISwapChain> swapChain;

    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    ComPtr<ID3D12DescriptorHeap> dsvHeap;

    ComPtr<ID3D12Resource> swapChainBuffers[2];
    ComPtr<ID3D12Resource> depthStencilBuffer;

    D3D12_VIEWPORT viewport = {};
    D3D12_RECT scissorRect = {};

    int width = 0;
    int height = 0;
    int msaaQuality = 0;

    UINT rtvDescSize = 0;
    UINT dsvDescSize = 0;
    UINT cbvSrvUavDescSize = 0;
};

struct Geometry
{
    ComPtr<ID3D12Resource> vertexBuffer;
    UINT bufferSize = 0;
};

struct AccelerationStructureBuffers
{
    ComPtr<ID3D12Resource> scratch;      // Scratch memory for AS builder
    ComPtr<ID3D12Resource> result;       // Where the AS is
    ComPtr<ID3D12Resource> instanceDesc; // Hold the matrices of the instances
};

struct ASInstance
{
    ComPtr<ID3D12Resource> bottomLevelResult;
    ComPtr<ID3D12Resource> bottomLevelScratch;
    DirectX::XMMATRIX transform;
    UINT instanceID = 0;
    UINT hitGroupIndex = 0;
};

struct ShaderBindingTable
{
    ID3D12Resource* resource;
    UINT64 rayGenStartAddress = 0;
    UINT   rayGenSectionSize = 0;
    UINT64 missStartAddress = 0;
    UINT   missSectionSize = 0;
    UINT   missStride = 0;
    UINT64 hitGroupStartAddress = 0;
    UINT   hitGroupSectionSize = 0;
    UINT   hitGroupStride = 0;
};

struct DXR
{
    AccelerationStructureBuffers topLevelASBuffers;

    std::vector<ASInstance> instances;

    ComPtr<IDxcBlob> rayTracingShaderLib;
    ComPtr<ID3D12RootSignature> rayGenSignature;
    ComPtr<ID3D12RootSignature> globalSignature;

    ComPtr<ID3D12StateObject> rtStateObject;
    ComPtr<ID3D12StateObjectProperties> rtStateObjectProps;

    ComPtr<ID3D12Resource> outputResource;
    ComPtr<ID3D12DescriptorHeap> srvUavHeap;

    ComPtr<ID3D12Resource> sbtStorage;
    ShaderBindingTable sbt;
};

DX12 dx;
DXR dxr;
Geometry triangle;
UINT currentFence = 0;
UINT currBackBuffer = 0;

inline ID3D12Resource* CreateBuffer(ID3D12Device* device,
                                    uint64_t size,
                                    D3D12_RESOURCE_FLAGS flags,
                                    D3D12_RESOURCE_STATES initState,
                                    const D3D12_HEAP_PROPERTIES& heapProps)
{
    D3D12_RESOURCE_DESC bufDesc = {};
    bufDesc.Alignment = 0;
    bufDesc.DepthOrArraySize = 1;
    bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufDesc.Flags = flags;
    bufDesc.Format = DXGI_FORMAT_UNKNOWN;
    bufDesc.Height = 1;
    bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bufDesc.MipLevels = 1;
    bufDesc.SampleDesc.Count = 1;
    bufDesc.SampleDesc.Quality = 0;
    bufDesc.Width = size;

    ID3D12Resource* pBuffer;
    HR(device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &bufDesc,
                                       initState, nullptr, IID_PPV_ARGS(&pBuffer)));
    return pBuffer;
}


ID3D12DescriptorHeap* CreateDescriptorHeap(UINT count)
{
    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {};
    descriptorHeapDesc.NumDescriptors = count;
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    descriptorHeapDesc.NodeMask = 0;

    ID3D12DescriptorHeap* heap = nullptr;
    dx.device->CreateDescriptorHeap(&descriptorHeapDesc, IID_PPV_ARGS(&heap));
    return heap;
}

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
    const D3D12_CPU_DESCRIPTOR_HANDLE baseAddr = dx.rtvHeap->GetCPUDescriptorHandleForHeapStart();
    const INT offset = currBackBuffer;
    const UINT descriptorSize = dx.rtvDescSize;
    return CD3DX12_CPU_DESCRIPTOR_HANDLE(baseAddr, offset, descriptorSize);
}

D3D12_CPU_DESCRIPTOR_HANDLE DepthStencilView()
{
    return dx.dsvHeap->GetCPUDescriptorHandleForHeapStart();
}

void CheckRaytracingSupport(ID3D12Device5* device)
{
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5 = {};
    HR(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5)));
    if (options5.RaytracingTier < D3D12_RAYTRACING_TIER_1_0)
    {
        throw std::runtime_error("Ray tracing not supported on device");
    }
}

void InitDX12(HWND hwnd)
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
    HR(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&device)));

    ID3D12Fence* fence = nullptr;
    HR(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));

    dx.dxgiFactory = dxgiFactory;
    dx.device = device;
    dx.fence = fence;

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

    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    HR(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue)));
    dx.commandQueue = commandQueue;

    ID3D12CommandAllocator* commandAllocator;
    HR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocator)));
    dx.commandAllocator = commandAllocator;

    // Create the command list
    ID3D12GraphicsCommandList4* commandList;
    HR(dx.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, dx.commandAllocator.Get(), nullptr, IID_PPV_ARGS(&commandList)));
    dx.commandList = commandList;

    // Command list needs to be closed before calling Reset
    dx.commandList->Close();

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
    // First parameter said it's a device, but in Dx12 you need actually a CommandQueue
    HR(dx.dxgiFactory->CreateSwapChain(dx.commandQueue.Get(), &sd, &swapChain));

    dx.swapChain = swapChain;

    dx.rtvDescSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    dx.dsvDescSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    dx.cbvSrvUavDescSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

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
                                          D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                          &c,
                                          IID_PPV_ARGS(&depthStencilBuffer)));

    dx.depthStencilBuffer = depthStencilBuffer;

    D3D12_CPU_DESCRIPTOR_HANDLE handle = dx.dsvHeap->GetCPUDescriptorHandleForHeapStart();
    dx.device->CreateDepthStencilView(dx.depthStencilBuffer.Get(), nullptr, handle); // nullptr only works when the buffer is not typeless

    dx.viewport.TopLeftX = 0;
    dx.viewport.TopLeftY = 0;
    dx.viewport.Width = FLOAT(dx.width);
    dx.viewport.Height = FLOAT(dx.height);
    dx.viewport.MinDepth = 0.0f;
    dx.viewport.MaxDepth = 1.0f;

    dx.scissorRect = { 0, 0, dx.width, dx.height };
}

void InitVertexBuffer()
{
    // Create the vertex buffer.
    Vertex triangleVertices[] =
    {
        { XMFLOAT3(0.0f, 0.5f, 0.0f),   XMFLOAT4(1.0f, 0.0f, 0.0f, 1.0f) },
        { XMFLOAT3(0.5f, -0.5f, 0.0f),  XMFLOAT4(0.0f, 1.0f, 0.0f, 1.0f) },
        { XMFLOAT3(-0.5f, -0.5f, 0.0f), XMFLOAT4(0.0f, 0.0f, 1.0f, 1.0f) }
    };

    const UINT vertexBufferSize = sizeof(triangleVertices);

    // An upload heap is used here for code simplicity and because there are very few vertices to actually transfer.

    CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize);
    CD3DX12_HEAP_PROPERTIES heapProp(D3D12_HEAP_TYPE_UPLOAD);
    HR(dx.device->CreateCommittedResource(&heapProp,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, // init state
        nullptr, // init value
        IID_PPV_ARGS(&triangle.vertexBuffer)));

    // Copy the triangle data to the vertex buffer.
    UINT8* data = 0;
    CD3DX12_RANGE readRange(0, 0); // We do not intend to read from this resource on the CPU.
    HR(triangle.vertexBuffer->Map(0, &readRange, (void**)&data));
    {
        memcpy(data, triangleVertices, vertexBufferSize);
    }
    triangle.vertexBuffer->Unmap(0, nullptr);
    triangle.bufferSize = vertexBufferSize;
}

void CreateTopLevelAccelerationStructures(const std::vector<ASInstance>& instances)
{
    const UINT numInstance = UINT(instances.size());

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags =
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE |
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS prebuildDesc = {};
    prebuildDesc.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    prebuildDesc.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    prebuildDesc.NumDescs = numInstance;
    prebuildDesc.Flags = buildFlags;

    // Building the acceleration structure (AS) requires some scratch space, as well as space to store the resulting structure
    // This function computes a conservative estimate of the memory requirements for both based on the number of bottom-level instances.
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info = {};
    dx.device->GetRaytracingAccelerationStructurePrebuildInfo(&prebuildDesc, &info);

    const UINT64 scratchSize = ROUND_UP(info.ScratchDataSizeInBytes, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
    const UINT64 resultSize = ROUND_UP(info.ResultDataMaxSizeInBytes, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
    const UINT64 updateSize = ROUND_UP(info.UpdateScratchDataSizeInBytes, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
    const UINT64 instanceDescsSize = ROUND_UP(sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * UINT64(numInstance), D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);

    std::stringstream sout;
    sout << "TLAS Scratch Size=" << scratchSize << " bytes\n"
         << "TLAS Update Size=" << updateSize << " bytes\n"
         << "TLAS Dest Size=" << resultSize << " bytes\n";
    OutputDebugStringA(sout.str().c_str());

    // Create the scratch and result buffers. Since the build is all done on GPU,
    // those can be allocated on the default heap
    dxr.topLevelASBuffers.scratch = CreateBuffer(
        dx.device.Get(), scratchSize,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT));

    dxr.topLevelASBuffers.result = CreateBuffer(
        dx.device.Get(), resultSize,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT));

    // The buffer describing the instances: ID, shader binding information, matrices ...
    // Those will be copied into the buffer by the helper through mapping,
    // so the buffer has to be allocated on the upload heap.
    dxr.topLevelASBuffers.instanceDesc = CreateBuffer(
        dx.device.Get(),
        instanceDescsSize,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD));

    // After all the buffers are allocated, or if only an update is required, we can build the acceleration structure.
    // Note that in the case of the update, we also pass the existing AS as the 'previous' AS, so that it can be refitted in place.

    ComPtr<ID3D12Resource> descriptorsBuffer = dxr.topLevelASBuffers.instanceDesc;

    // Copy the descriptors in the target descriptor buffer
    D3D12_RAYTRACING_INSTANCE_DESC* instanceDescs;
    descriptorsBuffer->Map(0, nullptr, (void**)(&instanceDescs));
    if (!instanceDescs)
    {
        throw std::logic_error("Cannot map the instance descriptor buffer - is it in the upload heap?");
    }

    // Initialize the memory to zero on the first time only
    ZeroMemory(instanceDescs, instanceDescsSize);

    // Create the description for each instance
    for (uint32_t i = 0; i < numInstance; i++)
    {
        instanceDescs[i].InstanceID = instances[i].instanceID; //visible in the shader in InstanceID()
        instanceDescs[i].InstanceContributionToHitGroupIndex = instances[i].hitGroupIndex; // Index of the hit group invoked upon intersection
        instanceDescs[i].Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE; // Instance flags, including backface culling, winding etc.
        instanceDescs[i].AccelerationStructure = instances[i].bottomLevelResult->GetGPUVirtualAddress(); // Get access to the bottom level
        instanceDescs[i].InstanceMask = 0xFF; // Visibility mask, always visible here

        DirectX::XMMATRIX m = XMMatrixTranspose(instances[i].transform); // GLM is column major, the INSTANCE_DESC is row major
        memcpy(instanceDescs[i].Transform, &m, sizeof(instanceDescs[i].Transform));
    }
    descriptorsBuffer->Unmap(0, nullptr);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
    buildDesc.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    buildDesc.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    buildDesc.Inputs.InstanceDescs = descriptorsBuffer->GetGPUVirtualAddress();
    buildDesc.Inputs.NumDescs = numInstance;
    buildDesc.DestAccelerationStructureData = dxr.topLevelASBuffers.result->GetGPUVirtualAddress();
    buildDesc.ScratchAccelerationStructureData = dxr.topLevelASBuffers.scratch->GetGPUVirtualAddress();
    buildDesc.SourceAccelerationStructureData = 0;
    buildDesc.Inputs.Flags = buildFlags;

    // Build the top-level AS
    dx.commandList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);

    // Wait for the builder to complete by setting a barrier on the resulting buffer.
    D3D12_RESOURCE_BARRIER uavBarrier = {};
    uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarrier.UAV.pResource = dxr.topLevelASBuffers.result.Get();
    uavBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    dx.commandList->ResourceBarrier(1, &uavBarrier);

    dx.commandList->Close();
    ID3D12CommandList* ppCommandLists[] = { dx.commandList.Get() };
    dx.commandQueue->ExecuteCommandLists(1, ppCommandLists);

    FlushCommandQueue();

    HR(dx.commandList->Reset(dx.commandAllocator.Get(), nullptr));
}

ASInstance CreateBottomLevelAccelerationStructures()
{
    // Vertex buffer & index buffer
    const UINT vertexSizeInBytes = sizeof(Vertex);
    const UINT vertexCount = 3;

    ID3D12Resource* vertexBuffer = triangle.vertexBuffer.Get();

    D3D12_RAYTRACING_GEOMETRY_DESC geoDescArray[1];
    geoDescArray[0] = {};
    geoDescArray[0].Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geoDescArray[0].Triangles.VertexBuffer.StartAddress = vertexBuffer->GetGPUVirtualAddress();
    geoDescArray[0].Triangles.VertexBuffer.StrideInBytes = vertexSizeInBytes;
    geoDescArray[0].Triangles.VertexCount = vertexCount;
    geoDescArray[0].Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geoDescArray[0].Triangles.IndexBuffer = 0;
    geoDescArray[0].Triangles.IndexFormat = DXGI_FORMAT_UNKNOWN;
    geoDescArray[0].Triangles.IndexCount = 0;
    geoDescArray[0].Triangles.Transform3x4 = 0;
    geoDescArray[0].Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;

    static_assert(ARRAYSIZE(geoDescArray) == 1, "One geometry!");

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags =
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE |
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;

    // Compute the size of scratch buffer and resulting buffer.
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs;
    inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    inputs.Flags = buildFlags;
    inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputs.NumDescs = ARRAYSIZE(geoDescArray);
    inputs.pGeometryDescs = geoDescArray;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info = {};
    dx.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &info);

    const UINT scratchBufferSize = ROUND_UP(info.ScratchDataSizeInBytes, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    const UINT destBufferSize = ROUND_UP(info.ResultDataMaxSizeInBytes, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    const UINT updateBufferSize = ROUND_UP(info.UpdateScratchDataSizeInBytes, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);

    // Create buffers based on the previously computed size
    ComPtr<ID3D12Resource> scratchBuffer;
    ComPtr<ID3D12Resource> destBuffer;
    scratchBuffer = CreateBuffer(dx.device.Get(),
                                 scratchBufferSize,
                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                 D3D12_RESOURCE_STATE_COMMON,
                                 CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT));

    destBuffer = CreateBuffer(dx.device.Get(),
                              destBufferSize,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                              D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                              CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT));

    // Fill the bottom level AS build desc
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build;
    build.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    build.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    build.Inputs.NumDescs = ARRAYSIZE(geoDescArray);
    build.Inputs.pGeometryDescs = geoDescArray;
    build.DestAccelerationStructureData = destBuffer->GetGPUVirtualAddress();
    build.ScratchAccelerationStructureData = scratchBuffer->GetGPUVirtualAddress();
    build.SourceAccelerationStructureData = 0;
    build.Inputs.Flags = buildFlags;

    // Build the AS
    dx.commandList->BuildRaytracingAccelerationStructure(&build, 0, nullptr);

    // Wait for the builder to complete.
    // This is particularly important as the construction of the top-level hierarchy is called right afterwards
    D3D12_RESOURCE_BARRIER uavBarrier = {};
    uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarrier.UAV.pResource = destBuffer.Get();
    uavBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    dx.commandList->ResourceBarrier(1, &uavBarrier);

    ASInstance ins;
    ins.bottomLevelResult = destBuffer;
    ins.bottomLevelScratch = scratchBuffer;
    ins.hitGroupIndex = 0;
    ins.instanceID = 0;
    ins.transform = XMMatrixIdentity();
    return ins;
}

void CreateAccelerationStructures()
{
    // Build the bottom AS from the Triangle vertex buffer
    ASInstance instance = CreateBottomLevelAccelerationStructures();

    // Just one instance for now
    dxr.instances.push_back(instance);
    CreateTopLevelAccelerationStructures(dxr.instances);
}

ComPtr<ID3D12RootSignature>
CreateRootSignature(const D3D12_ROOT_SIGNATURE_DESC& desc)
{
    ComPtr<ID3D12RootSignature> rootSig;

    ComPtr<ID3DBlob> blob;
    ComPtr<ID3DBlob> error;

    HR(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error));
    HR(dx.device->CreateRootSignature(1, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&rootSig)));

    if (error)
    {
        OutputDebugStringW((wchar_t*)error->GetBufferPointer());
    }
    return rootSig;
}

ComPtr<ID3D12RootSignature> CreateRayGenSignature()
{
    D3D12_ROOT_SIGNATURE_DESC desc = {};
    desc.NumParameters = 0;
    desc.pParameters = nullptr;
    desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;

    ComPtr<ID3D12RootSignature> rootSig = CreateRootSignature(desc);
    return rootSig;
}

void CreateRTGlobalRootSignature()
{
    D3D12_DESCRIPTOR_RANGE range[2];
    range[0].BaseShaderRegister = 0; // u0
    range[0].NumDescriptors = 1;
    range[0].RegisterSpace = 0;
    range[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    range[0].OffsetInDescriptorsFromTableStart = 0;

    range[1].BaseShaderRegister = 0; // t0-t1
    range[1].NumDescriptors = 2;
    range[1].RegisterSpace = 0;
    range[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    range[1].OffsetInDescriptorsFromTableStart = 1;

    D3D12_ROOT_PARAMETER param = {};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    param.DescriptorTable.NumDescriptorRanges = ARRAYSIZE(range);
    param.DescriptorTable.pDescriptorRanges = range;

    // Creation of the global root signature
    D3D12_ROOT_SIGNATURE_DESC rootDesc = {};
    rootDesc.NumParameters = 1;
    rootDesc.pParameters = &param;
    rootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> serialized;
    ComPtr<ID3DBlob> error;

    // Create the empty global root signature
    HR(D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &error));
    HR(dx.device->CreateRootSignature(0, // node mask
                                      serialized->GetBufferPointer(),
                                      serialized->GetBufferSize(),
                                      IID_PPV_ARGS(&dxr.globalSignature)));
}

void CreateRayTracingShaderAndSignatures()
{
    CreateRTGlobalRootSignature();

    dxr.rayTracingShaderLib = CompileShaderLibrary2(L"RayGen.hlsl");
    dxr.rayGenSignature = CreateRayGenSignature();
}

void CreateRaytracingPipeline()
{
    CD3DX12_STATE_OBJECT_DESC psoDesc(D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE);

    // 1. DXIL libraries
    CD3DX12_SHADER_BYTECODE rayGenByteCode(dxr.rayTracingShaderLib->GetBufferPointer(), dxr.rayTracingShaderLib->GetBufferSize());
    auto rayGenLib = psoDesc.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
    rayGenLib->SetDXILLibrary(&rayGenByteCode);
    rayGenLib->DefineExport(L"RayGen");
    rayGenLib->DefineExport(L"Miss");
    rayGenLib->DefineExport(L"ClosestHit");

    // 2. Hit Group x1
    auto hitGroup = psoDesc.CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();
    hitGroup->SetClosestHitShaderImport(L"ClosestHit");
    hitGroup->SetHitGroupExport(L"HitGroup");
    hitGroup->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);

    // 3. Shader config x1
    // Payload size & attribute size
    const UINT payloadSize = 4 * sizeof(float); // RGB + distance
    const UINT maxAttributeSize = 2 * sizeof(float); // barycentric coordinates (the default one)
    auto shaderConfig = psoDesc.CreateSubobject<CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
    shaderConfig->Config(payloadSize, maxAttributeSize);

    // 4. Local Root Signatures
    auto rayGenSignature = psoDesc.CreateSubobject<CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT>();
    rayGenSignature->SetRootSignature(dxr.rayGenSignature.Get());

    // 5. Associations
    auto associateRayGen = psoDesc.CreateSubobject<CD3DX12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT>();
    associateRayGen->SetSubobjectToAssociate(*rayGenSignature);
    associateRayGen->AddExport(L"RayGen");

    // 6. Global Root Signature
    auto globalRootSignature = psoDesc.CreateSubobject<CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
    globalRootSignature->SetRootSignature(dxr.globalSignature.Get());

    // 7. Pipeline config
    auto pipelineConfig = psoDesc.CreateSubobject<CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT>();
    pipelineConfig->Config(1); // maxRecursionDepth

    ID3D12StateObject* rtStateObject = nullptr;

    // Create the state object
    HR(dx.device->CreateStateObject(psoDesc, IID_PPV_ARGS(&rtStateObject)));

    dxr.rtStateObject = rtStateObject;

    HR(dxr.rtStateObject->QueryInterface(IID_PPV_ARGS(&dxr.rtStateObjectProps)));
}

void CreateRaytracingOutputBuffer()
{
    D3D12_RESOURCE_DESC resDesc = {};
    resDesc.DepthOrArraySize = 1;
    resDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    // The back-buffer is actually DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, but sRGB
    // formats cannot be used with UAVs. For accuracy we should convert to sRGB
    // ourselves in the shader
    resDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;

    resDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    resDesc.Width = CLIENT_WIDTH;
    resDesc.Height = CLIENT_HEIGHT;
    resDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    resDesc.MipLevels = 1;
    resDesc.SampleDesc.Count = 1;

    auto defaultHeapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    HR(dx.device->CreateCommittedResource(
        &defaultHeapProps, D3D12_HEAP_FLAG_NONE, &resDesc,
        D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr,
        IID_PPV_ARGS(&dxr.outputResource)));
}

void CreateRtHeapAndDescriptors()
{
    // Create a SRV/UAV/CBV descriptor heap.
    // We need 2 entries, 1 UAV for the ray-tracing output and 1 SRV for the TLAS
    dxr.srvUavHeap = CreateDescriptorHeap(3);

    auto baseAddress = dxr.srvUavHeap->GetCPUDescriptorHandleForHeapStart();

    // u0
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;

    CD3DX12_CPU_DESCRIPTOR_HANDLE uavHandle(baseAddress, 0, dx.cbvSrvUavDescSize);
    dx.device->CreateUnorderedAccessView(dxr.outputResource.Get(), nullptr, &uavDesc, uavHandle);

    // Add the Top Level AS SRV right after the ray-tracing output buffer
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc1;
    srvDesc1.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc1.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    srvDesc1.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc1.RaytracingAccelerationStructure.Location = dxr.topLevelASBuffers.result->GetGPUVirtualAddress();

    // t0
    CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle1(baseAddress, 1, dx.cbvSrvUavDescSize);
    dx.device->CreateShaderResourceView(nullptr, &srvDesc1, srvHandle1);

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc2 = {};
    srvDesc2.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc2.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srvDesc2.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc2.Buffer.FirstElement = 0; // rsc.vertexBuffer->GetGPUVirtualAddress();
    srvDesc2.Buffer.NumElements = 3;
    srvDesc2.Buffer.StructureByteStride = sizeof(Vertex);
    srvDesc2.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

    CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle2(baseAddress, 2, dx.cbvSrvUavDescSize);
    dx.device->CreateShaderResourceView(triangle.vertexBuffer.Get(), &srvDesc2, srvHandle2);
}

void CreateShaderBindingTable()
{
    const UINT shaderIdSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    const UINT shaderRecordAlignment = D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT;

    const UINT numRayGen = 1;
    const UINT numRayGenParameters = 0;
    const UINT numMiss = 1;
    const UINT numMissParameters = 0;
    const UINT numHitGroup = 1;
    const UINT numHitGroupParameters = 0;

    // A ShaderRecord is made of a shader ID and a set of parameters, taking 8 bytes each.
    const UINT rayGenRecordSize = ROUND_UP(shaderIdSize + 8 * numRayGenParameters, shaderRecordAlignment);
    const UINT missRecordSize = ROUND_UP(shaderIdSize + 8 * numMissParameters, shaderRecordAlignment);
    const UINT hitGroupRecordSize = ROUND_UP(shaderIdSize + 8 * numHitGroupParameters, shaderRecordAlignment);

    const UINT rayGenSectionSize = ROUND_UP(numRayGen * rayGenRecordSize, D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
    const UINT missSectionSize = ROUND_UP(numMiss * missRecordSize, D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
    const UINT hitGroupSectionSize = ROUND_UP(numHitGroup * hitGroupRecordSize, D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);

    const UINT tableSize = rayGenSectionSize + missSectionSize + hitGroupSectionSize;

    auto stateProperties = dxr.rtStateObjectProps.Get();

    std::vector<UINT8> tableData;
    tableData.resize(tableSize);

    UINT8* data = tableData.data();
    for (int i = 0; i < numRayGen; ++i)
    {
        const void* shaderId = stateProperties->GetShaderIdentifier(L"RayGen");
        memcpy(data, shaderId, sizeof(shaderId));

        data += rayGenRecordSize;
    }

    data = tableData.data() + rayGenSectionSize;

    for (int i = 0; i < numMiss; ++i)
    {
        const void* shaderId = stateProperties->GetShaderIdentifier(L"Miss");
        memcpy(data, shaderId, sizeof(shaderId));

        data += missRecordSize;
    }

    data = tableData.data() + rayGenSectionSize + missSectionSize;

    for (int i = 0; i < numHitGroup; ++i)
    {
        const void* shaderId = stateProperties->GetShaderIdentifier(L"HitGroup");
        memcpy(data, shaderId, sizeof(shaderId));

        data += hitGroupRecordSize;
    }

    dxr.sbtStorage = CreateBuffer(dx.device.Get(),
                                  tableSize,
                                  D3D12_RESOURCE_FLAG_NONE,
                                  D3D12_RESOURCE_STATE_GENERIC_READ,
                                  CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD));

    UINT8* pGpuData = nullptr;
    HRESULT hr = dxr.sbtStorage->Map(0, nullptr, (void**)&pGpuData);
    if (FAILED(hr))
    {
        throw std::logic_error("Could not map the shader binding table");
    }
    memcpy(pGpuData, tableData.data(), tableData.size());
    dxr.sbtStorage->Unmap(0, nullptr);

    dxr.sbt.resource = dxr.sbtStorage.Get();
    dxr.sbt.rayGenStartAddress = dxr.sbtStorage->GetGPUVirtualAddress();
    dxr.sbt.rayGenSectionSize = rayGenSectionSize;
    dxr.sbt.missStartAddress = dxr.sbtStorage->GetGPUVirtualAddress() + rayGenSectionSize;
    dxr.sbt.missSectionSize = missSectionSize;
    dxr.sbt.missStride = missRecordSize;
    dxr.sbt.hitGroupStartAddress = dxr.sbtStorage->GetGPUVirtualAddress() + rayGenSectionSize + missSectionSize;
    dxr.sbt.hitGroupSectionSize = hitGroupSectionSize;
    dxr.sbt.hitGroupStride = hitGroupRecordSize;
}

void InitDXR()
{
    dx.commandList->Reset(dx.commandAllocator.Get(), nullptr);

    CheckRaytracingSupport(dx.device.Get());
    CreateAccelerationStructures();

    CreateRayTracingShaderAndSignatures();
    CreateRaytracingPipeline();
    CreateRaytracingOutputBuffer();
    CreateRtHeapAndDescriptors();
    CreateShaderBindingTable();

    dx.commandList->Close();
}

void Draw(UINT64 frameNo)
{
    HR(dx.commandAllocator->Reset());
    HR(dx.commandList->Reset(dx.commandAllocator.Get(), nullptr));

    // This needs to be reset whenever the command list is reset.
    dx.commandList->RSSetViewports(1, &dx.viewport);
    dx.commandList->RSSetScissorRects(1, &dx.scissorRect);

    dx.commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(CurrentBackBuffer(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));
    {
        // Bind the descriptor heap giving access to the top-level acceleration structure, as well as the ray-tracing output
        std::vector<ID3D12DescriptorHeap*> heaps = { dxr.srvUavHeap.Get() };
        dx.commandList->SetDescriptorHeaps(UINT(heaps.size()), heaps.data());
        dx.commandList->SetComputeRootSignature(dxr.globalSignature.Get());
        dx.commandList->SetComputeRootDescriptorTable(0, CD3DX12_GPU_DESCRIPTOR_HANDLE(dxr.srvUavHeap->GetGPUDescriptorHandleForHeapStart(), 0, dx.cbvSrvUavDescSize));

        // On the last frame, the ray-tracing output was used as a copy source
        // to copy its contents into the render target.
        // Now we need to transition it to a UAV so that the shaders can write in it.
        auto transition = CD3DX12_RESOURCE_BARRIER::Transition(
            dxr.outputResource.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        dx.commandList->ResourceBarrier(1, &transition);

        // Setup the ray-tracing task
        D3D12_DISPATCH_RAYS_DESC desc = {};

        desc.RayGenerationShaderRecord.StartAddress = dxr.sbt.rayGenStartAddress;
        desc.RayGenerationShaderRecord.SizeInBytes = dxr.sbt.rayGenSectionSize;

        desc.MissShaderTable.StartAddress = dxr.sbt.missStartAddress;
        desc.MissShaderTable.SizeInBytes = dxr.sbt.missSectionSize;
        desc.MissShaderTable.StrideInBytes = dxr.sbt.missStride;

        desc.HitGroupTable.StartAddress = dxr.sbt.hitGroupStartAddress;
        desc.HitGroupTable.SizeInBytes = dxr.sbt.hitGroupSectionSize;
        desc.HitGroupTable.StrideInBytes = dxr.sbt.hitGroupStride;

        // Dimensions of the image to render, identical to a kernel launch dimension
        desc.Width = CLIENT_WIDTH;
        desc.Height = CLIENT_HEIGHT;
        desc.Depth = 1;

        dx.commandList->SetPipelineState1(dxr.rtStateObject.Get());
        dx.commandList->DispatchRays(&desc);

        // The ray-tracing output needs to be copied to the actual render target used for display.
        // For this, we need to transition the ray-tracing output from a UAV to a copy source,
        // and the render target buffer to a copy destination.
        // We can then do the actual copy, before transitioning the render target buffer into a render target,
        // that will be then used to display the image
        transition = CD3DX12_RESOURCE_BARRIER::Transition(
            dxr.outputResource.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
        dx.commandList->ResourceBarrier(1, &transition);

        transition = CD3DX12_RESOURCE_BARRIER::Transition(
            CurrentBackBuffer(),
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_DEST);
        dx.commandList->ResourceBarrier(1, &transition);

        dx.commandList->CopyResource(CurrentBackBuffer(), dxr.outputResource.Get());

        transition = CD3DX12_RESOURCE_BARRIER::Transition(
            CurrentBackBuffer(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_RENDER_TARGET);
        dx.commandList->ResourceBarrier(1, &transition);
    }

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

void Shutdown()
{

}

void KeyDown(WPARAM wparam)
{
    if (wparam == VK_SPACE)
    {
    }
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam)
{
    LRESULT result = 0;
    switch (msg)
    {
    case WM_KEYDOWN:
    {
        if (wparam == VK_ESCAPE) DestroyWindow(hwnd);
        else KeyDown(wparam);
        break;
    }
    case WM_DESTROY:
    {
        Shutdown();
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

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int)
{
    // Open a window
    int errorCode = 0;
    HWND hwnd = CreateMainWindow(hInstance, &errorCode);

    if (errorCode != 0)
        return errorCode;

    InitDX12(hwnd);
    InitVertexBuffer();
    InitDXR();

    UINT64 frameNo = 0;

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
        Draw(frameNo);
        frameNo++;
    }

    return 0;
}
