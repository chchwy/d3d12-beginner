
#include "pch.h"
#include "debug.h"
#include "shader.h"
//#include "DXRHelper.h"
#include "nv_helpers_dx12/TopLevelASGenerator.h"
#include "nv_helpers_dx12/BottomLevelASGenerator.h"
#include "nv_helpers_dx12/RaytracingPipelineGenerator.h"
#include "nv_helpers_dx12/RootSignatureGenerator.h"
#include "nv_helpers_dx12/ShaderBindingTableGenerator.h"

#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxcompiler.lib")
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

using Microsoft::WRL::ComPtr;
using DirectX::XMFLOAT3;
using DirectX::XMFLOAT4;
using DirectX::XMMatrixIdentity;

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
    ComPtr<ID3D12GraphicsCommandList4> commandList;
    ComPtr<ID3D12CommandAllocator> commandAllocator;
    ComPtr<IDXGISwapChain> swapChain;

    ComPtr<ID3D12RootSignature> rootSignature;
    ComPtr<ID3D12PipelineState> pipelineState;

    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    ComPtr<ID3D12DescriptorHeap> dsvHeap;

    ComPtr<ID3D12Resource> swapChainBuffers[2];
    ComPtr<ID3D12Resource> depthStencilBuffer;

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

struct AccelerationStructureBuffers
{
    ComPtr<ID3D12Resource> pScratch;      // Scratch memory for AS builder
    ComPtr<ID3D12Resource> pResult;       // Where the AS is
    ComPtr<ID3D12Resource> pInstanceDesc; // Hold the matrices of the instances
};

struct DXR
{
    ComPtr<ID3D12Resource> bottomLevelAS; // Storage for the bottom Level AS

    nv_helpers_dx12::TopLevelASGenerator topLevelASGenerator;
    AccelerationStructureBuffers topLevelASBuffers;

    std::vector<std::pair<ComPtr<ID3D12Resource>, DirectX::XMMATRIX>> instances;

    ComPtr<IDxcBlob> rayGenLibrary;
    ComPtr<IDxcBlob> hitLibrary;
    ComPtr<IDxcBlob> missLibrary;
    ComPtr<ID3D12RootSignature> rayGenSignature;
    ComPtr<ID3D12RootSignature> hitSignature;
    ComPtr<ID3D12RootSignature> missSignature;
    ComPtr<ID3D12StateObject> rtStateObject;
    ComPtr<ID3D12StateObjectProperties> rtStateObjectProps;

    ComPtr<ID3D12Resource> outputResource;
    ComPtr<ID3D12DescriptorHeap> srvUavHeap;

    nv_helpers_dx12::ShaderBindingTableGenerator sbtHelper;
    ComPtr<ID3D12Resource> sbtStorage;
};

DX12 dx;
DXR dxr;
Resource rsc;
UINT currentFence = 0;
UINT currBackBuffer = 0;
bool raster = true;

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

void CheckRaytracingSupport(ID3D12Device5* device)
{
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5 = {};
    HR(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5,
                                   &options5, sizeof(options5)));
    if (options5.RaytracingTier < D3D12_RAYTRACING_TIER_1_0)
        throw std::runtime_error("Ray tracing not supported on device");
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
    HR(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&device)));

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
    rtvDesc.NumDescriptors = NUM_BACK_BUFFER;
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

    // Create the command list
    ID3D12GraphicsCommandList4* commandList;
    HR(dx.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, dx.commandAllocator.Get(), dx.pipelineState.Get(), IID_PPV_ARGS(&commandList)));
    dx.commandList = commandList;

    // Create the vertex buffer.
    {
        // Define the geometry for a triangle.
        Vertex triangleVertices[] =
        {
            { XMFLOAT3(0.0f, 0.5f, 0.0f),   XMFLOAT4(1.0f, 0.0f, 0.0f, 1.0f) },
            { XMFLOAT3(0.5f, -0.5f, 0.0f),  XMFLOAT4(0.0f, 1.0f, 0.0f, 1.0f) },
            { XMFLOAT3(-0.5f, -0.5f, 0.0f), XMFLOAT4(0.0f, 0.0f, 1.0f, 1.0f) }
        };

        const UINT vertexBufferSize = sizeof(triangleVertices);

        // Note: using upload heaps to transfer static data like vertex buffers is not recommended.
        // Every time the GPU needs it, the upload heap will be marshalled over.
        // Please read up on Default Heap usage.
        // An upload heap is used here for code simplicity and because there are very few vertices to actually transfer.
        CD3DX12_HEAP_PROPERTIES heapProperties(D3D12_HEAP_TYPE_UPLOAD);
        CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize);

        HR(dx.device->CreateCommittedResource(
            &heapProperties,
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

    // Command list needs to be closed before calling Reset
    commandList->Close();

    // Resource barrier for depth stencil buffer
    commandList->Reset(dx.commandAllocator.Get(), dx.pipelineState.Get());

    CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(dx.depthStencilBuffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    dx.commandList->ResourceBarrier(1, &barrier);
    dx.commandList->Close(); // done recording

    ID3D12CommandList* cmdLists[] = { dx.commandList.Get() };
    dx.commandQueue->ExecuteCommandLists(1, cmdLists);

    FlushCommandQueue();
}

void Draw()
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

    // Specify the buffers we are going to render to.
    dx.commandList->OMSetRenderTargets(1, &CurrentBackBufferView(), true, &DepthStencilView());

    if (raster)
    {
        // Clear back buffer & depth stencil buffer
        const float clearColor[]{ 67 / 255.f, 183 / 255.f, 194 / 255.f, 1.0f };
        dx.commandList->ClearRenderTargetView(CurrentBackBufferView(), clearColor, 0, nullptr);
        dx.commandList->ClearDepthStencilView(DepthStencilView(), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 0, 0, nullptr);

        // Draw the triangle
        dx.commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        dx.commandList->IASetVertexBuffers(0, 1, &rsc.vertexBufferView);
        dx.commandList->DrawInstanced(3, 1, 0, 0);
    }
    else
    {
        // #DXR
        // Bind the descriptor heap giving access to the top-level acceleration
        // structure, as well as the ray-tracing output
        std::vector<ID3D12DescriptorHeap*> heaps = { dxr.srvUavHeap.Get() };
        dx.commandList->SetDescriptorHeaps(UINT(heaps.size()), heaps.data());

        // On the last frame, the ray-tracing output was used as a copy source, to
        // copy its contents into the render target. Now we need to transition it to
        // a UAV so that the shaders can write in it.
        auto transition = CD3DX12_RESOURCE_BARRIER::Transition(
            dxr.outputResource.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        dx.commandList->ResourceBarrier(1, &transition);

        // Setup the ray-tracing task
        D3D12_DISPATCH_RAYS_DESC desc = {};

        // The layout of the SBT is as follows: ray generation shader, miss
        // shaders, hit groups. As described in the CreateShaderBindingTable method,
        // all SBT entries of a given type have the same size to allow a fixed
        // stride.
        // The ray generation shaders are always at the beginning of the SBT.
        uint32_t rayGenerationSectionSizeInBytes = dxr.sbtHelper.GetRayGenSectionSize();
        desc.RayGenerationShaderRecord.StartAddress = dxr.sbtStorage->GetGPUVirtualAddress();
        desc.RayGenerationShaderRecord.SizeInBytes = rayGenerationSectionSizeInBytes;

        // The miss shaders are in the second SBT section, right after the ray
        // generation shader. We have one miss shader for the camera rays and one
        // for the shadow rays, so this section has a size of 2*m_sbtEntrySize. We
        // also indicate the stride between the two miss shaders, which is the size
        // of a SBT entry
        uint32_t missSectionSizeInBytes = dxr.sbtHelper.GetMissSectionSize();
        desc.MissShaderTable.StartAddress =
            dxr.sbtStorage->GetGPUVirtualAddress() + rayGenerationSectionSizeInBytes;
        desc.MissShaderTable.SizeInBytes = missSectionSizeInBytes;
        desc.MissShaderTable.StrideInBytes = dxr.sbtHelper.GetMissEntrySize();

        // The hit groups section start after the miss shaders. In this sample we
        // have one 1 hit group for the triangle
        uint32_t hitGroupsSectionSize = dxr.sbtHelper.GetHitGroupSectionSize();
        desc.HitGroupTable.StartAddress = dxr.sbtStorage->GetGPUVirtualAddress()
            + rayGenerationSectionSizeInBytes
            + missSectionSizeInBytes;

        desc.HitGroupTable.SizeInBytes = hitGroupsSectionSize;
        desc.HitGroupTable.StrideInBytes = dxr.sbtHelper.GetHitGroupEntrySize();

        // Dimensions of the image to render, identical to a kernel launch dimension
        desc.Width = CLIENT_WIDTH;
        desc.Height = CLIENT_HEIGHT;
        desc.Depth = 1;

        // Bind the ray-tracing pipeline
        dx.commandList->SetPipelineState1(dxr.rtStateObject.Get());
        // Dispatch the rays and write to the ray-tracing output
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

// Create the acceleration structure of an instance
AccelerationStructureBuffers
CreateBottomLevelAS(std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> vVertexBuffers)
{
    nv_helpers_dx12::BottomLevelASGenerator bottomLevelAS;

    // Adding all vertex buffers and not transforming their position.
    for (const auto &bufferPair : vVertexBuffers)
    {
        ComPtr<ID3D12Resource> buffer = bufferPair.first;
        uint32_t numVertices = bufferPair.second;
        bottomLevelAS.AddVertexBuffer(buffer.Get(), 0, numVertices,
                                      sizeof(Vertex), 0, 0);
    }

    // The AS build requires some scratch space to store temporary information.
    // The amount of scratch memory is dependent on the scene complexity.
    UINT64 scratchSizeInBytes = 0;
    // The final AS also needs to be stored in addition to the existing vertex
    // buffers. It size is also dependent on the scene complexity.
    UINT64 resultSizeInBytes = 0;

    bottomLevelAS.ComputeASBufferSizes(dx.device.Get(), false, &scratchSizeInBytes,
                                       &resultSizeInBytes);

    // Once the sizes are obtained, the application is responsible for allocating
    // the necessary buffers. Since the entire generation will be done on the GPU,
    // we can directly allocate those on the default heap
    AccelerationStructureBuffers buffers;

    buffers.pScratch = nv_helpers_dx12::CreateBuffer(
        dx.device.Get(), scratchSizeInBytes,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON,
        nv_helpers_dx12::kDefaultHeapProps);

    buffers.pResult = nv_helpers_dx12::CreateBuffer(
        dx.device.Get(), resultSizeInBytes,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        nv_helpers_dx12::kDefaultHeapProps);

    // Build the acceleration structure. Note that this call integrates a barrier
    // on the generated AS, so that it can be used to compute a top-level AS right
    // after this method.
    bottomLevelAS.Generate(dx.commandList.Get(),
                           buffers.pScratch.Get(),
                           buffers.pResult.Get(),
                           false,
                           nullptr);

    return buffers;
}

// Create the main acceleration structure that holds
void CreateTopLevelAS(const std::vector<std::pair<ComPtr<ID3D12Resource>, DirectX::XMMATRIX>>& instances)
{
    // Gather all the instances into the builder helper
    for (size_t i = 0; i < instances.size(); i++)
    {
        dxr.topLevelASGenerator.AddInstance(instances[i].first.Get(), instances[i].second, UINT(i), UINT(0));
    }

    // As for the bottom-level AS, the building the AS requires some scratch space
    // to store temporary data in addition to the actual AS. In the case of the
    // top-level AS, the instance descriptors also need to be stored in GPU
    // memory. This call outputs the memory requirements for each (scratch,
    // results, instance descriptors) so that the application can allocate the
    // corresponding memory
    UINT64 scratchSize = 0;
    UINT64 resultSize = 0;
    UINT64 instanceDescsSize = 0;

    dxr.topLevelASGenerator.ComputeASBufferSizes(dx.device.Get(), true,
                                                 &scratchSize,
                                                 &resultSize,
                                                 &instanceDescsSize);

    // Create the scratch and result buffers. Since the build is all done on GPU,
    // those can be allocated on the default heap
    dxr.topLevelASBuffers.pScratch = nv_helpers_dx12::CreateBuffer(
        dx.device.Get(), scratchSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nv_helpers_dx12::kDefaultHeapProps);

    dxr.topLevelASBuffers.pResult = nv_helpers_dx12::CreateBuffer(
        dx.device.Get(),
        resultSize,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        nv_helpers_dx12::kDefaultHeapProps);

    // The buffer describing the instances: ID, shader binding information,
    // matrices ... Those will be copied into the buffer by the helper through
    // mapping, so the buffer has to be allocated on the upload heap.
    dxr.topLevelASBuffers.pInstanceDesc = nv_helpers_dx12::CreateBuffer(
        dx.device.Get(),
        instanceDescsSize,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nv_helpers_dx12::kUploadHeapProps);

    // After all the buffers are allocated, or if only an update is required,
    // we can build the acceleration structure. Note that in the case of the update
    // we also pass the existing AS as the 'previous' AS, so that it can be refitted in place.
    dxr.topLevelASGenerator.Generate(dx.commandList.Get(),
                                     dxr.topLevelASBuffers.pScratch.Get(),
                                     dxr.topLevelASBuffers.pResult.Get(),
                                     dxr.topLevelASBuffers.pInstanceDesc.Get());
}

// Create all acceleration structures, bottom and top
void CreateAccelerationStructures()
{
    // Build the bottom AS from the Triangle vertex buffer
    AccelerationStructureBuffers bottomLevelBuffers =
        CreateBottomLevelAS({ std::make_pair(rsc.vertexBuffer.Get(), 3) });

    // Just one instance for now
    dxr.instances = { std::make_pair(bottomLevelBuffers.pResult, XMMatrixIdentity()) };
    CreateTopLevelAS(dxr.instances);

    // Flush the command list and wait for it to finish
    dx.commandList->Close();
    ID3D12CommandList* ppCommandLists[] = { dx.commandList.Get() };
    dx.commandQueue->ExecuteCommandLists(1, ppCommandLists);

    FlushCommandQueue();

    // Once the command list is finished executing, reset it to be reused for rendering
    HR(dx.commandList->Reset(dx.commandAllocator.Get(), dx.pipelineState.Get()));

    // Store the AS buffers. The rest of the buffers will be released once we exit the function
    dxr.bottomLevelAS = bottomLevelBuffers.pResult;
}

ComPtr<ID3D12RootSignature> CreateRayGenSignature()
{
    nv_helpers_dx12::RootSignatureGenerator rsg;
    rsg.AddHeapRangesParameter(
        {
            {
                0 /*u0*/,
                1 /*1 descriptor */,
                0 /*use the implicit register space 0*/,
                D3D12_DESCRIPTOR_RANGE_TYPE_UAV /* UAV representing the output buffer*/,
                0 /*heap slot where the UAV is defined*/
            },
            {
                0 /*t0*/,
                1,
                0,
                D3D12_DESCRIPTOR_RANGE_TYPE_SRV /*Top-level acceleration structure*/,
                1
            }
        });
    return rsg.Generate(dx.device.Get(), true);
}

ComPtr<ID3D12RootSignature> CreateMissSignature()
{
    nv_helpers_dx12::RootSignatureGenerator rsc;
    return rsc.Generate(dx.device.Get(), true);
}

ComPtr<ID3D12RootSignature> CreateHitSignature()
{
    nv_helpers_dx12::RootSignatureGenerator rsc;
    rsc.AddRootParameter(D3D12_ROOT_PARAMETER_TYPE_SRV);
    return rsc.Generate(dx.device.Get(), true);
}

void CreateRaytracingPipeline()
{
    nv_helpers_dx12::RayTracingPipelineGenerator pipeline(dx.device.Get());

    dxr.rayGenLibrary = CompileShaderLibrary2(L"RayGen.hlsl");
    dxr.missLibrary = CompileShaderLibrary2(L"Miss.hlsl");
    dxr.hitLibrary = CompileShaderLibrary2(L"Hit.hlsl");

    pipeline.AddLibrary(dxr.rayGenLibrary.Get(), { L"RayGen" });
    pipeline.AddLibrary(dxr.missLibrary.Get(), { L"Miss" });
    pipeline.AddLibrary(dxr.hitLibrary.Get(), { L"ClosestHit" });

    dxr.rayGenSignature = CreateRayGenSignature();
    dxr.missSignature = CreateMissSignature();
    dxr.hitSignature = CreateHitSignature();

    pipeline.AddHitGroup(L"HitGroup", L"ClosestHit");

    pipeline.AddRootSignatureAssociation(dxr.rayGenSignature.Get(), { L"RayGen" });
    pipeline.AddRootSignatureAssociation(dxr.missSignature.Get(), { L"Miss" });
    pipeline.AddRootSignatureAssociation(dxr.hitSignature.Get(), { L"HitGroup" });

    pipeline.SetMaxPayloadSize(4 * sizeof(float)); // RGB + distance
    pipeline.SetMaxAttributeSize(2 * sizeof(float)); // barycentric coordinates
    pipeline.SetMaxRecursionDepth(1);
    dxr.rtStateObject = pipeline.Generate();
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
    HR(dx.device->CreateCommittedResource(
        &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &resDesc,
        D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr,
        IID_PPV_ARGS(&dxr.outputResource)));
}

void CreateShaderResourceHeap()
{
    // Create a SRV/UAV/CBV descriptor heap. We need 2 entries
    // 1 UAV for the ray-tracing output and 1 SRV for the TLAS
    dxr.srvUavHeap = nv_helpers_dx12::CreateDescriptorHeap(
        dx.device.Get(), 2, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true);

    D3D12_CPU_DESCRIPTOR_HANDLE srvHandle =
        dxr.srvUavHeap->GetCPUDescriptorHandleForHeapStart();

    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    dx.device->CreateUnorderedAccessView(dxr.outputResource.Get(), nullptr, &uavDesc, srvHandle);

    // Add the Top Level AS SRV right after the ray-tracing output buffer
    srvHandle.ptr += dx.cbvSrvUavDescSize;

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc;
    srvDesc.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.RaytracingAccelerationStructure.Location =
        dxr.topLevelASBuffers.pResult->GetGPUVirtualAddress();

    // Write the acceleration structure view in the heap
    dx.device->CreateShaderResourceView(nullptr, &srvDesc, srvHandle);
}

void CreateShaderBindingTable()
{
    // The SBT helper class collects calls to Add*Program.  If called several
    // times, the helper must be emptied before re-adding shaders.
    dxr.sbtHelper.Reset();

    // The pointer to the beginning of the heap is the only parameter required by
    // shaders without root parameters
    D3D12_GPU_DESCRIPTOR_HANDLE srvUavHeapHandle =
        dxr.srvUavHeap->GetGPUDescriptorHandleForHeapStart();

    // The helper treats both root parameter pointers and heap pointers as void*,
    // while DX12 uses the
    // D3D12_GPU_DESCRIPTOR_HANDLE to define heap pointers. The pointer in this
    // struct is a UINT64, which then has to be reinterpreted as a pointer.
    auto heapPointer = reinterpret_cast<UINT64 *>(srvUavHeapHandle.ptr);

    // The ray generation only uses heap data
    dxr.sbtHelper.AddRayGenerationProgram(L"RayGen", { heapPointer });

    // The miss and hit shaders do not access any external resources: instead they
    // communicate their results through the ray payload
    dxr.sbtHelper.AddMissProgram(L"Miss", {});

    // Adding the triangle hit shader
    dxr.sbtHelper.AddHitGroup(L"HitGroup",
                            { (void *)(rsc.vertexBuffer->GetGPUVirtualAddress()) });

    // Compute the size of the SBT given the number of shaders and their
    // parameters
    uint32_t sbtSize = dxr.sbtHelper.ComputeSBTSize();

    // Create the SBT on the upload heap. This is required as the helper will use
    // mapping to write the SBT contents. After the SBT compilation it could be
    // copied to the default heap for performance.
    dxr.sbtStorage = nv_helpers_dx12::CreateBuffer(dx.device.Get(), sbtSize,
                                                   D3D12_RESOURCE_FLAG_NONE,
                                                   D3D12_RESOURCE_STATE_GENERIC_READ,
                                                   nv_helpers_dx12::kUploadHeapProps);
    if (!dxr.sbtStorage)
    {
        throw std::logic_error("Could not allocate the shader binding table");
    }
    // Compile the SBT from the shader and parameters info
    dxr.sbtHelper.Generate(dxr.sbtStorage.Get(), dxr.rtStateObjectProps.Get());
}

void InitDXR()
{
    dx.commandList->Reset(dx.commandAllocator.Get(), dx.pipelineState.Get());

    CheckRaytracingSupport(dx.device.Get());
    CreateAccelerationStructures();
    
    dx.commandList->Close();

    CreateRaytracingPipeline();
    CreateRaytracingOutputBuffer();
    CreateShaderResourceHeap();
    CreateShaderBindingTable();
}

void KeyDown(WPARAM wparam)
{
    if (wparam == VK_SPACE)
    {
        raster = !raster;
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
        PostQuitMessage(0);
        break;
    }
    default:
        result = DefWindowProcW(hwnd, msg, wparam, lparam);
    }
    return result;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int)
{
    // Open a window
    HWND hwnd;
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
            return GetLastError();
        }

        RECT initialRect = { 0, 0, CLIENT_WIDTH, CLIENT_HEIGHT };
        AdjustWindowRectEx(&initialRect, WS_OVERLAPPEDWINDOW, FALSE, WS_EX_OVERLAPPEDWINDOW);
        LONG initialWidth = initialRect.right - initialRect.left;
        LONG initialHeight = initialRect.bottom - initialRect.top;

        hwnd = CreateWindowExW(WS_EX_OVERLAPPEDWINDOW,
                               winClass.lpszClassName,
                               L"02. Draw the first Triangle",
                               WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                               CW_USEDEFAULT, CW_USEDEFAULT,
                               initialWidth,
                               initialHeight,
                               0, 0, hInstance, 0);

        if (!hwnd)
        {
            MessageBoxA(0, "CreateWindowEx failed", "Fatal Error", MB_OK);
            return GetLastError();
        }
    }

    InitD3D(hwnd);
    InitPipeline();
    InitDXR();

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
        Draw();
    }

    return 0;
}