#include "pch.h"
#include "debug.h"

#pragma comment(lib,"d3dcompiler.lib")
#pragma comment(lib, "D3D12.lib")
#pragma comment(lib, "dxgi.lib")

using Microsoft::WRL::ComPtr;

const static UINT CLIENT_WIDTH = 1024;
const static UINT CLIENT_HEIGHT = 768;
const static DXGI_FORMAT BACK_BUFFER_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;
const static UINT NUM_BACK_BUFFER = 2;

#define ENABLE_DEBUG_LAYER

struct DX12
{
    ComPtr<IDXGIFactory> dxgiFactory;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12Fence> mFence;
    ComPtr<ID3D12CommandQueue> mCommandQueue;
    ComPtr<ID3D12GraphicsCommandList> mCommandList;
    ComPtr<ID3D12CommandAllocator> mCommandAllocator;
    ComPtr<IDXGISwapChain> mSwapChain;

    ComPtr<ID3D12DescriptorHeap> mRtvHeap;
    ComPtr<ID3D12DescriptorHeap> mDsvHeap;

    ComPtr<ID3D12Resource> mSwapChainBuffers[2];
    ComPtr<ID3D12Resource> mDepthStencilBuffer;

    D3D12_VIEWPORT mViewport;
    D3D12_RECT mScissorRect;

    int mWidth = 0;
    int mHeight = 0;
    int mMSAAQuality = 0;

    UINT mRtvDescSize = 0;
    UINT mDsvDescSize = 0;
    UINT mCbvSrvUavDescSize = 0;

    UINT mCurrentFence = 0;
    UINT mCurrBackBuffer = 0;
};

DX12 dx;

void FlushCommandQueue()
{
    dx.mCurrentFence += 1;

    // Add an instruction to the command queue to set a new fence point.  Because we
    // are on the GPU timeline, the new fence point won't be set until the GPU finishes
    // processing all the commands prior to this Signal().
    HR(dx.mCommandQueue->Signal(dx.mFence.Get(), dx.mCurrentFence));

    if (dx.mFence->GetCompletedValue() < dx.mCurrentFence)
    {
        HANDLE eventHandle = CreateEventEx(nullptr, false, false, EVENT_ALL_ACCESS);

        dx.mFence->SetEventOnCompletion(dx.mCurrentFence, eventHandle);

        WaitForSingleObject(eventHandle, INFINITE);
        CloseHandle(eventHandle);
    }
}

ID3D12Resource* CurrentBackBuffer()
{
    return dx.mSwapChainBuffers[dx.mCurrBackBuffer].Get();
}

D3D12_CPU_DESCRIPTOR_HANDLE CurrentBackBufferView()
{
    return CD3DX12_CPU_DESCRIPTOR_HANDLE(
        dx.mRtvHeap->GetCPUDescriptorHandleForHeapStart(),
        dx.mCurrBackBuffer,
        dx.mRtvDescSize);
}

D3D12_CPU_DESCRIPTOR_HANDLE DepthStencilView()
{
    return dx.mDsvHeap->GetCPUDescriptorHandleForHeapStart();
}

void InitD3D(HWND hwnd)
{
#ifdef ENABLE_DEBUG_LAYER
    ID3D12Debug* debugController;
    D3D12GetDebugInterface(IID_PPV_ARGS(&debugController));
    debugController->EnableDebugLayer();
#endif

    IDXGIFactory4* dxgiFactory = nullptr;
    HR(CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)));

    ID3D12Device* device = nullptr;
    HR(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));

    ID3D12Fence* fence = nullptr;
    HR(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));

    dx.dxgiFactory = dxgiFactory;
    dx.device = device;
    dx.mFence = fence;

    dx.mRtvDescSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    dx.mDsvDescSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    dx.mCbvSrvUavDescSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS msaaQualityLevels;
    msaaQualityLevels.Format = BACK_BUFFER_FORMAT;
    msaaQualityLevels.SampleCount = 4;
    msaaQualityLevels.Flags = D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE;
    msaaQualityLevels.NumQualityLevels = 0;
    dx.device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &msaaQualityLevels, sizeof(msaaQualityLevels));

    dx.mMSAAQuality = msaaQualityLevels.NumQualityLevels;

    // Create CommandList
    ID3D12CommandQueue* commandQueue;
    ID3D12CommandAllocator* commandAllocator;
    ID3D12GraphicsCommandList* commandList;

    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    HR(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue)));
    HR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocator)));
    HR(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocator, nullptr, IID_PPV_ARGS(&commandList)));

    dx.mCommandQueue = commandQueue;
    dx.mCommandList = commandList;
    dx.mCommandAllocator = commandAllocator;

    // Start off in a closed state.
    // This is because the first time we refer to the command list we will Reset it
    // and it needs to be closed before calling Reset
    commandList->Close();

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
    sd.SampleDesc.Quality = dx.mMSAAQuality - 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = NUM_BACK_BUFFER;
    sd.OutputWindow = hwnd;
    sd.Windowed = true;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

    IDXGISwapChain* swapChain = nullptr;
    // First parameter said it's a device, but in Dx12 you need to pass a CommandQueue
    HR(dx.dxgiFactory->CreateSwapChain(dx.mCommandQueue.Get(), &sd, &swapChain));

    dx.mSwapChain = swapChain;

    // Create Heap
    D3D12_DESCRIPTOR_HEAP_DESC rtvDesc;
    rtvDesc.NumDescriptors = NUM_BACK_BUFFER;
    rtvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    rtvDesc.NodeMask = 0;
    HR(device->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(&dx.mRtvHeap)));

    D3D12_DESCRIPTOR_HEAP_DESC dsvDesc;
    dsvDesc.NumDescriptors = 1;
    dsvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    dsvDesc.NodeMask = 0;
    HR(device->CreateDescriptorHeap(&dsvDesc, IID_PPV_ARGS(&dx.mDsvHeap)));

    // Back buffer Render Target View
    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHeapHandle(dx.mRtvHeap->GetCPUDescriptorHandleForHeapStart());
    for (int i = 0; i < NUM_BACK_BUFFER; ++i)
    {
        HR(dx.mSwapChain->GetBuffer(i, IID_PPV_ARGS(&dx.mSwapChainBuffers[i])));

        dx.device->CreateRenderTargetView(dx.mSwapChainBuffers[i].Get(), nullptr, rtvHeapHandle);
        rtvHeapHandle.Offset(1, dx.mRtvDescSize);
    }

    dx.mCurrBackBuffer = 0;

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
    d.SampleDesc.Quality = dx.mMSAAQuality - 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

    D3D12_CLEAR_VALUE c;
    c.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    c.DepthStencil.Depth = 1.0f;
    c.DepthStencil.Stencil = 0;


    ID3D12Resource* depthStencilBuffer = nullptr;
    CD3DX12_HEAP_PROPERTIES heapProp(D3D12_HEAP_TYPE_DEFAULT);
    HR(dx.device->CreateCommittedResource(&heapProp, D3D12_HEAP_FLAG_NONE,
                                        &d, D3D12_RESOURCE_STATE_COMMON, &c,
                                        IID_PPV_ARGS(&depthStencilBuffer)));

    dx.mDepthStencilBuffer = depthStencilBuffer;

    D3D12_CPU_DESCRIPTOR_HANDLE handle = dx.mDsvHeap->GetCPUDescriptorHandleForHeapStart();
    dx.device->CreateDepthStencilView(dx.mDepthStencilBuffer.Get(), nullptr, handle); // nullptr only works when the buffer is not typeless

    CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(dx.mDepthStencilBuffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    dx.mCommandList->ResourceBarrier(1, &barrier);

    dx.mCommandList->Close();

    ID3D12CommandList* cmdLists[] = { dx.mCommandList.Get() };
    dx.mCommandQueue->ExecuteCommandLists(1, cmdLists);
}

void Render()
{
    // We can only reset when the associated command lists have finished execution on the GPU.
    HR(dx.mCommandAllocator->Reset());

    // A command list can be reset after it has been added to the command queue via ExecuteCommandList.
    // Reusing the command list reuses memory.
    HR(dx.mCommandList->Reset(dx.mCommandAllocator.Get(), nullptr));

    dx.mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(CurrentBackBuffer(),
                                                                              D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));
    // This needs to be reset whenever the command list is reset.
    dx.mCommandList->RSSetViewports(1, &dx.mViewport);
    dx.mCommandList->RSSetScissorRects(1, &dx.mScissorRect);

    // Clear back buffer & depth stencil buffer
    FLOAT red[]{ 67 / 255.f, 183 / 255.f, 194 / 255.f, 1.0f };
    dx.mCommandList->ClearRenderTargetView(CurrentBackBufferView(), red, 0, nullptr);
    dx.mCommandList->ClearDepthStencilView(DepthStencilView(), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 0, 0, nullptr);

    // Specify the buffers we are going to render to.
    dx.mCommandList->OMSetRenderTargets(1, &CurrentBackBufferView(), true, &DepthStencilView());

    // Indicate a state transition on the resource usage.
    dx.mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(CurrentBackBuffer(),
                                                                           D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));

    HR(dx.mCommandList->Close()); // Done recording commands.

    // Execute command list
    ID3D12CommandList* cmdsLists[] = { dx.mCommandList.Get() };
    dx.mCommandQueue->ExecuteCommandLists(_countof(cmdsLists), cmdsLists);

    HR(dx.mSwapChain->Present(0, 0));
    dx.mCurrBackBuffer = (dx.mCurrBackBuffer + 1) % NUM_BACK_BUFFER;

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

        RECT initialRect = { 0, 0, 1024, 768 };
        AdjustWindowRectEx(&initialRect, WS_OVERLAPPEDWINDOW, FALSE, WS_EX_OVERLAPPEDWINDOW);
        LONG initialWidth = initialRect.right - initialRect.left;
        LONG initialHeight = initialRect.bottom - initialRect.top;

        hwnd = CreateWindowExW(WS_EX_OVERLAPPEDWINDOW,
                               winClass.lpszClassName,
                               L"01. Initialize D3D12",
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