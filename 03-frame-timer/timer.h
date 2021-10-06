#pragma once

class GameTimer
{
public:
    GameTimer()
    {
        LARGE_INTEGER countsPerSec;
        QueryPerformanceFrequency(&countsPerSec);
        mSecondsPerCount = 1.0 / double(countsPerSec.QuadPart);
    }

    void GameTimer::Reset()
    {
        LARGE_INTEGER currTime;
        QueryPerformanceCounter(&currTime);

        mBaseTime = currTime.QuadPart;
        mPrevTime = currTime.QuadPart;
    }

    void Tick()
    {
        LARGE_INTEGER currTime;
        QueryPerformanceCounter(&currTime);
        mCurrTime = currTime.QuadPart;

        // Time difference between this frame and the previous.
        mDeltaTime = (mCurrTime - mPrevTime) * mSecondsPerCount;

        // Prepare for next frame.
        mPrevTime = mCurrTime;

        // Force nonnegative.  The DXSDK's CDXUTTimer mentions that if the
        // processor goes into a power save mode or we get shuffled to another
        // processor, then mDeltaTime can be negative.
        if (mDeltaTime < 0.0)
        {
            mDeltaTime = 0.0;
        }
    }

    float TotalTime() const
    {
        return (float)((mCurrTime - mBaseTime) * mSecondsPerCount);
    }

    float DeltaTime() const
    {
        return (float)mDeltaTime;
    }

    double mSecondsPerCount = 0.0;
    double mDeltaTime = 0.0;
    int64_t mBaseTime = 0;
    int64_t mPrevTime = 0;
    int64_t mCurrTime = 0;
};