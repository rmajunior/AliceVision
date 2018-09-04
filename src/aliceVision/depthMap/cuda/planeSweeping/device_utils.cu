// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

namespace aliceVision {
namespace depthMap {

// Helper functions

clock_t tic()
{
    return clock();
}

// returns the ms passed after last call to tic()
float toc(clock_t ticClk)
{
    return (float)((clock() - ticClk) * 1000.0 / CLOCKS_PER_SEC);
}

template <typename T>
class BufPtr
{
public:
    __host__ __device__
    BufPtr( T* ptr, int pitch )
        : _ptr( (unsigned char*)ptr )
        , _pitch( pitch )
    { }

    __host__ __device__
    inline T*       ptr()       { return (T*)      _ptr; }
    __host__ __device__
    inline const T* ptr() const { return (const T*)_ptr; }

    __host__ __device__
    inline T*       row( int y )       { return (T*)      (_ptr + y * _pitch); }
    __host__ __device__
    inline const T* row( int y ) const { return (const T*)(_ptr + y * _pitch); }

    __host__ __device__
    inline T&       at( int x, int y )       { return row(y)[x]; }
    __host__ __device__
    inline const T& at( int x, int y ) const { return row(y)[x]; }
private:
    BufPtr( );
    BufPtr( const BufPtr& );
    BufPtr& operator*=( const BufPtr& );

    unsigned char* const _ptr;
    const int            _pitch;
};

/**
* @brief
* @param[int] ptr
* @param[int] pitch raw length of a line in bytes
* @param[int] x
* @param[int] y
* @return
*/
template <typename T>
__device__ T* get2DBufferAt(T* ptr, int pitch, int x, int y)
{
    return &(BufPtr<T>(ptr,pitch).at(x,y));
    // return ((T*)(((char*)ptr) + y * pitch)) + x;
}

/**
* @brief
* @param[int] ptr
* @param[int] spitch raw length of a 2D array in bytes
* @param[int] pitch raw length of a line in bytes
* @param[int] x
* @param[int] y
* @return
*/
template <typename T>
__device__ T* get3DBufferAt(T* ptr, int spitch, int pitch, int x, int y, int z)
{

    return ((T*)(((char*)ptr) + z * spitch + y * pitch)) + x;
}

/*

// function clamping x between a and b

__device__ int clamp(int x, int a, int b){

return fmaxf(a, fminf(b,x));

}

*/

} // namespace depthMap
} // namespace aliceVision
