// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <aliceVision/system/Logger.hpp>

namespace aliceVision {
namespace depthMap {

template <unsigned Dim> class CudaSizeBase
{
public:
  CudaSizeBase()
  {
    #pragma unroll
    for(int i = Dim; i--;)
      size[i] = 0;
  }
  inline size_t operator[](size_t i) const { return size[i]; }
  inline size_t &operator[](size_t i) { return size[i]; }
  inline CudaSizeBase operator+(const CudaSizeBase<Dim> &s) const {
    CudaSizeBase<Dim> r;

    #pragma unroll
    for(size_t i = Dim; i--;)
      r[i] = (*this)[i] + s[i];

    return r;
  }
  inline CudaSizeBase operator-(const CudaSizeBase<Dim> &s) const {
    CudaSizeBase<Dim> r;

    #pragma unroll
    for(size_t i = Dim; i--;)
      r[i] = (*this)[i] - s[i];

    return r;
  }
  inline size_t getSize() const {
    size_t s = 1;

    #pragma unroll
    for(int i = Dim; i--;)
      s *= size[i];

    return s;
  }
protected:
  size_t size[Dim];
};

template <unsigned Dim>
class CudaSize: public CudaSizeBase<Dim>
{
    CudaSize() {}
};

template <>
class CudaSize<1>: public CudaSizeBase<1>
{
public:
    CudaSize() {}
    explicit CudaSize(size_t s0) { size[0] = s0; }
};

template <>
class CudaSize<2>: public CudaSizeBase<2>
{
public:
    CudaSize() {}
    CudaSize(size_t s0, size_t s1) { size[0] = s0; size[1] = s1; }
};

template <>
class CudaSize<3>: public CudaSizeBase<3>
{
public:
    CudaSize() {}
    CudaSize(size_t s0, size_t s1, size_t s2) { size[0] = s0; size[1] = s1; size[2] = s2; }
};

template <unsigned Dim>
bool operator==(const CudaSizeBase<Dim> &s1, const CudaSizeBase<Dim> &s2)
{
  for(int i = Dim; i--;)
    if(s1[i] != s2[i])
      return false;

  return true;
}

template <unsigned Dim>
bool operator!=(const CudaSizeBase<Dim> &s1, const CudaSizeBase<Dim> &s2)
{
  for(size_t i = Dim; i--;)
    if(s1[i] != s2[i])
      return true;

  return false;
}

template <unsigned Dim>
CudaSize<Dim> operator/(const CudaSize<Dim> &lhs, const float &rhs) {
  if (rhs == 0)
    fprintf(stderr, "Division by zero!!\n");
  CudaSize<Dim> out = lhs;
  for(size_t i = 0; i < Dim; ++i)
    out[i] /= rhs;

  return out;
}

template <unsigned Dim>
CudaSize<Dim> operator-(const CudaSize<Dim> &lhs, const CudaSize<Dim> &rhs) {
  CudaSize<Dim> out = lhs;
  for(size_t i = Dim; i--;)
    out[i]-= rhs[i];
  return out;
}

template <class Type, unsigned Dim> class CudaHostMemoryHeap
{
    Type* buffer;
    CudaSize<Dim> size;
public:
    CudaHostMemoryHeap( )
        : buffer( nullptr )
    { }

    explicit CudaHostMemoryHeap(const CudaSize<Dim> &_size)
        : buffer( nullptr )
    {
        allocate( _size );
        memset(buffer, 0, getBytes() );
    }

    CudaHostMemoryHeap<Type,Dim>& operator=(const CudaHostMemoryHeap<Type,Dim>& rhs)
    {
        if( buffer != nullptr )
        {
            allocate( rhs.getSize() );
        }
        else if( size != rhs.getSize() )
        {
            deallocate();
            allocate( rhs.getSize() );
        }

        memcpy(buffer, rhs.buffer, rhs.getBytes() );
        return *this;
    }

    ~CudaHostMemoryHeap()
    {
        deallocate();
    }

    const CudaSize<Dim>& getSize() const
    {
        return size;
    }

    size_t getBytes() const
    {
        return size.getSize() * sizeof(Type);
        // const int sx = (Dim >= 1) ? size[0] : 1;
        // const int sy = (Dim >= 2) ? size[1] : 1;
        // const int sz = (Dim >= 3) ? size[2] : 1;
        // return sx * sy * sz * sizeof (Type);
    }

    Type *getBuffer()
    {
        return buffer;
    }
    const Type *getBuffer() const
    {
        return buffer;
    }
    Type& operator()(size_t x)
    {
        return buffer[x];
    }
    Type& operator()(size_t x, size_t y)
    {
        const int sx = (Dim >= 1) ? size[0] : 1;
        return buffer[y * sx + x];
    }

    void allocate( const CudaSize<Dim> &_size )
    {
        size = _size;
        const int sx = (Dim >= 1) ? size[0] : 1;
        const int sy = (Dim >= 2) ? size[1] : 1;
        const int sz = (Dim >= 3) ? size[2] : 1;
        cudaError_t err = cudaMallocHost( &buffer, sx * sy * sz * sizeof (Type) );
        if( err != cudaSuccess )
        {
            ALICEVISION_LOG_ERROR( "Could not allocate pinned host memory in " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(err) );
            throw std::runtime_error( "Could not allocate CUDA host memory" );
        }
    }

    void deallocate( )
    {
        if( buffer == nullptr ) return;
        cudaFreeHost(buffer);
        buffer = nullptr;
    }
};

template <class Type, unsigned Dim> class CudaDeviceMemoryPitched
{
    Type* buffer;
    size_t pitch;
    CudaSize<Dim> size;
public:
    CudaDeviceMemoryPitched( )
        : buffer( nullptr )
    { }

    explicit CudaDeviceMemoryPitched(const CudaSize<Dim> &_size)
    {
        allocate( _size );
    }

    explicit CudaDeviceMemoryPitched(const CudaHostMemoryHeap<Type, Dim> &rhs)
    {
        allocate( rhs.getSize() );
        copyFrom( rhs );
    }

    ~CudaDeviceMemoryPitched()
    {
        deallocate();
    }

    CudaDeviceMemoryPitched<Type,Dim>& operator=(const CudaDeviceMemoryPitched<Type,Dim> & rhs)
    {
        if( buffer == nullptr )
        {
            allocate( rhs.size );
        }
        else if( size != rhs.getSize() )
        {
            deallocate( );
            allocate( rhs.getSize() );
        }
        copyFrom( rhs );
        return *this;
    }

    // void bindToTexture( texture<Type, 2, cudaReadModeNormalizedFloat>& texref )
    // void bindToTexture( texture<Type, 2, cudaReadModeElementType>& texref )
    template<typename texturetype>
    void bindToTexture( texturetype& texref )
    {
        cudaError_t err = cudaBindTexture2D( 0, // offset
                                             texref,
                                             this->getBuffer(),
                                             cudaCreateChannelDesc<Type>(),
                                             getSize()[0],
                                             getSize()[1],
                                             getPitch() );
        if( err != cudaSuccess )
        {
            ALICEVISION_LOG_ERROR( "Failed to bind texture reference to pitched memory, " << cudaGetErrorString( err ) );
        }
    }

    // see below with copy() functions
    void copyFrom( const CudaHostMemoryHeap<Type, Dim>& _src );

    const CudaSize<Dim>& getSize() const
    {
        return size;
    }
    size_t getBytes() const
    {
        size_t s;
        s = pitch;
        for(unsigned i = 1; i < Dim; ++i)
            s *= size[i];
        return s;
    }
    size_t getPitch() const
    {
        return pitch;
    }
    Type *getBuffer()
    {
        return buffer;
    }
    const Type *getBuffer() const
    {
        return buffer;
    }
    Type& operator()(size_t x)
    {
        return buffer[x];
    }
    Type& operator()(size_t x, size_t y)
    {
        Type* row = getRow( y );
        return row[x];
    }

    const CudaSize<Dim> stride() const
    {
        CudaSize<Dim> s;
        s[0] = pitch;
        for(unsigned i = 1; i < Dim; ++i)
            s[i] = s[i - 1] * size[i];
        return s;
    }

    void allocate( const CudaSize<Dim> &_size )
    {
        size = _size;

        if(Dim == 2)
        {
            cudaError_t err = cudaMallocPitch<Type>(&buffer, &pitch, size[0] * sizeof(Type), size[1]);
            if( err != cudaSuccess )
            {
                int devid;
                cudaGetDevice( &devid );
                ALICEVISION_LOG_ERROR( "Device " << devid << " alloc " << getBytes() << " bytes failed in " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(err) );
                throw std::runtime_error( "Could not allocate pitched device memory" );
            }
        }
        else if(Dim == 3)
        {
            cudaExtent extent;
            extent.width = size[0] * sizeof(Type);
            extent.height = size[1];
            extent.depth = size[2];
            cudaPitchedPtr pitchDevPtr;
            cudaError_t err = cudaMalloc3D(&pitchDevPtr, extent);
            if( err != cudaSuccess )
            {
                int devid;
                cudaGetDevice( &devid );
                long bytes = size[0] * size[1] * size[2] * sizeof(Type);
                ALICEVISION_LOG_ERROR( "Device " << devid << " alloc "
                                    << size[0] << "x" << size[1] << "x" << size[2] << "x" << sizeof(Type) << " = "
                                    << bytes << " bytes ("
                << (int)(bytes/1024.0f/1024.0f) << " MB) failed in " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(err) );
                throw std::runtime_error( "Could not allocate 3D device memory" );
            }

            buffer = (Type*)pitchDevPtr.ptr;
            pitch = pitchDevPtr.pitch;
        }
    }

    void deallocate()
    {
        if( buffer == nullptr ) return;

        cudaError_t err = cudaFree(buffer);
        if( err != cudaSuccess )
        {
            ALICEVISION_LOG_ERROR( "Device free failed, " << cudaGetErrorString(err) );
        }

        buffer = nullptr;
    }

private:
    Type* getRow( size_t row )
    {
        unsigned char* ptr = (unsigned char*)buffer;
        ptr += row * pitch;
        return (Type*)ptr;
    }
};

template <class Type> class CudaDeviceMemory
{
    Type* buffer;
    size_t sx;
public:
    explicit CudaDeviceMemory(const size_t size)
    {
        allocate( size );
    }

    explicit inline CudaDeviceMemory(const CudaHostMemoryHeap<Type,1> &rhs)
    {
        allocate( rhs.getSize()[0] );
        copy(*this, rhs);
    }

    // constructor with synchronous copy
    CudaDeviceMemory(const Type* inbuf, const size_t size )
    {
        allocate( size );
        copyFrom( inbuf, size );
    }

    // constructor with asynchronous copy
    CudaDeviceMemory(const Type* inbuf, const size_t size, cudaStream_t stream )
    {
        sx = size;
        allocate( size );
        copyFrom( inbuf, size, stream );
    }

    ~CudaDeviceMemory()
    {
        deallocate( );
    }

    CudaDeviceMemory<Type> & operator=(const CudaDeviceMemory<Type> & rhs)
    {
        if( buffer == nullptr )
        {
            allocate( rhs.getSize() );
        }
        else if( sx != rhs.getSize() )
        {
            deallocate( );
            allocate( rhs.getSize() );
        }
        copy(*this, rhs);
        return *this;
    }

    size_t getSize() const
    {
        return sx;
    }
    size_t getBytes() const
    {
        return sx*sizeof(Type);
    }
    Type *getBuffer()
    {
        return buffer;
    }
    const Type *getBuffer() const
    {
        return buffer;
    }

    void allocate( const size_t size )
    {
        sx = size;

        cudaError_t err = cudaMalloc(&buffer, sx * sizeof(Type) );
        if( err != cudaSuccess )
        {
            ALICEVISION_LOG_ERROR( "Could not allocate pinned host memory in " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(err) );
            throw std::runtime_error( "Could not allocate pinned host memory." );
        }
    }

    void deallocate()
    {
        if( buffer == nullptr ) return;

        cudaError_t err = cudaFree(buffer);
        if( err != cudaSuccess )
        {
            ALICEVISION_LOG_ERROR( "Device free failed, " << cudaGetErrorString(err) );
        }

        buffer = nullptr;
    }

    void copyFrom( const Type* inbuf, const size_t num )
    {
        cudaMemcpyKind kind = cudaMemcpyHostToDevice;
        cudaError_t err = cudaMemcpy( buffer, inbuf, num * sizeof(Type), kind );
        if( err != cudaSuccess )
        {
            ALICEVISION_LOG_ERROR( "Failed to copy from flat host buffer to CudaDeviceMemory in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) );
            throw std::runtime_error( "Failed to copy from flat host buffer to CudaDeviceMemory" );
        }
    }

    void copyFrom( const Type* inbuf, const size_t num, cudaStream_t stream )
    {
        cudaMemcpyKind kind = cudaMemcpyHostToDevice;
        cudaError_t err = cudaMemcpyAsync( buffer, inbuf, num * sizeof(Type), kind, stream );
        if( err != cudaSuccess )
        {
            ALICEVISION_LOG_ERROR( "Failed to copy from flat host buffer to CudaDeviceMemory in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) );
            throw std::runtime_error( "Failed to copy from flat host buffer to CudaDeviceMemory" );
        }
    }
};

template <class Type, unsigned Dim> class CudaArray
{
  cudaArray *array;
  size_t sx, sy, sz;
  CudaSize<Dim> size;
public:
  explicit CudaArray(const CudaSize<Dim> &_size)
  {
    size = _size;
    sx = 1;
    sy = 1;
    sz = 1;
    if (Dim >= 1) sx = _size[0];
    if (Dim >= 2) sy = _size[1];
    if (Dim >= 3) sx = _size[2];
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<Type>();
    if(Dim == 1)
    {
      cudaError_t err = cudaMallocArray(&array, &channelDesc, _size[0], 1, cudaArraySurfaceLoadStore);
      if( err != cudaSuccess )
      {
        ALICEVISION_LOG_ERROR( "Device alloc 1D array failed in " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(err) );
        throw std::runtime_error( "Could not allocate device array memory." );
      }
    }
    else if(Dim == 2)
    {
      cudaError_t err = cudaMallocArray(&array, &channelDesc, _size[0], _size[1], cudaArraySurfaceLoadStore);
      if( err != cudaSuccess )
      {
        ALICEVISION_LOG_ERROR( "Device alloc 2D array failed in " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(err) );
        throw std::runtime_error( "Could not allocate device array memory." );
      }
    }
    else
    {
      cudaExtent extent;
      extent.width = _size[0];
      extent.height = _size[1];
      extent.depth = _size[2];
      for(unsigned i = 3; i < Dim; ++i)
        extent.depth *= _size[i];
      cudaError_t err = cudaMalloc3DArray(&array, &channelDesc, extent);
      if( err != cudaSuccess )
      {
        ALICEVISION_LOG_ERROR( "Device alloc 3D array failed in " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(err) );
        throw std::runtime_error( "Could not allocate device array memory." );
      }
    }
  }
  explicit inline CudaArray(const CudaDeviceMemoryPitched<Type, Dim> &rhs)
  {
    size = rhs.getSize();
    sx = 1;
    sy = 1;
    sz = 1;
    if (Dim >= 1) sx = rhs.getSize()[0];
    if (Dim >= 2) sy = rhs.getSize()[1];
    if (Dim >= 3) sx = rhs.getSize()[2];
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<Type>();
    if(Dim == 1)
    {
      cudaError_t err = cudaMallocArray(&array, &channelDesc, size[0], 1, cudaArraySurfaceLoadStore);
      if( err != cudaSuccess )
      {
        ALICEVISION_LOG_ERROR( "Device alloc 1D array failed in " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(err) );
        throw std::runtime_error( "Could not allocate device array memory." );
      }
    }
    else if(Dim == 2)
    {
      cudaError_t err = cudaMallocArray(&array, &channelDesc, size[0], size[1], cudaArraySurfaceLoadStore);
      if( err != cudaSuccess )
      {
        ALICEVISION_LOG_ERROR( "Device alloc 2D array failed in " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(err) );
        throw std::runtime_error( "Could not allocate device array memory." );
      }
    }
    else
    {
      cudaExtent extent;
      extent.width = size[0];
      extent.height = size[1];
      extent.depth = size[2];
      for(unsigned i = 3; i < Dim; ++i)
        extent.depth *= size[i];
      cudaError_t err = cudaMalloc3DArray(&array, &channelDesc, extent);
      if( err != cudaSuccess )
      {
        ALICEVISION_LOG_ERROR( "Device alloc 3D array failed in " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(err) );
        throw std::runtime_error( "Could not allocate device array memory." );
      }
    }
    copy(*this, rhs);
  }
  explicit inline CudaArray(const CudaHostMemoryHeap<Type, Dim> &rhs)
  {
    size = rhs.getSize();
    sx = 1;
    sy = 1;
    sz = 1;
    if (Dim >= 1) sx = rhs.getSize()[0];
    if (Dim >= 2) sy = rhs.getSize()[1];
    if (Dim >= 3) sx = rhs.getSize()[2];
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<Type>();
    if(Dim == 1)
    {
      cudaError_t err = cudaMallocArray(&array, &channelDesc, size[0], 1, cudaArraySurfaceLoadStore);
      if( err != cudaSuccess )
      {
        ALICEVISION_LOG_ERROR( "Device alloc 1D array failed in " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(err) );
        throw std::runtime_error( "Could not allocate device array memory." );
      }
    }
    else if(Dim == 2)
    {
      cudaError_t err = cudaMallocArray(&array, &channelDesc, size[0], size[1], cudaArraySurfaceLoadStore);
      if( err != cudaSuccess )
      {
        ALICEVISION_LOG_ERROR( "Device alloc 2D array failed in " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(err) );
        throw std::runtime_error( "Could not allocate device array memory." );
      }
    }
    else
    {
      cudaExtent extent;
      extent.width = size[0];
      extent.height = size[1];
      extent.depth = size[2];
      for(unsigned i = 3; i < Dim; ++i)
        extent.depth *= size[i];
      cudaError_t err = cudaMalloc3DArray(&array, &channelDesc, extent);
      if( err != cudaSuccess )
      {
        ALICEVISION_LOG_ERROR( "Device alloc 3D array failed in " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(err) );
        throw std::runtime_error( "Could not allocate device array memory." );
      }
    }
    copy(*this, rhs);
  }
  virtual ~CudaArray()
  {
    cudaFreeArray(array);
  }
  size_t getBytes() const
  {
    size_t s;
    s = 1;
    for(unsigned i = 0; i < Dim; ++i)
      s *= size[i];
    return s;
  }
  const CudaSize<Dim>& getSize() const
  {
    return size;
  }
  size_t getPitch() const
  {
    return size[0] * sizeof (Type);
  }
  cudaArray *getArray()
  {
    return array;
  }
  const cudaArray *getArray() const
  {
    return array;
  }
};

template<class Type, unsigned Dim> void copy(CudaHostMemoryHeap<Type, Dim>& _dst, const CudaDeviceMemoryPitched<Type, Dim>& _src)
{
  cudaMemcpyKind kind = cudaMemcpyDeviceToHost;
  if(Dim == 1) {
    cudaError_t err = cudaMemcpy(_dst.getBuffer(), _src.getBuffer(), _src.getBytes(), kind);
    if( err != cudaSuccess )
    {
      ALICEVISION_LOG_ERROR( "Failed to copy : " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
      throw std::runtime_error( "Failed to copy." );
    }
  }
  else if(Dim == 2) {
    cudaError_t err = cudaMemcpy2D(_dst.getBuffer(),
                                   _dst.getSize()[0] * sizeof (Type),
                                   _src.getBuffer(),
                                   _src.getPitch(),
                                   _dst.getSize()[0] * sizeof (Type),
                                   _dst.getSize()[1], kind);
    if( err != cudaSuccess )
    {
      ALICEVISION_LOG_ERROR( "Failed to copy : " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
      throw std::runtime_error( "Failed to copy." );
    }
  }
  else if(Dim >= 3) {
    for (unsigned int slice=0; slice<_dst.getSize()[2]; slice++)
    {
      cudaError_t err = cudaMemcpy2D( _dst.getBuffer() + slice * _dst.getSize()[0] * _dst.getSize()[1],
                                      _dst.getSize()[0] * sizeof (Type),
                                      (unsigned char*)_src.getBuffer() + slice * _src.stride()[1],
                                      _src.getPitch(),
                                      _dst.getSize()[0] * sizeof (Type),
                                      _dst.getSize()[1],
                                      kind);
      if( err != cudaSuccess )
      {
        ALICEVISION_LOG_ERROR( "Failed to copy : " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
        throw std::runtime_error( "Failed to copy." );
      }
    }
  }
}

template<class Type> void copy(CudaHostMemoryHeap<Type,1>& _dst, const CudaDeviceMemory<Type>& _src)
{
  cudaMemcpyKind kind = cudaMemcpyDeviceToHost;
  cudaError_t err = cudaMemcpy(_dst.getBuffer(), _src.getBuffer(), _src.getBytes(), kind);
  if( err != cudaSuccess )
  {
    ALICEVISION_LOG_ERROR( "Failed to ropy from CudaHostMemoryHeap to CudaDeviceMemory in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) );
    throw std::runtime_error( "Failed to copy." );
  }
}

template<class Type, unsigned Dim> void copy(CudaHostMemoryHeap<Type, Dim>& _dst, const CudaArray<Type, Dim>& _src)
{
  cudaMemcpyKind kind = cudaMemcpyDeviceToHost;
  if(Dim == 1) {
    cudaError_t err = cudaMemcpyFromArray(_dst.getBuffer(), _src.getArray(), 0, 0, _dst.getSize()[0] * sizeof (Type), kind);
    if( err != cudaSuccess )
    {
      ALICEVISION_LOG_ERROR( "Failed to copy : " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
      throw std::runtime_error( "Failed to copy." );
    }
  }
  else if(Dim == 2) {
    cudaError_t err = cudaMemcpy2DFromArray(_dst.getBuffer(),
                                            _dst.getSize()[0] * sizeof (Type),
                                            _src.getArray(),
                                            0,
                                            0,
                                            _dst.getSize()[0] * sizeof (Type),
                                            _dst.getSize()[1],
                                            kind);
    if( err != cudaSuccess )
    {
      ALICEVISION_LOG_ERROR( "Failed to copy : " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
      throw std::runtime_error( "Failed to copy." );
    }
  }
  else if(Dim == 3) {
    cudaMemcpy3DParms p = { 0 };
    p.srcArray = const_cast<cudaArray *>(_src.getArray());
    p.srcPtr.pitch = _src.getPitch();
    p.srcPtr.xsize = _src.getSize()[0];
    p.srcPtr.ysize = _src.getSize()[1];
    p.dstPtr.ptr = (void *)_dst.getBuffer();
    p.dstPtr.pitch = _dst.getSize()[0] * sizeof (Type);
    p.dstPtr.xsize = _dst.getSize()[0];
    p.dstPtr.ysize = _dst.getSize()[1];
    p.extent.width = _dst.getSize()[0];
    p.extent.height = _dst.getSize()[1];
    p.extent.depth = _dst.getSize()[2];
    for(unsigned i = 3; i < Dim; ++i)
      p.extent.depth *= _src.getSize()[i];
    p.kind = kind;
    cudaError_t err = cudaMemcpy3D(&p);
    if( err != cudaSuccess )
    {
      ALICEVISION_LOG_ERROR( "Failed to copy : " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
      throw std::runtime_error( "Failed to copy." );
    }
  }
}

template<class Type, unsigned Dim>
void CudaDeviceMemoryPitched<Type, Dim>::copyFrom( const CudaHostMemoryHeap<Type, Dim>& _src )
{
    const cudaMemcpyKind kind = cudaMemcpyHostToDevice;
    if(Dim == 1)
    {
        cudaError_t err = cudaMemcpy(this->getBuffer(), _src.getBuffer(), _src.getBytes(), kind);
        if( err != cudaSuccess )
        {
            ALICEVISION_LOG_ERROR( "Failed to copy : " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
            throw std::runtime_error( "Failed to copy." );
        }
    }
    else if(Dim == 2)
    {
        cudaError_t err = cudaMemcpy2D(this->getBuffer(),
                                       this->getPitch(),
                                       _src.getBuffer(),
                                       _src.getSize()[0] * sizeof (Type),
                                       _src.getSize()[0] * sizeof (Type),
                                       _src.getSize()[1],
                                       kind);
        if( err != cudaSuccess )
        {
            ALICEVISION_LOG_ERROR( "Failed to copy : " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
            throw std::runtime_error( "Failed to copy." );
        }
    }
    else if(Dim >= 3)
    {
        for (unsigned int slice=0; slice<_src.getSize()[2]; slice++)
        {
            cudaError_t err = cudaMemcpy2D( &this->getBuffer()[slice * this->stride()[1]],
                                            this->getPitch(),
                                            _src.getBuffer() + slice * _src.getSize()[0] * _src.getSize()[1],
                                            _src.getSize()[0] * sizeof (Type),
                                            _src.getSize()[0] * sizeof (Type),
                                            _src.getSize()[1],
                                            kind);
            if( err != cudaSuccess )
            {
                ALICEVISION_LOG_ERROR( "Failed to copy : " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
                throw std::runtime_error( "Failed to copy." );
            }
        }
    }
}

template<class Type, unsigned Dim> void copy(CudaDeviceMemoryPitched<Type, Dim>& _dst, const CudaHostMemoryHeap<Type, Dim>& _src)
{
    _dst.copyFrom( _src );
}

template<class Type, unsigned Dim> void copy(CudaDeviceMemoryPitched<Type, Dim>& _dst, const CudaDeviceMemoryPitched<Type, Dim>& _src)
{
  cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
  if(Dim == 1) {
    cudaError_t err = cudaMemcpy(_dst.getBuffer(), _src.getBuffer(), _src.getBytes(), kind);
    if( err != cudaSuccess )
    {
      ALICEVISION_LOG_ERROR( "Failed to copy : " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
      throw std::runtime_error( "Failed to copy." );
    }
  }
  else if(Dim == 2) {
    cudaError_t err = cudaMemcpy2D(_dst.getBuffer(),
                                   _dst.getPitch(),
                                   _src.getBuffer(),
                                   _src.getPitch(),
                                   _src.getSize()[0] * sizeof(Type),
                                   _src.getSize()[1],
                                   kind);
    if( err != cudaSuccess )
    {
      ALICEVISION_LOG_ERROR( "Failed to copy : " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
      throw std::runtime_error( "Failed to copy." );
    }
  }
  else if(Dim >= 3) {
    for (unsigned int slice=0; slice<_src.getSize()[2]; slice++)
    {
      cudaError_t err = cudaMemcpy2D( &_dst.getBuffer()[slice * _dst.stride()[1]],
                                      _dst.getPitch(),
                                      (unsigned char*)_src.getBuffer() + slice * _src.stride()[1],
                                      _src.getPitch(),
                                      _src.getSize()[0] * sizeof(Type),
                                      _src.getSize()[1],
                                      kind);
      if( err != cudaSuccess )
      {
        ALICEVISION_LOG_ERROR( "Failed to copy : " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
        throw std::runtime_error( "Failed to copy." );
      }
    }
  }
}

template<class Type> void copy(CudaDeviceMemory<Type>& _dst, const CudaHostMemoryHeap<Type,1>& _src)
{
  cudaMemcpyKind kind = cudaMemcpyHostToDevice;
  cudaError_t err = cudaMemcpy(_dst.getBuffer(), _src.getBuffer(), _src.getBytes(), kind);
  if( err != cudaSuccess )
  {
    ALICEVISION_LOG_ERROR( "Failed to copy from CudaHostMemoryHeap to CudaDeviceMemory: " << cudaGetErrorString(err) );
    throw std::runtime_error( "Failed to copy." );
  }
}

template<class Type> void copy(CudaDeviceMemory<Type>& _dst, const Type* buffer, const size_t numelems )
{
    _dst.copyFrom( buffer, numelems );
}

template<class Type, unsigned Dim> void copy(CudaDeviceMemoryPitched<Type, Dim>& _dst, const CudaArray<Type, Dim>& _src)
{
  cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
  if(Dim == 1) {
    cudaError_t err = cudaMemcpyFromArray(_dst.getBuffer(), _src.getArray(), 0, 0, _src.getSize()[0] * sizeof(Type), kind);
    if( err != cudaSuccess )
    {
      ALICEVISION_LOG_ERROR( "Failed to copy : " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
      throw std::runtime_error( "Failed to copy." );
    }
  }
  else if(Dim == 2) {
    cudaError_t err = cudaMemcpy2DFromArray(_dst.getBuffer(),
                                            _dst.getPitch(),
                                            _src.getArray(),
                                            0,
                                            0,
                                            _src.getSize()[0] * sizeof(Type),
                                            _src.getSize()[1],
                                            kind);
    if( err != cudaSuccess )
    {
      ALICEVISION_LOG_ERROR( "Failed to copy : " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
      throw std::runtime_error( "Failed to copy." );
    }
  }
  else if(Dim == 3) {
    cudaMemcpy3DParms p = { 0 };
    p.srcArray = const_cast<cudaArray *>(_src.getArray());
    p.srcPtr.pitch = _src.getPitch();
    p.srcPtr.xsize = _src.getSize()[0];
    p.srcPtr.ysize = _src.getSize()[1];
    p.dstPtr.ptr = (void *)_dst.getBuffer();
    p.dstPtr.pitch = _dst.getPitch();
    p.dstPtr.xsize = _dst.getSize()[0];
    p.dstPtr.ysize = _dst.getSize()[1];
    p.extent.width = _src.getSize()[0];
    p.extent.height = _src.getSize()[1];
    p.extent.depth = _src.getSize()[2];
    for(unsigned i = 3; i < Dim; ++i)
      p.extent.depth *= _src.getSize()[i];
    p.kind = kind;
    cudaError_t err = cudaMemcpy3D(&p);
    if( err != cudaSuccess )
    {
      ALICEVISION_LOG_ERROR( "Failed to copy : " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
      throw std::runtime_error( "Failed to copy." );
    }
  }
}

template<class Type, unsigned Dim> void copy(CudaArray<Type, Dim>& _dst, const CudaHostMemoryHeap<Type, Dim>& _src)
{
  cudaMemcpyKind kind = cudaMemcpyHostToDevice;
  if(Dim == 1) {
    cudaError_t err = cudaMemcpyToArray(_dst.getArray(), 0, 0, _src.getBuffer(), _src.getSize()[0] * sizeof (Type), kind);
    if( err != cudaSuccess )
    {
      ALICEVISION_LOG_ERROR( "Failed to copy : " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
      throw std::runtime_error( "Failed to copy." );
    }
  }
  else if(Dim == 2) {
    cudaError_t err = cudaMemcpy2DToArray(_dst.getArray(),
                                          0,
                                          0,
                                          _src.getBuffer(),
                                          _src.getSize()[0] * sizeof (Type),
                                          _src.getSize()[0] * sizeof (Type),
                                          _src.getSize()[1],
                                          kind);
    if( err != cudaSuccess )
    {
      ALICEVISION_LOG_ERROR( "Failed to copy : " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
      throw std::runtime_error( "Failed to copy." );
    }
  }
  else if(Dim == 3) {
    cudaMemcpy3DParms p = { 0 };
    p.srcPtr.ptr = (void *)_src.getBuffer();
    p.srcPtr.pitch = _src.getSize()[0] * sizeof (Type);
    p.srcPtr.xsize = _src.getSize()[0];
    p.srcPtr.ysize = _src.getSize()[1];
    p.dstArray = _dst.getArray();
    p.dstPtr.pitch = _dst.getPitch();
    p.dstPtr.xsize = _dst.getSize()[0];
    p.dstPtr.ysize = _dst.getSize()[1];
    p.extent.width = _src.getSize()[0];
    p.extent.height = _src.getSize()[1];
    p.extent.depth = _src.getSize()[2];
    for(unsigned i = 3; i < Dim; ++i)
      p.extent.depth *= _src.getSize()[i];
    p.kind = kind;
    cudaError_t err = cudaMemcpy3D(&p);
    if( err != cudaSuccess )
    {
      ALICEVISION_LOG_ERROR( "Failed to copy : " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
      throw std::runtime_error( "Failed to copy." );
    }
  }
}

template<class Type, unsigned Dim> void copy(CudaArray<Type, Dim>& _dst, const CudaDeviceMemoryPitched<Type, Dim>& _src)
{
  cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
  if(Dim == 1) {
    cudaError_t err = cudaMemcpyToArray(_dst.getArray(), 0, 0, _src.getBuffer(), _src.getSize()[0] * sizeof(Type), kind);
    if( err != cudaSuccess )
    {
      ALICEVISION_LOG_ERROR( "Failed to copy : " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
      throw std::runtime_error( "Failed to copy." );
    }
  }
  else if(Dim == 2) {
    cudaError_t err = cudaMemcpy2DToArray(_dst.getArray(),
                                          0,
                                          0,
                                          _src.getBuffer(),
                                          _src.getPitch(),
                                          _src.getSize()[0] * sizeof(Type),
                                          _src.getSize()[1],
                                          kind);
    if( err != cudaSuccess )
    {
      ALICEVISION_LOG_ERROR( "Failed to copy : " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
      throw std::runtime_error( "Failed to copy." );
    }
  }
  else if(Dim == 3) {
    cudaMemcpy3DParms p = { 0 };
    p.srcPtr.ptr = (void *)_src.getBuffer();
    p.srcPtr.pitch = _src.getPitch();
    p.srcPtr.xsize = _src.getSize()[0];
    p.srcPtr.ysize = _src.getSize()[1];
    p.dstArray = _dst.getArray();
    p.dstPtr.pitch = _dst.getPitch();
    p.dstPtr.xsize = _dst.getSize()[0];
    p.dstPtr.ysize = _dst.getSize()[1];
    p.extent.width = _src.getSize()[0];
    p.extent.height = _src.getSize()[1];
    p.extent.depth = _src.getSize()[2];
    for(unsigned i = 3; i < Dim; ++i)
      p.extent.depth *= _src.getSize()[i];
    p.kind = kind;
    cudaError_t err = cudaMemcpy3D(&p);
    if( err != cudaSuccess )
    {
      ALICEVISION_LOG_ERROR( "Failed to copy : " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
      throw std::runtime_error( "Failed to copy." );
    }
  }
}

template<class Type, unsigned Dim> void copy(Type* _dst, size_t sx, size_t sy, const CudaDeviceMemoryPitched<Type, Dim>& _src)
{
  if(Dim == 2) {
    cudaError_t err = cudaMemcpy2D(_dst, sx * sizeof (Type), _src.getBuffer(), _src.getPitch(), sx * sizeof (Type), sy, cudaMemcpyDeviceToHost);
    if( err != cudaSuccess )
    {
      ALICEVISION_LOG_ERROR( "Failed to copy : " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
      throw std::runtime_error( "Failed to copy device to host." );
    }
  }
}

template<class Type, unsigned Dim> void copy(CudaDeviceMemoryPitched<Type, Dim>& _dst, const Type* _src, size_t sx, size_t sy)
{
  if(Dim == 2) {
    cudaError_t err = cudaMemcpy2D(_dst.getBuffer(), _dst.getPitch(), _src, sx * sizeof (Type), sx * sizeof(Type), sy, cudaMemcpyHostToDevice);
    if( err != cudaSuccess )
    {
      ALICEVISION_LOG_ERROR( "Failed to copy : " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
      throw std::runtime_error( "Failed to copy host to device." );
    }
  }
}

template<class Type, unsigned Dim> void copy(Type* _dst, size_t sx, size_t sy, size_t sz, const CudaDeviceMemoryPitched<Type, Dim>& _src)
{
  if(Dim >= 3) {
    // for (unsigned int slice=0; slice<sz; slice++)
    // {
      cudaError_t err = cudaMemcpy2D( _dst,
                                      sx * sizeof (Type),
                                      (unsigned char*)_src.getBuffer(),
                                      _src.stride()[0], // _src.getPitch(),
                                      sx * sizeof (Type),
                                      sy * sz,
                                      cudaMemcpyDeviceToHost);
      if( err != cudaSuccess )
      {
        ALICEVISION_LOG_ERROR( "Failed to copy : " << std::endl
            << "    " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
        throw std::runtime_error("Failed to copy.");
      }
    // }
  }
}

template<class Type, unsigned Dim> void copy(Type* _dst, size_t sx, size_t sy, size_t sz, const CudaDeviceMemoryPitched<Type, Dim>& _src, cudaStream_t stream)
{
  if(Dim >= 3) {
    // for (unsigned int slice=0; slice<sz; slice++)
    // {
      cudaError_t err = cudaMemcpy2DAsync( _dst,
                                           sx * sizeof (Type),
                                           (unsigned char*)_src.getBuffer(),
                                           _src.stride()[0], // _src.getPitch(),
                                           sx * sizeof (Type),
                                           sy * sz,
                                           cudaMemcpyDeviceToHost,
                                           stream );
      if( err != cudaSuccess )
      {
        ALICEVISION_LOG_ERROR( "Failed to copy : " << std::endl
            << "    " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
        throw std::runtime_error("Failed to copy.");
      }
    // }
  }
}

template<class Type, unsigned Dim> void copy(CudaDeviceMemoryPitched<Type, Dim>& _dst, const Type* _src, size_t sx, size_t sy, size_t sz)
{
  if(Dim >= 3) {
    // for (unsigned int slice=0; slice<sz; slice++)
    // {
      cudaError_t err = cudaMemcpy2D( (unsigned char*)_dst.getBuffer(),
                                      _dst.getPitch(),
                                      _src,
                                      sx * sizeof(Type),
                                      sx * sizeof(Type),
                                      sy * sz,
                                      cudaMemcpyHostToDevice);
      if( err != cudaSuccess )
      {
        ALICEVISION_LOG_ERROR( "Failed to copy : " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
        throw std::runtime_error( "Failed to copy host to device." );
      }
    // }
  }
}

template<class Type> void copy2D( Type* dst, size_t sx, size_t sy,
                                  Type* src, size_t src_pitch,
                                  cudaStream_t stream )
{
    cudaError_t err = cudaMemcpy2DAsync( dst,
                                         sx * sizeof(Type),
                                         src,
                                         src_pitch,
                                         sx * sizeof(Type),
                                         sy,
                                         cudaMemcpyDeviceToHost,
                                         stream );
    if( err != cudaSuccess )
    {
        ALICEVISION_LOG_ERROR( "Failed to copy : " << std::endl
            << "    " << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) );
        throw std::runtime_error("Failed to copy.");
    }
}
struct cameraStructBase
{
    float  P[12];
    float  iP[9];
    float  R[9];
    float  iR[9];
    float  K[9];
    float  iK[9];
    float3 C;
    float3 XVect;
    float3 YVect;
    float3 ZVect;
};

typedef cameraStructBase DeviceCameraStructBase;

struct cameraStruct
{
    const cameraStructBase* param_hst;
    const cameraStructBase* param_dev;
    CudaHostMemoryHeap<uchar4, 2>* tex_rgba_hmh;
    int camId;
    cudaStream_t stream; // allow async work on cameras used in parallel
};

struct ps_parameters
{
    int epipShift;
    int rotX;
    int rotY;
};

struct TexturedArray
{
    CudaDeviceMemoryPitched<uchar4, 2>* arr;
    cudaTextureObject_t tex;
};
typedef std::vector<std::vector<TexturedArray> > Pyramid;

#define MAX_PTS 500           // 500
#define MAX_PATCH_PIXELS 2500 // 50*50

} // namespace depthMap
} // namespace aliceVision
