// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "SemiGlobalMatchingRcTc.hpp"
#include <aliceVision/mvsUtils/common.hpp>
#include <aliceVision/system/nvtx.hpp>

namespace aliceVision {
namespace depthMap {

SemiGlobalMatchingRcTc::SemiGlobalMatchingRcTc(
            const std::vector<int>& _index_set,
            const std::vector<std::vector<float> >& _rcTcDepths,
            int _rc,
            const StaticVector<int>& _tc,
            int _scale,
            int _step,
            SemiGlobalMatchingParams* _sp,
            StaticVectorBool* _rcSilhoueteMap)
    : index_set( _index_set )
    , sp( _sp )
    , rc( _rc )
    , tc( _tc )
    , scale( _scale )
    , step( _step )
    , w( sp->mp->getWidth(rc) / (scale * step) )
    , h( sp->mp->getHeight(rc) / (scale * step) )
    , rcTcDepths( _rcTcDepths )
{
    epipShift = 0.0f;

    rcSilhoueteMap = _rcSilhoueteMap;
}

SemiGlobalMatchingRcTc::~SemiGlobalMatchingRcTc()
{
    //
}

void SemiGlobalMatchingRcTc::computeDepthSimMapVolume(
        std::vector<StaticVector<unsigned char> >& volume,
        float& volumeMBinGPUMem,
        int wsh,
        float gammaC,
        float gammaP)
{
    const long tall = clock();

    const int volStepXY = step;
    const int volDimX = w;
    const int volDimY = h;
    const int zDimsAtATime = 32;
    int maxDimZ = *index_set.begin();

    for( auto j : index_set )
    {
        const int volDimZ = rcTcDepths[j].size();

        volume[j].resize( volDimX * volDimY * volDimZ );

        maxDimZ = std::max( maxDimZ, volDimZ );
    }

    float* volume_buf = new float[ index_set.size() * volDimX * volDimY * maxDimZ ];

    std::map<int,float*> volume_tmp;

    int ct = 0;
    for( auto j : index_set )
    {
        volume_tmp.emplace( j, &volume_buf[ct * volDimX * volDimY * maxDimZ] );
        ct++;
    }

    /* request this device to allocate
     *   (max_img - 1) * X * Y * dims_at_a_time * sizeof(float)
     * of device memory. max_img include the rc images, therefore -1.
     */
    std::vector<CudaDeviceMemoryPitched<float, 3>*> volume_tmp_on_gpu;
    sp->cps->allocTempVolume( volume_tmp_on_gpu,
                              sp->cps->maxImagesInGPU() - 1,
                              volDimX,
                              volDimY,
                              zDimsAtATime );

    const int volume_offset = volDimX * volDimY * maxDimZ;
    volumeMBinGPUMem =
            sp->cps->sweepPixelsToVolume( index_set,
                                          volume_buf,
                                          volume_offset,
                                          volume_tmp_on_gpu,
                                          volDimX, volDimY,
                                          volStepXY,
                                          zDimsAtATime,
                                          rcTcDepths,
                                          rc, tc,
                                          rcSilhoueteMap,
                                          wsh, gammaC, gammaP, scale, 1,
                                          0.0f);

    sp->cps->freeTempVolume( volume_tmp_on_gpu );

    /*
     * TODO: This conversion operation on the host consumes a lot of time,
     *       about 1/3 of the actual computation. Work to avoid it.
     */
    nvtxPushA( "host-copy of volume", __FILE__, __LINE__ );
    for( auto j : index_set )
    {
        const int volDimZ = rcTcDepths[j].size();

        for( int i=0; i<volDimX * volDimY * volDimZ; i++ )
        {
            float* ptr = volume_tmp[j];
            volume[j][i] = (unsigned char)( 255.0f * std::max(std::min(ptr[i],1.0f),0.0f) );
        }
    }

    delete [] volume_buf;

    // delete volume_tmp;

    if(sp->mp->verbose)
        mvsUtils::printfElapsedTime(tall, "SemiGlobalMatchingRcTc::computeDepthSimMapVolume ");

    for( auto j : index_set )
    {
        const int volDimZ = rcTcDepths[j].size();

        if(sp->P3 > 0)
        {
#pragma omp parallel for
            for(int y = 0; y < volDimY; y++)
            {
                for(int x = 0; x < volDimX; x++)
                {
                    volume[j][(volDimZ - 1) * volDimY * volDimX + y * volDimX + x] = sp->P3;
                    volume[j][(volDimZ - 2) * volDimY * volDimX + y * volDimX + x] = sp->P3;
                    volume[j][(volDimZ - 3) * volDimY * volDimX + y * volDimX + x] = sp->P3;
                    volume[j][(volDimZ - 4) * volDimY * volDimX + y * volDimX + x] = sp->P3;
                }
            }
        }
    }
    nvtxPop( "host-copy of volume" );
}

} // namespace depthMap
} // namespace aliceVision
