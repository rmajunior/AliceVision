// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "SemiGlobalMatchingRcTc.hpp"
#include <aliceVision/mvsUtils/common.hpp>

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

StaticVector<Voxel>* SemiGlobalMatchingRcTc::getPixels()
{
    StaticVector<Voxel>* pixels = new StaticVector<Voxel>();

    pixels->reserve(w * h);

    for(int y = 0; y < h; y++)
    {
        for(int x = 0; x < w; x++)
        {
            if(rcSilhoueteMap == nullptr)
            {
                pixels->push_back(Voxel(x * step, y * step, 0));
            }
            else
            {
                bool isBackgroundPixel = (*rcSilhoueteMap)[y * w + x];
                if(!isBackgroundPixel)
                {
                    pixels->push_back(Voxel(x * step, y * step, 0));
                }
            }
        }
    }
    return pixels;
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

    const int volume_offset = volDimX * volDimY * maxDimZ;
    volumeMBinGPUMem =
            sp->cps->sweepPixelsToVolume( index_set,
                                          volume_buf,
                                          volume_offset,
                                          volDimX, volDimY,
                                          volStepXY,
                                          zDimsAtATime,
                                          rcTcDepths,
                                          rc, tc,
                                          wsh, gammaC, gammaP, scale, 1,
                                          0.0f);

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
}

} // namespace depthMap
} // namespace aliceVision
