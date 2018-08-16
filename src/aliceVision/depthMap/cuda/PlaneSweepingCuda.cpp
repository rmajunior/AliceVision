// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "PlaneSweepingCuda.hpp"
#include <aliceVision/system/Logger.hpp>
#include <aliceVision/mvsData/Matrix3x3.hpp>
#include <aliceVision/mvsData/Matrix3x4.hpp>
#include <aliceVision/mvsData/OrientedPoint.hpp>
#include <aliceVision/mvsData/SeedPoint.hpp>
#include <aliceVision/mvsUtils/common.hpp>
#include <aliceVision/mvsUtils/fileIO.hpp>
#include <aliceVision/depthMap/cuda/commonStructures.hpp>
#include <aliceVision/depthMap/cuda/planeSweeping/plane_sweeping_cuda.hpp>

#include <iostream>
#include <stdexcept>

namespace aliceVision {
namespace depthMap {

static void cps_fillCamera(cameraStruct& cam, int c, mvsUtils::MultiViewParams* mp, int scale, const char* called_from )
{
    std::cerr << "Calling " << __FUNCTION__
              << "     to fill info for camera " << c << " at scale " << scale
              << " from " << called_from << std::endl;

    cam.scale = scale;

    Matrix3x3 scaleM;
    scaleM.m11 = 1.0 / (float)scale;
    scaleM.m12 = 0.0;
    scaleM.m13 = 0.0;
    scaleM.m21 = 0.0;
    scaleM.m22 = 1.0 / (float)scale;
    scaleM.m23 = 0.0;
    scaleM.m31 = 0.0;
    scaleM.m32 = 0.0;
    scaleM.m33 = 1.0;
    Matrix3x3 K = scaleM * mp->KArr[c];

    Matrix3x3 iK = K.inverse();
    Matrix3x4 P = K * (mp->RArr[c] | (Point3d(0.0, 0.0, 0.0) - mp->RArr[c] * mp->CArr[c]));
    Matrix3x3 iP = mp->iRArr[c] * iK;

    cam.base.C.x = mp->CArr[c].x;
    cam.base.C.y = mp->CArr[c].y;
    cam.base.C.z = mp->CArr[c].z;

    cam.base.P[0] = P.m11;
    cam.base.P[1] = P.m21;
    cam.base.P[2] = P.m31;
    cam.base.P[3] = P.m12;
    cam.base.P[4] = P.m22;
    cam.base.P[5] = P.m32;
    cam.base.P[6] = P.m13;
    cam.base.P[7] = P.m23;
    cam.base.P[8] = P.m33;
    cam.base.P[9] = P.m14;
    cam.base.P[10] = P.m24;
    cam.base.P[11] = P.m34;

    cam.base.iP[0] = iP.m11;
    cam.base.iP[1] = iP.m21;
    cam.base.iP[2] = iP.m31;
    cam.base.iP[3] = iP.m12;
    cam.base.iP[4] = iP.m22;
    cam.base.iP[5] = iP.m32;
    cam.base.iP[6] = iP.m13;
    cam.base.iP[7] = iP.m23;
    cam.base.iP[8] = iP.m33;

    cam.base.R[0] = mp->RArr[c].m11;
    cam.base.R[1] = mp->RArr[c].m21;
    cam.base.R[2] = mp->RArr[c].m31;
    cam.base.R[3] = mp->RArr[c].m12;
    cam.base.R[4] = mp->RArr[c].m22;
    cam.base.R[5] = mp->RArr[c].m32;
    cam.base.R[6] = mp->RArr[c].m13;
    cam.base.R[7] = mp->RArr[c].m23;
    cam.base.R[8] = mp->RArr[c].m33;

    cam.base.iR[0] = mp->iRArr[c].m11;
    cam.base.iR[1] = mp->iRArr[c].m21;
    cam.base.iR[2] = mp->iRArr[c].m31;
    cam.base.iR[3] = mp->iRArr[c].m12;
    cam.base.iR[4] = mp->iRArr[c].m22;
    cam.base.iR[5] = mp->iRArr[c].m32;
    cam.base.iR[6] = mp->iRArr[c].m13;
    cam.base.iR[7] = mp->iRArr[c].m23;
    cam.base.iR[8] = mp->iRArr[c].m33;

    cam.base.K[0] = K.m11;
    cam.base.K[1] = K.m21;
    cam.base.K[2] = K.m31;
    cam.base.K[3] = K.m12;
    cam.base.K[4] = K.m22;
    cam.base.K[5] = K.m32;
    cam.base.K[6] = K.m13;
    cam.base.K[7] = K.m23;
    cam.base.K[8] = K.m33;

    cam.base.iK[0] = iK.m11;
    cam.base.iK[1] = iK.m21;
    cam.base.iK[2] = iK.m31;
    cam.base.iK[3] = iK.m12;
    cam.base.iK[4] = iK.m22;
    cam.base.iK[5] = iK.m32;
    cam.base.iK[6] = iK.m13;
    cam.base.iK[7] = iK.m23;
    cam.base.iK[8] = iK.m33;
}

static void cps_fillCameraData(mvsUtils::ImagesCache* ic, cameraStruct& cam, int c, mvsUtils::MultiViewParams* mp)
{
    std::cerr << "Calling " << __FUNCTION__
              << "    to fill pixels for camera " << c << std::endl;

    // memcpyGrayImageFromFileToArr(cam->tex_hmh->getBuffer(), mp->indexes[c], mp, true, 1, 0);
    // memcpyRGBImageFromFileToArr(
    //	cam->tex_hmh_r->getBuffer(),
    //	cam->tex_hmh_g->getBuffer(),
    //	cam->tex_hmh_b->getBuffer(), mp->indexes[c], mp, true, 1, 0);

    ic->refreshData(c);

    Pixel pix;
    for(pix.y = 0; pix.y < mp->getHeight(c); pix.y++)
    {
        for(pix.x = 0; pix.x < mp->getWidth(c); pix.x++)
        {
             uchar4& pix_rgba = ic->transposed ? (*cam.tex_rgba_hmh)(pix.x, pix.y) : (*cam.tex_rgba_hmh)(pix.y, pix.x);
             const rgb& pc = ic->getPixelValue(pix, c);
             pix_rgba.x = pc.r;
             pix_rgba.y = pc.g;
             pix_rgba.z = pc.b;
             pix_rgba.w = 0;
        }
    }
}

PlaneSweepingCuda::PlaneSweepingCuda(int _CUDADeviceNo, mvsUtils::ImagesCache* _ic, mvsUtils::MultiViewParams* _mp,
                                         mvsUtils::PreMatchCams* _pc, int _scales)
{
    CUDADeviceNo = _CUDADeviceNo;

    ic = _ic;
    scales = _scales;
    mp = _mp;
    pc = _pc;

    const int maxImageWidth = mp->getMaxImageWidth();
    const int maxImageHeight = mp->getMaxImageHeight();

    verbose = mp->verbose;

    float oneimagemb = 4.0f * (((float)(maxImageWidth * maxImageHeight) / 1024.0f) / 1024.0f);
    for(int scale = 2; scale <= scales; ++scale)
    {
        oneimagemb += 4.0 * (((float)((maxImageWidth / scale) * (maxImageHeight / scale)) / 1024.0) / 1024.0);
    }
    float maxmbGPU = 100.0f;
    nImgsInGPUAtTime = (int)(maxmbGPU / oneimagemb);
    nImgsInGPUAtTime = std::max(2, std::min(mp->ncams, nImgsInGPUAtTime));

    // TODO remove nbest ... now must be 1
    nbest = 1;
    nbestkernelSizeHalf = 1;

    doVizualizePartialDepthMaps = mp->_ini.get<bool>("grow.visualizePartialDepthMaps", false);
    useRcDepthsOrRcTcDepths = mp->_ini.get<bool>("grow.useRcDepthsOrRcTcDepths", false);

    minSegSize = mp->_ini.get<int>("fuse.minSegSize", 100);
    varianceWSH = mp->_ini.get<int>("global.varianceWSH", 4);

    subPixel = mp->_ini.get<bool>("global.subPixel", true);

    ALICEVISION_LOG_INFO("PlaneSweepingCuda:" << std::endl
                         << "\t- nImgsInGPUAtTime: " << nImgsInGPUAtTime << std::endl
                         << "\t- scales: " << scales << std::endl
                         << "\t- subPixel: " << (subPixel ? "Yes" : "No") << std::endl
                         << "\t- varianceWSH: " << varianceWSH);

    // allocate global on the device
    ps_deviceAllocate(ps_texs_arr, nImgsInGPUAtTime, maxImageWidth, maxImageHeight, scales, CUDADeviceNo);

    cams     .resize(nImgsInGPUAtTime);
    camsRcs  .resize(nImgsInGPUAtTime);
    camsTimes.resize(nImgsInGPUAtTime);

    for(int rc = 0; rc < nImgsInGPUAtTime; ++rc)
    {
        cams[rc].tex_rgba_hmh =
            new CudaHostMemoryHeap<uchar4, 2>(CudaSize<2>(maxImageWidth, maxImageHeight));

        cps_fillCamera(cams[rc], rc, mp, 1, __FUNCTION__ );
        cps_fillCameraData(ic, cams[rc], rc, mp);
        camsRcs[rc]   = rc;
        camsTimes[rc] = clock();
        ps_deviceUpdateCam(ps_texs_arr, cams[rc], rc, CUDADeviceNo,
                           nImgsInGPUAtTime, scales, maxImageWidth, maxImageHeight, varianceWSH);
    }
}

void PlaneSweepingCuda::cameraToDevice( int rc, const StaticVector<int>& tcams )
{
    std::cerr << "Called " << __FUNCTION__ << " with cameras" << std::endl
              << "    rc = " << rc << std::endl;
    for( auto it : tcams )
    {
        std::cerr << "    " << it << std::endl;
    }
}

int PlaneSweepingCuda::addCam(int rc, int scale, const char* calling_func)
{
    // fist is oldest
    int id = camsRcs.indexOf(rc);
    if(id == -1)
    {
        // get oldest id
        int oldestId = camsTimes.minValId();

        long t1 = clock();

        cps_fillCamera(cams[oldestId], rc, mp, scale, calling_func );
        cps_fillCameraData(ic, cams[oldestId], rc, mp);
        ps_deviceUpdateCam(ps_texs_arr, cams[oldestId], oldestId,
                           CUDADeviceNo, nImgsInGPUAtTime, scales, mp->getMaxImageWidth(), mp->getMaxImageHeight(), varianceWSH);

        if(verbose)
            mvsUtils::printfElapsedTime(t1, "copy image from disk to GPU ");

        camsRcs[oldestId] = rc;
        camsTimes[oldestId] = clock();
        id = oldestId;
    }
    else
    {

        cps_fillCamera(cams[id], rc, mp, scale, calling_func );
        // cps_fillCameraData((cameraStruct*)(*cams)[id], rc, mp, H, scales);
        // ps_deviceUpdateCam((cameraStruct*)(*cams)[id], id, scales);

        camsTimes[id] = clock();
    }
    return id;
}

PlaneSweepingCuda::~PlaneSweepingCuda(void)
{
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // deallocate global on the device
    ps_deviceDeallocate(ps_texs_arr, CUDADeviceNo, nImgsInGPUAtTime, scales);

    for(int c = 0; c < cams.size(); c++)
    {
        delete cams[c].tex_rgba_hmh;
    }

    mp = NULL;
}

void PlaneSweepingCuda::getMinMaxdepths(int rc, const StaticVector<int>& tcams, float& minDepth, float& midDepth,
                                          float& maxDepth)
{
    StaticVector<SeedPoint>* seeds;
    mvsUtils::loadSeedsFromFile(&seeds, rc, mp, mvsUtils::EFileType::seeds);

    float minCamDist = (float)mp->_ini.get<double>("prematching.minCamDist", 0.0f);
    float maxCamDist = (float)mp->_ini.get<double>("prematching.maxCamDist", 15.0f);
    float maxDepthScale = (float)mp->_ini.get<double>("prematching.maxDepthScale", 1.5f);
    bool minMaxDepthDontUseSeeds = mp->_ini.get<bool>("prematching.minMaxDepthDontUseSeeds", false);

    if((seeds->empty()) || minMaxDepthDontUseSeeds)
    {
        minDepth = 0.0f;
        maxDepth = 0.0f;
        for(int c = 0; c < tcams.size(); c++)
        {
            int tc = tcams[c];
            minDepth += (mp->CArr[rc] - mp->CArr[tc]).size() * minCamDist;
            maxDepth += (mp->CArr[rc] - mp->CArr[tc]).size() * maxCamDist;
        }
        minDepth /= (float)tcams.size();
        maxDepth /= (float)tcams.size();
        midDepth = (minDepth + maxDepth) / 2.0f;
    }
    else
    {
        OrientedPoint rcplane;
        rcplane.p = mp->CArr[rc];
        rcplane.n = mp->iRArr[rc] * Point3d(0.0, 0.0, 1.0);
        rcplane.n = rcplane.n.normalize();

        minDepth = std::numeric_limits<float>::max();
        maxDepth = -std::numeric_limits<float>::max();

        // StaticVector<sortedId> *sos = new StaticVector<sortedId>();
        // sos->reserve(seeds->size());
        // for (int i=0;i<seeds->size();i++) {
        //	sos->push_back(sortedId(i,pointPlaneDistance((*seeds)[i].op.p,rcplane.p,rcplane.n)));
        //};
        // qsort(&(*sos)[0],sos->size(),sizeof(sortedId),qsortCompareSortedIdAsc);
        // minDepth = (*sos)[(int)((float)sos->size()*0.1f)].value;
        // maxDepth = (*sos)[(int)((float)sos->size()*0.9f)].value;

        Point3d cg = Point3d(0.0f, 0.0f, 0.0f);
        for(int i = 0; i < seeds->size(); i++)
        {
            SeedPoint* sp = &(*seeds)[i];
            cg = cg + sp->op.p;
            float depth = pointPlaneDistance(sp->op.p, rcplane.p, rcplane.n);
            minDepth = std::min(minDepth, depth);
            maxDepth = std::max(maxDepth, depth);
        }
        cg = cg / (float)seeds->size();
        midDepth = pointPlaneDistance(cg, rcplane.p, rcplane.n);

        maxDepth = maxDepth * maxDepthScale;
    }

    delete seeds;
}

StaticVector<float>* PlaneSweepingCuda::getDepthsByPixelSize(int rc, float minDepth, float midDepth, float maxDepth,
                                                               int scale, int step, int maxDepthsHalf)
{
    float d = (float)step;

    OrientedPoint rcplane;
    rcplane.p = mp->CArr[rc];
    rcplane.n = mp->iRArr[rc] * Point3d(0.0, 0.0, 1.0);
    rcplane.n = rcplane.n.normalize();

    int ndepthsMidMax = 0;
    float maxdepth = midDepth;
    while((maxdepth < maxDepth) && (ndepthsMidMax < maxDepthsHalf))
    {
        Point3d p = rcplane.p + rcplane.n * maxdepth;
        float pixSize = mp->getCamPixelSize(p, rc, (float)scale * d);
        maxdepth += pixSize;
        ndepthsMidMax++;
    }

    int ndepthsMidMin = 0;
    float mindepth = midDepth;
    while((mindepth > minDepth) && (ndepthsMidMin < maxDepthsHalf * 2 - ndepthsMidMax))
    {
        Point3d p = rcplane.p + rcplane.n * mindepth;
        float pixSize = mp->getCamPixelSize(p, rc, (float)scale * d);
        mindepth -= pixSize;
        ndepthsMidMin++;
    }

    // getNumberOfDepths
    float depth = mindepth;
    int ndepths = 0;
    float pixSize = 1.0f;
    while((depth < maxdepth) && (pixSize > 0.0f) && (ndepths < 2 * maxDepthsHalf))
    {
        Point3d p = rcplane.p + rcplane.n * depth;
        pixSize = mp->getCamPixelSize(p, rc, (float)scale * d);
        depth += pixSize;
        ndepths++;
    }

    StaticVector<float>* out = new StaticVector<float>();
    out->reserve(ndepths);

    // fill
    depth = mindepth;
    pixSize = 1.0f;
    ndepths = 0;
    while((depth < maxdepth) && (pixSize > 0.0f) && (ndepths < 2 * maxDepthsHalf))
    {
        out->push_back(depth);
        Point3d p = rcplane.p + rcplane.n * depth;
        pixSize = mp->getCamPixelSize(p, rc, (float)scale * d);
        depth += pixSize;
        ndepths++;
    }

    // check if it is asc
    for(int i = 0; i < out->size() - 1; i++)
    {
        if((*out)[i] >= (*out)[i + 1])
        {

            for(int j = 0; j <= i + 1; j++)
            {
                ALICEVISION_LOG_TRACE("getDepthsByPixelSize: check if it is asc: " << (*out)[j]);
            }
            throw std::runtime_error("getDepthsByPixelSize not asc.");
        }
    }

    return out;
}

StaticVector<float>* PlaneSweepingCuda::getDepthsRcTc(int rc, int tc, int scale, float midDepth,
                                                        int maxDepthsHalf)
{
    OrientedPoint rcplane;
    rcplane.p = mp->CArr[rc];
    rcplane.n = mp->iRArr[rc] * Point3d(0.0, 0.0, 1.0);
    rcplane.n = rcplane.n.normalize();

    Point2d rmid = Point2d((float)mp->getWidth(rc) / 2.0f, (float)mp->getHeight(rc) / 2.0f);
    Point2d pFromTar, pToTar; // segment of epipolar line of the principal point of the rc camera to the tc camera
    getTarEpipolarDirectedLine(&pFromTar, &pToTar, rmid, rc, tc, mp);

    int allDepths = static_cast<int>((pToTar - pFromTar).size());
    if(verbose == true)
    {
        ALICEVISION_LOG_DEBUG("allDepths: " << allDepths);
    }

    Point2d pixelVect = ((pToTar - pFromTar).normalize()) * std::max(1.0f, (float)scale);
    // printf("%f %f %i %i\n",pixelVect.size(),((float)(scale*step)/3.0f),scale,step);

    Point2d cg = Point2d(0.0f, 0.0f);
    Point3d cg3 = Point3d(0.0f, 0.0f, 0.0f);
    int ncg = 0;
    // navigate through all pixels of the epilolar segment
    // Compute the middle of the valid pixels of the epipolar segment (in rc camera) of the principal point (of the rc camera)
    for(int i = 0; i < allDepths; i++)
    {
        Point2d tpix = pFromTar + pixelVect * (float)i;
        Point3d p;
        if(triangulateMatch(p, rmid, tpix, rc, tc, mp)) // triangulate principal point from rc with tpix
        {
            float depth = orientedPointPlaneDistance(p, rcplane.p, rcplane.n); // todo: can compute the distance to the camera (as it's the principal point it's the same)
            if( mp->isPixelInImage(tpix, tc)
                && (depth > 0.0f)
                && checkPair(p, rc, tc, mp, pc->minang, pc->maxang) )
            {
                cg = cg + tpix;
                cg3 = cg3 + p;
                ncg++;
            }
        }
    }
    if(ncg == 0)
    {
        return new StaticVector<float>();
    }
    cg = cg / (float)ncg;
    cg3 = cg3 / (float)ncg;
    allDepths = ncg;

    if(verbose == true)
    {
        ALICEVISION_LOG_DEBUG("All correct depths: " << allDepths);
    }

    Point2d midpoint = cg;
    if(midDepth > 0.0f)
    {
        Point3d midPt = rcplane.p + rcplane.n * midDepth;
        mp->getPixelFor3DPoint(&midpoint, midPt, tc);
    }

    // compute the direction
    float direction = 1.0f;
    {
        Point3d p;
        if(!triangulateMatch(p, rmid, midpoint, rc, tc, mp))
        {
            StaticVector<float>* out = new StaticVector<float>();
            return out;
        }

        float depth = orientedPointPlaneDistance(p, rcplane.p, rcplane.n);

        if(!triangulateMatch(p, rmid, midpoint + pixelVect, rc, tc, mp))
        {
            StaticVector<float>* out = new StaticVector<float>();
            return out;
        }

        float depthP1 = orientedPointPlaneDistance(p, rcplane.p, rcplane.n);
        if(depth > depthP1)
        {
            direction = -1.0f;
        }
    }

    StaticVector<float>* out1 = new StaticVector<float>();
    out1->reserve(2 * maxDepthsHalf);

    Point2d tpix = midpoint;
    float depthOld = -1.0f;
    int istep = 0;
    bool ok = true;

    // compute depths for all pixels from the middle point to on one side of the epipolar line
    while((out1->size() < maxDepthsHalf) && (mp->isPixelInImage(tpix, tc) == true) && (ok == true))
    {
        tpix = tpix + pixelVect * direction;

        Point3d refvect = mp->iCamArr[rc] * rmid;
        Point3d tarvect = mp->iCamArr[tc] * tpix;
        float rptpang = angleBetwV1andV2(refvect, tarvect);

        Point3d p;
        ok = triangulateMatch(p, rmid, tpix, rc, tc, mp);

        float depth = orientedPointPlaneDistance(p, rcplane.p, rcplane.n);
        if (mp->isPixelInImage(tpix, tc)
            && (depth > 0.0f) && (depth > depthOld)
            && checkPair(p, rc, tc, mp, pc->minang, pc->maxang)
            && (rptpang > pc->minang)  // WARNING if vects are near parallel thaen this results to strange angles ...
            && (rptpang < pc->maxang)) // this is the propper angle ... beacause is does not depend on the triangluated p
        {
            out1->push_back(depth);
            // if ((tpix.x!=tpixold.x)||(tpix.y!=tpixold.y)||(depthOld>=depth))
            //{
            // printf("after %f %f %f %f %i %f %f\n",tpix.x,tpix.y,depth,depthOld,istep,ang,kk);
            //};
        }
        else
        {
            ok = false;
        }
        depthOld = depth;
        istep++;
    }

    StaticVector<float>* out2 = new StaticVector<float>();
    out2->reserve(2 * maxDepthsHalf);
    tpix = midpoint;
    istep = 0;
    ok = true;

    // compute depths for all pixels from the middle point to the other side of the epipolar line
    while((out2->size() < maxDepthsHalf) && (mp->isPixelInImage(tpix, tc) == true) && (ok == true))
    {
        Point3d refvect = mp->iCamArr[rc] * rmid;
        Point3d tarvect = mp->iCamArr[tc] * tpix;
        float rptpang = angleBetwV1andV2(refvect, tarvect);

        Point3d p;
        ok = triangulateMatch(p, rmid, tpix, rc, tc, mp);

        float depth = orientedPointPlaneDistance(p, rcplane.p, rcplane.n);
        if(mp->isPixelInImage(tpix, tc)
            && (depth > 0.0f) && (depth < depthOld) 
            && checkPair(p, rc, tc, mp, pc->minang, pc->maxang)
            && (rptpang > pc->minang)  // WARNING if vects are near parallel thaen this results to strange angles ...
            && (rptpang < pc->maxang)) // this is the propper angle ... beacause is does not depend on the triangluated p
        {
            out2->push_back(depth);
            // printf("%f %f\n",tpix.x,tpix.y);
        }
        else
        {
            ok = false;
        }

        depthOld = depth;
        tpix = tpix - pixelVect * direction;
    }

    // printf("out2\n");
    StaticVector<float>* out = new StaticVector<float>();
    out->reserve(2 * maxDepthsHalf);
    for(int i = out2->size() - 1; i >= 0; i--)
    {
        out->push_back((*out2)[i]);
        // printf("%f\n",(*out2)[i]);
    }
    // printf("out1\n");
    for(int i = 0; i < out1->size(); i++)
    {
        out->push_back((*out1)[i]);
        // printf("%f\n",(*out1)[i]);
    }

    delete out2;
    delete out1;

    // we want to have it in ascending order
    if((*out)[0] > (*out)[out->size() - 1])
    {
        StaticVector<float>* outTmp = new StaticVector<float>();
        outTmp->reserve(out->size());
        for(int i = out->size() - 1; i >= 0; i--)
        {
            outTmp->push_back((*out)[i]);
        }
        delete out;
        out = outTmp;
    }

    // check if it is asc
    for(int i = 0; i < out->size() - 1; i++)
    {
        if((*out)[i] > (*out)[i + 1])
        {

            for(int j = 0; j <= i + 1; j++)
            {
                ALICEVISION_LOG_TRACE("getDepthsRcTc: check if it is asc: " << (*out)[j]);
            }
            ALICEVISION_LOG_WARNING("getDepthsRcTc: not asc");

            if(out->size() > 1)
            {
                qsort(&(*out)[0], out->size(), sizeof(float), qSortCompareFloatAsc);
            }
        }
    }

    if(verbose == true)
    {
        ALICEVISION_LOG_DEBUG("used depths: " << out->size());
    }

    return out;
}

bool PlaneSweepingCuda::refineRcTcDepthMap(bool useTcOrRcPixSize, int nStepsToRefine, StaticVector<float>* simMap,
                                             StaticVector<float>* rcDepthMap, int rc, int tc, int scale, int wsh,
                                             float gammaC, float gammaP, float epipShift, int xFrom, int wPart)
{
    // int w = mp->getWidth(rc)/scale;
    int w = wPart;
    int h = mp->getHeight(rc) / scale;

    long t1 = clock();

    StaticVector<int> camsids(2);
    camsids[0] = addCam(rc, scale, __FUNCTION__);

    if(verbose)
        ALICEVISION_LOG_DEBUG("\t- rc: " << rc << std::endl << "\t- tcams: " << tc);

    camsids[1] = addCam(tc, scale, __FUNCTION__);

    std::vector<cameraStruct*> ttcams( camsids.size() );
    for(int i = 0; i < camsids.size(); i++)
    {
        ttcams[i] = &cams[camsids[i]];
        ttcams[i]->camId = camsids[i];
        if(i == 0)
        {
            ttcams[i]->rc = rc;
        }
        else
        {
            ttcams[i]->rc = tc;
        }
    }

    // sweep
    ps_refineRcDepthMap(ps_texs_arr, simMap->getDataWritable().data(), rcDepthMap->getDataWritable().data(), nStepsToRefine,
                        ttcams,
                        w, h, mp->getWidth(rc) / scale,
                        mp->getHeight(rc) / scale, scale - 1, CUDADeviceNo, nImgsInGPUAtTime, scales, verbose, wsh,
                        gammaC, gammaP, epipShift, useTcOrRcPixSize, xFrom);

    /*
    CudaHostMemoryHeap<float, 3> tmpSimVolume_hmh(CudaSize<3>(201, 201, nStepsToRefine));

    ps_refineRcTcDepthMapSGM(
            &tmpSimVolume_hmh,
            &simMap_hmh,
            &rcDepthMap_hmh,
            nStepsToRefine,
            rcDepthMap_hmh,
            ttcams, camsids->size(),
            w, h,
            scale-1, scales,
            verbose, wsh, gammaC, gammaP, epipShift,
            0.0001f, 1.0f
    );
    */

    if(verbose)
        mvsUtils::printfElapsedTime(t1);

    return true;
}

float PlaneSweepingCuda::sweepPixelsToVolume( int nDepthsToSearch, StaticVector<float>& volume,
                                              int volDimX, int volDimY, int volDimZ,
                                              int volStepXY,
                                              StaticVector<float>* depths,
                                              int rc, int wsh, float gammaC,
                                              float gammaP,
                                              StaticVector<Voxel>* pixels, // use rcam image
                                              int scale, int step,
                                              const StaticVector<int>& tcams, // use tcam image
                                              float epipShift)
{
    if(verbose)
        ALICEVISION_LOG_DEBUG("sweepPixelsVolume:" << std::endl
                              << "\t- scale: " << scale << std::endl
                              << "\t- step: " << step << std::endl
                              << "\t- npixels: " << pixels->size() << std::endl
                              << "\t- volStepXY: " << volStepXY << std::endl
                              << "\t- volDimX: " << volDimX << std::endl
                              << "\t- volDimY: " << volDimY << std::endl
                              << "\t- volDimZ: " << volDimZ);

    int w = mp->getWidth(rc) / scale;
    int h = mp->getHeight(rc) / scale;

    long t1 = clock();

    if((tcams.size() == 0) || (pixels->size() == 0)) {
        return -1.0f;
    }

    StaticVector<int> camsids( tcams.size() + 1 );
    camsids[0] = addCam(rc, scale, __FUNCTION__);

    if(verbose)
        ALICEVISION_LOG_DEBUG("rc: " << rc << std::endl << "tcams: ");

    for(int c = 0; c < tcams.size(); ++c)
    {
        int tc = tcams[c];
        if(verbose)
            ALICEVISION_LOG_DEBUG("\t- " << tc);
        camsids[c+1] = addCam(tc, scale, __FUNCTION__);
    }

    std::vector<cameraStruct*> ttcams( camsids.size() );
    for(int i = 0; i < camsids.size(); i++) {
        ttcams[i] = &cams[camsids[i]];
        ttcams[i]->camId = camsids[i];
        if (i == 0) {
            ttcams[i]->rc = rc;
        } else {
            ttcams[i]->rc = tcams[i - 1];
        }
    }

#if 0
    CudaHostMemoryHeap<float, 1> depths_hmh(CudaSize<1>(depths->size()));

    for(int x = 0; x < depths->size(); x++)
    {
        depths_hmh(x) = (*depths)[x];
    }

    CudaDeviceMemory<float> depths_dev( depths_hmh );
#else
    const float* depth_data = depths->getData().data();
    CudaDeviceMemory<float> depths_dev( depth_data, depths->size() );
#endif

    // sweep
    float volumeMBinGPUMem = ps_planeSweepingGPUPixelsVolume(
        ps_texs_arr,
        volume.getDataWritable().data(),
        ttcams, w, h, volStepXY,
        volDimX, volDimY, volDimZ,
        depths_dev,
        nDepthsToSearch,
        wsh,
        nbestkernelSizeHalf,
        scale - 1, CUDADeviceNo, nImgsInGPUAtTime, scales, verbose, false, nbest,
        true, gammaC, gammaP, subPixel, epipShift);

    if(verbose)
        mvsUtils::printfElapsedTime(t1);

    return volumeMBinGPUMem;
}

/**
 * @param[inout] volume input similarity volume (after Z reduction)
 */
bool PlaneSweepingCuda::SGMoptimizeSimVolume(int rc, StaticVector<unsigned char>* volume, 
                                               int volDimX, int volDimY, int volDimZ, 
                                               int volStepXY, int scale,
                                               unsigned char P1, unsigned char P2)
{
    if(verbose)
        ALICEVISION_LOG_DEBUG("SGM optimizing volume:" << std::endl
                              << "\t- volDimX: " << volDimX << std::endl
                              << "\t- volDimY: " << volDimY << std::endl
                              << "\t- volDimZ: " << volDimZ);

    long t1 = clock();

    ps_SGMoptimizeSimVolume(ps_texs_arr,
                            cams[addCam(rc, scale, __FUNCTION__)],
                            volume->getDataWritable().data(), volDimX, volDimY, volDimZ, volStepXY,
                            verbose, P1, P2, scale - 1, // TODO: move the '- 1' inside the function
                            CUDADeviceNo, nImgsInGPUAtTime, scales);

    if(verbose)
        mvsUtils::printfElapsedTime(t1);

    return true;
}

// make_float3(avail,total,used)
Point3d PlaneSweepingCuda::getDeviceMemoryInfo()
{
    float3 dmif3 = ps_getDeviceMemoryInfo();
    return Point3d(dmif3.x, dmif3.y, dmif3.z);
}

bool PlaneSweepingCuda::fuseDepthSimMapsGaussianKernelVoting(int w, int h, StaticVector<DepthSim>* oDepthSimMap,
                                                               const StaticVector<StaticVector<DepthSim>*>* dataMaps,
                                                               int nSamplesHalf, int nDepthsToRefine, float sigma)
{
    long t1 = clock();

    // sweep
    CudaHostMemoryHeap<float2, 2>** dataMaps_hmh = new CudaHostMemoryHeap<float2, 2>*[dataMaps->size()];
    for(int i = 0; i < dataMaps->size(); i++)
    {
        dataMaps_hmh[i] = new CudaHostMemoryHeap<float2, 2>(CudaSize<2>(w, h));
        for(int y = 0; y < h; y++)
        {
            for(int x = 0; x < w; x++)
            {
                float2& data_hmh = (*dataMaps_hmh[i])(x, y);
                const DepthSim& data = (*(*dataMaps)[i])[y * w + x];
                data_hmh.x = data.depth;
                data_hmh.y = data.sim;
            }
        }
    }

    CudaHostMemoryHeap<float2, 2> oDepthSimMap_hmh(CudaSize<2>(w, h));

    ps_fuseDepthSimMapsGaussianKernelVoting(&oDepthSimMap_hmh, dataMaps_hmh, dataMaps->size(), nSamplesHalf,
                                            nDepthsToRefine, sigma, w, h, verbose);

    for(int y = 0; y < h; y++)
    {
        for(int x = 0; x < w; x++)
        {
            const float2& oDepthSim_hmh = oDepthSimMap_hmh(x, y);
            DepthSim& oDepthSim = (*oDepthSimMap)[y * w + x];
            oDepthSim.depth = oDepthSim_hmh.x;
            oDepthSim.sim = oDepthSim_hmh.y;
        }
    }

    for(int i = 0; i < dataMaps->size(); i++)
    {
        delete dataMaps_hmh[i];
    }
    delete[] dataMaps_hmh;

    if(verbose)
        mvsUtils::printfElapsedTime(t1);

    return true;
}

bool PlaneSweepingCuda::optimizeDepthSimMapGradientDescent(StaticVector<DepthSim>* oDepthSimMap,
                                                             StaticVector<StaticVector<DepthSim>*>* dataMaps, int rc,
                                                             int nSamplesHalf, int nDepthsToRefine, float sigma,
                                                             int nIters, int yFrom, int hPart)
{
    if(mp->verbose)
        ALICEVISION_LOG_DEBUG("optimizeDepthSimMapGradientDescent.");

    int scale = 1;
    int w = mp->getWidth(rc);
    int h = hPart;

    long t1 = clock();

    StaticVector<int> camsids;
    camsids.push_back(addCam(rc, scale, __FUNCTION__));
    if(verbose)
        printf("rc: %i, ", rc);

    cameraStruct** ttcams = new cameraStruct*[camsids.size()];
    for(int i = 0; i < camsids.size(); i++)
    {
        ttcams[i] = &cams[camsids[i]];
        ttcams[i]->camId = camsids[i];
        ttcams[i]->rc = rc;
    }

    // sweep
    CudaHostMemoryHeap<float2, 2>** dataMaps_hmh = new CudaHostMemoryHeap<float2, 2>*[dataMaps->size()];
    for(int i = 0; i < dataMaps->size(); i++)
    {
        dataMaps_hmh[i] = new CudaHostMemoryHeap<float2, 2>(CudaSize<2>(w, h));
        for(int y = 0; y < h; y++)
        {
            for(int x = 0; x < w; x++)
            {
                int jO = (y + yFrom) * w + x;
                float2& h_data = (*dataMaps_hmh[i])(x, y);
                const DepthSim& data = (*(*dataMaps)[i])[jO];
                h_data.x = data.depth;
                h_data.y = data.sim;
            }
        }
    }

    CudaHostMemoryHeap<float2, 2> oDepthSimMap_hmh(CudaSize<2>(w, h));

    ps_optimizeDepthSimMapGradientDescent(ps_texs_arr, &oDepthSimMap_hmh, dataMaps_hmh,
                                          dataMaps->size(), nSamplesHalf, nDepthsToRefine, nIters, sigma, ttcams,
                                          camsids.size(), w, h, scale - 1, CUDADeviceNo, nImgsInGPUAtTime, scales,
                                          verbose, yFrom);

    for(int y = 0; y < h; y++)
    {
        for(int x = 0; x < w; x++)
        {
            int jO = (y + yFrom) * w + x;
            DepthSim& oDepthSim = (*oDepthSimMap)[jO];
            const float2& h_oDepthSim = oDepthSimMap_hmh(x, y);

            oDepthSim.depth = h_oDepthSim.x;
            oDepthSim.sim = h_oDepthSim.y;
        }
    }

    for(int i = 0; i < dataMaps->size(); i++)
    {
        delete dataMaps_hmh[i];
    }
    delete[] dataMaps_hmh;

    for(int i = 0; i < camsids.size(); i++)
    {
        ttcams[i] = NULL;
    }
    delete[] ttcams;

    if(verbose)
        mvsUtils::printfElapsedTime(t1);

    return true;
}

bool PlaneSweepingCuda::getSilhoueteMap(StaticVectorBool* oMap, int scale, int step, const rgb maskColor, int rc)
{
    if(verbose)
        ALICEVISION_LOG_DEBUG("getSilhoueteeMap: rc: " << rc);

    int w = mp->getWidth(rc) / scale;
    int h = mp->getHeight(rc) / scale;

    long t1 = clock();

    int camId = addCam(rc, scale, __FUNCTION__);

    uchar4 maskColorRgb;
    maskColorRgb.x = maskColor.r;
    maskColorRgb.y = maskColor.g;
    maskColorRgb.z = maskColor.b;
    maskColorRgb.w = 1.0f;

    CudaHostMemoryHeap<bool, 2> omap_hmh(CudaSize<2>(w / step, h / step));

    ps_getSilhoueteMap(ps_texs_arr, &omap_hmh, w, h, scale - 1, CUDADeviceNo,
                       nImgsInGPUAtTime, scales, step, camId, maskColorRgb, verbose);

    for(int i = 0; i < (w / step) * (h / step); i++)
    {
        (*oMap)[i] = omap_hmh.getBuffer()[i];
    }

    if(verbose)
        mvsUtils::printfElapsedTime(t1);

    return true;
}

int listCUDADevices(bool verbose)
{
    return ps_listCUDADevices(verbose);
}

} // namespace depthMap
} // namespace aliceVision
