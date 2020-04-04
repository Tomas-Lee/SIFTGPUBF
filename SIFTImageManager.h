#pragma once

#ifndef _IMAGE_MANAGER_H_
#define _IMAGE_MANAGER_H_

#include <iostream>
#include "SiftCameraParams.h"
#include "SiftGPU.h"
#include "SiftMatch.h"

extern "C" void updateConstantSiftCameraParams(const SiftCameraParams& params);

class SIFTImageManager {
public:
    SIFTImageManager(int width,int height){
        color_width=width;
        color_height=height;
        depth_width=width;
        depth_height=height;
        minKeyScale=3;
        sensorDepthMin=0.1;
        sensorDepthMax=4;
        featureCountThreshold=150;
        enableTiming=false;
        maxNumKeypointsPerImage=1024;
        maxImageMatches=1;
        maxMatchesPerImagePairRaw=128;
        ratioMax=0.8;
        siftMatchThresh=0.7;

        Init();
    }
    ~SIFTImageManager(){
        freeCUDA();
    }
    void Init(){
        SiftCameraParams siftCameraParams;
        siftCameraParams.m_depthWidth = depth_width;
        siftCameraParams.m_depthHeight = depth_height;
        siftCameraParams.m_intensityWidth = color_width;
        siftCameraParams.m_intensityHeight = color_height;
        siftCameraParams.m_minKeyScale = minKeyScale; //GlobalBundlingState::get().s_minKeyScale;
        updateConstantSiftCameraParams(siftCameraParams);

        m_sift = new SiftGPU;
        m_sift->SetParams(color_width, color_height, enableTiming, featureCountThreshold, sensorDepthMin, sensorDepthMax);
        m_sift->InitSiftGPU();

        m_siftMatcher = new SiftMatchGPU(maxNumKeypointsPerImage);
        m_siftMatcher->InitSiftMatch();

        allocCUDA();
    }

    void RunSIFT1(const unsigned char* cpu_color_uchar_ptr,ushort* cpu_depth_ushort_ptr, float depth_scale){
        resampleColorToIntensity(dev_color1,cpu_color_uchar_ptr,color_width,color_height);
        resampleDepthToFloat(dev_depth1,cpu_depth_ushort_ptr,depth_width,depth_height,depth_scale);
        m_sift->RunSIFT(dev_color1,dev_depth1);
        numKeypoints1 = m_sift->GetKeyPointsAndDescriptorsCUDA(imageGPU1, dev_depth1, maxNumKeypointsPerImage);
        std::cout<<"Extract 1 sift features = "<<numKeypoints1<<std::endl;
    }

    void RunSIFT2(const unsigned char* cpu_color_uchar_ptr,ushort* cpu_depth_ushort_ptr, float depth_scale){
        resampleColorToIntensity(dev_color2,cpu_color_uchar_ptr,color_width,color_height);
        resampleDepthToFloat(dev_depth2,cpu_depth_ushort_ptr,depth_width,depth_height,depth_scale);
        m_sift->RunSIFT(dev_color2,dev_depth2);
        numKeypoints2 = m_sift->GetKeyPointsAndDescriptorsCUDA(imageGPU2, dev_depth2, maxNumKeypointsPerImage);
        std::cout<<"Extract 2 sift features = "<<numKeypoints2<<std::endl;
    }

    void Match(){
        m_siftMatcher->SetDescriptors(0, numKeypoints1, (unsigned char*)imageGPU1.d_keyPointDescs);
        m_siftMatcher->SetDescriptors(1, numKeypoints2, (unsigned char*)imageGPU2.d_keyPointDescs);
        m_siftMatcher->GetSiftMatch(numKeypoints1, imagePairMatch, make_uint2(0,numKeypoints1), siftMatchThresh, ratioMax);

        //just for show inspect match
        int cpu_numMatches=-1;
        float cpu_distances[maxImageMatches*128];
        uint2 cpu_keyPointIndices[maxImageMatches*128];
        cutilSafeCall(cudaMemcpy(&cpu_numMatches, imagePairMatch.d_numMatches, sizeof(int)*maxImageMatches, cudaMemcpyDeviceToHost));
        cutilSafeCall(cudaMemcpy(cpu_distances, imagePairMatch.d_distances, sizeof(float)*maxImageMatches*128, cudaMemcpyDeviceToHost));
        cutilSafeCall(cudaMemcpy(cpu_keyPointIndices, imagePairMatch.d_keyPointIndices, sizeof(uint)*maxImageMatches*128, cudaMemcpyDeviceToHost));
        for(int i=0;i<128;i++){
//        cout<<cpu_distances[i]<<" ";
            std::cout<<"[ "<<cpu_keyPointIndices[i].x<<" , "<<cpu_keyPointIndices[i].y<<" ]   ";
        }
    }

private:
    //param
    int color_width;
    int color_height;
    int depth_width;
    int depth_height;
    float minKeyScale=3;
    float sensorDepthMin=0.1;
    float sensorDepthMax=4;
    int featureCountThreshold=150;
    bool enableTiming=false;
    int maxNumKeypointsPerImage=1024;
    int maxImageMatches=1;
    int maxMatchesPerImagePairRaw=128;
    float ratioMax=0.8;
    float siftMatchThresh=0.7;

    //
    SiftGPU* m_sift;
    SiftMatchGPU* m_siftMatcher;
    SIFTImageGPU imageGPU1; //in gpu
    SIFTImageGPU imageGPU2;
    ImagePairMatch imagePairMatch;

    float* dev_color1;
    float* dev_depth1;
    float* dev_color2;
    float* dev_depth2;
    int numKeypoints1=-1;
    int numKeypoints2=-1;

    void allocCUDA(){
        cutilSafeCall(cudaMalloc(&(imagePairMatch.d_numMatches),sizeof(int)*maxImageMatches));   //设置获取1024个关键点
        cutilSafeCall(cudaMalloc(&(imagePairMatch.d_distances),sizeof(float)*maxImageMatches*maxMatchesPerImagePairRaw));   //总共存储128个距离
        cutilSafeCall(cudaMalloc(&(imagePairMatch.d_keyPointIndices),sizeof(uint2)*maxImageMatches*maxMatchesPerImagePairRaw));   //设置获取1024个关键点

        cutilSafeCall(cudaMalloc(&(imageGPU1.d_keyPoints),sizeof(SIFTKeyPoint)*maxMatchesPerImagePairRaw));   //设置获取1024个关键点
        cutilSafeCall(cudaMalloc(&(imageGPU1.d_keyPointDescs),sizeof(SIFTKeyPointDesc)*maxMatchesPerImagePairRaw));   //设置获取1024个关键点

        cutilSafeCall(cudaMalloc(&(imageGPU2.d_keyPoints),sizeof(SIFTKeyPoint)*maxMatchesPerImagePairRaw));   //设置获取1024个关键点
        cutilSafeCall(cudaMalloc(&(imageGPU2.d_keyPointDescs),sizeof(SIFTKeyPointDesc)*maxMatchesPerImagePairRaw));   //设置获取1024个关键点

        cutilSafeCall(cudaMalloc(&dev_color1,sizeof(float)*color_width*color_height));   //设置获取1024个关键点
        cutilSafeCall(cudaMalloc(&dev_depth1,sizeof(float)*depth_width*depth_height));   //设置获取1024个关键点

        cutilSafeCall(cudaMalloc(&dev_color2,sizeof(float)*color_width*color_height));   //设置获取1024个关键点
        cutilSafeCall(cudaMalloc(&dev_depth2,sizeof(float)*depth_width*depth_height));   //设置获取1024个关键点
    }
    void freeCUDA(){
        cudaFree(imagePairMatch.d_numMatches);
        cudaFree(imagePairMatch.d_distances);
        cudaFree(imagePairMatch.d_keyPointIndices);

        cudaFree(imageGPU1.d_keyPoints);
        cudaFree(imageGPU1.d_keyPointDescs);

        cudaFree(imageGPU2.d_keyPoints);
        cudaFree(imageGPU2.d_keyPointDescs);

        cudaFree(dev_color1);
        cudaFree(dev_depth1);

        cudaFree(dev_color2);
        cudaFree(dev_depth2);

    }

    void resampleColorToIntensity(float* dev_output, const unsigned char* cpu_input, unsigned int inputWidth, unsigned int inputHeight) {
        float* cpu_output=new float[inputWidth*inputHeight*3];
        int ncols = inputWidth *3;
        int nrows = inputHeight ;    //rgb has 3 channels
        for (int i = 0; i < nrows*ncols; i = i + 3) {
            float tmp = (0.299f*cpu_input[i] + 0.587f*cpu_input[i + 1] + 0.114f*cpu_input[i + 2]) / 255.0f;   //输出0-1之间的值
            cpu_output[i / 3] = tmp;
//            std::cout<<tmp<<" ";
//            if(i%2048==0) std::cout<<std::endl;
        }
        cutilSafeCall(cudaMemcpy(dev_output, cpu_output, sizeof(float)*inputWidth*inputHeight, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        delete cpu_output;
    }

    void resampleDepthToFloat(float* dev_output, ushort* cpu_input, unsigned int inputWidth, unsigned int inputHeight, float depth_scale) {
        float* cpu_output=new float[inputWidth*inputHeight];
        int ncols = inputWidth;
        int nrows = inputHeight;
        int cnt = 0;
        for (int i = 0; i < nrows*ncols; i++) {
            float tmp = static_cast<float>(cpu_input[i]) / depth_scale;
            cpu_output[i] = tmp;
        }
        cutilSafeCall(cudaMemcpy(dev_output, cpu_output, sizeof(float)*inputWidth*inputHeight, cudaMemcpyHostToDevice));
        delete cpu_output;
    }

};


#endif

