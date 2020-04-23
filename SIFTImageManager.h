#pragma once

#ifndef _IMAGE_MANAGER_H_
#define _IMAGE_MANAGER_H_

#include <iostream>

#include <opencv2/opencv.hpp>

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
        minKeyScale=3.0f;
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
        vkps1=GetOpenCVKeypoints(imageGPU1);
    }

    void RunSIFT2(const unsigned char* cpu_color_uchar_ptr,ushort* cpu_depth_ushort_ptr, float depth_scale){
        resampleColorToIntensity(dev_color2,cpu_color_uchar_ptr,color_width,color_height);
        resampleDepthToFloat(dev_depth2,cpu_depth_ushort_ptr,depth_width,depth_height,depth_scale);
        m_sift->RunSIFT(dev_color2,dev_depth2);
        numKeypoints2 = m_sift->GetKeyPointsAndDescriptorsCUDA(imageGPU2, dev_depth2, maxNumKeypointsPerImage);
        std::cout<<"Extract 2 sift features = "<<numKeypoints2<<std::endl;
        vkps2=GetOpenCVKeypoints(imageGPU2);
    }

    std::vector<cv::DMatch> Match(){
        m_siftMatcher->SetDescriptors(0, numKeypoints1, (unsigned char*)imageGPU1.d_keyPointDescs);
        m_siftMatcher->SetDescriptors(1, numKeypoints2, (unsigned char*)imageGPU2.d_keyPointDescs);
        m_siftMatcher->GetSiftMatch(numKeypoints1, imagePairMatch, make_uint2(0,0), siftMatchThresh, ratioMax);

        //just for show inspect match
        int cpu_numMatches=-1;
        float cpu_distances[maxImageMatches*128];
        uint2 cpu_keyPointIndices[maxImageMatches*128];
        cutilSafeCall(cudaMemcpy(&cpu_numMatches, imagePairMatch.d_numMatches, sizeof(int)*maxImageMatches, cudaMemcpyDeviceToHost));
        cutilSafeCall(cudaMemcpy(cpu_distances, imagePairMatch.d_distances, sizeof(float)*maxImageMatches*128, cudaMemcpyDeviceToHost));
        cutilSafeCall(cudaMemcpy(cpu_keyPointIndices, imagePairMatch.d_keyPointIndices, sizeof(uint)*maxImageMatches*128, cudaMemcpyDeviceToHost));
        std::cout<<" Matches num = "<<cpu_numMatches<<std::endl;

        std::vector<cv::DMatch> matches;
        for(int i=0;i<cpu_numMatches;i++){
//            std::cout<<cpu_distances[i]<<" "<<std::endl;
            std::cout<<"[ "<<cpu_keyPointIndices[i].x<<" , "<<cpu_keyPointIndices[i].y<<" ]   ";
//            int id=GetKeypointIDbyXY(cpu_keyPointIndices[i].x, cpu_keyPointIndices[i].y);
            matches.push_back(cv::DMatch(cpu_keyPointIndices[i].x,cpu_keyPointIndices[i].y,2,cpu_distances[i]));
        }
        return matches;
    }

    std::vector<cv::KeyPoint> GetOpenCVKeypoints(SIFTImageGPU image_gpu){
        //将cuda中的特征点传回host
        SIFTKeyPoint kps[maxNumKeypointsPerImage];
        cutilSafeCall(cudaMemcpy(kps, image_gpu.d_keyPoints, sizeof(int)*maxNumKeypointsPerImage, cudaMemcpyDeviceToHost));
        int kps_num;
        cutilSafeCall(cudaMemcpy(&kps_num, image_gpu.d_keyPointCounter, sizeof(int), cudaMemcpyDeviceToHost));
        //生成opencv格式的关键点坐标
        std::vector<cv::KeyPoint> vkps;
        for(int i=0;i<kps_num;i++){
            vkps.push_back(cv::KeyPoint(kps[i].pos.x,kps[i].pos.y,1 )); //kp size default =1
//            std::cout<<kps[i].pos.x<<" "<<kps[i].pos.y<<std::endl;
            std::cout<<"[ "<<kps[i].pos.x<<" , "<<kps[i].pos.y<<" ]   ";

        }
        std::cout<<std::endl;
        return vkps;
    }
    int GetKeypointIDbyXY(float x, float y){
        std::cout<<"Process frame ---"<<x<<"---"<<y<<"----"<<std::endl;
        for(int j=0;j<numKeypoints2;j++){
            if(x==vkps2[j].pt.x && y==vkps2[j].pt.y){
                return j;
            }
        }
        std::cout<<"[ "<<x<<" , "<<y<<" ] is not in img2"<<std::endl;
        return -1;
    }

    std::vector<cv::KeyPoint> vkps1;
    std::vector<cv::KeyPoint> vkps2;


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
    SIFTImageGPU imageGPU1; //结构体对象是建立在host内存当中的，但是成员指针却是指向gpu显存的
    SIFTImageGPU imageGPU2;
    ImagePairMatch imagePairMatch;

    float* dev_color1;
    float* dev_depth1;
    float* dev_color2;
    float* dev_depth2;
    int numKeypoints1=-1;
    int numKeypoints2=-1;


    void allocCUDA(){
        //分配match
        cutilSafeCall(cudaMalloc(&(imagePairMatch.d_numMatches),sizeof(int)*maxImageMatches));   //设置一个图最多匹配几个图
        cutilSafeCall(cudaMalloc(&(imagePairMatch.d_distances),sizeof(float)*maxImageMatches*maxMatchesPerImagePairRaw));   //总共存储128个距离
        cutilSafeCall(cudaMalloc(&(imagePairMatch.d_keyPointIndices),sizeof(uint2)*maxImageMatches*maxMatchesPerImagePairRaw));
        //alloc imageGPU1
        cutilSafeCall(cudaMalloc(&(imageGPU1.d_keyPoints),sizeof(SIFTKeyPoint)*maxNumKeypointsPerImage));   //设置获取4096个关键点
        cutilSafeCall(cudaMalloc(&(imageGPU1.d_keyPointDescs),sizeof(SIFTKeyPointDesc)*maxNumKeypointsPerImage));   //设置获取4096个关键点
        cutilSafeCall(cudaMalloc(&(imageGPU1.d_keyPointCounter),sizeof(int)));   //设置该图像的特征点个数
        //alloc imageGPU2
        cutilSafeCall(cudaMalloc(&(imageGPU2.d_keyPoints),sizeof(SIFTKeyPoint)*maxNumKeypointsPerImage));   //设置获取4096个关键点
        cutilSafeCall(cudaMalloc(&(imageGPU2.d_keyPointDescs),sizeof(SIFTKeyPointDesc)*maxNumKeypointsPerImage));   //设置获取4096个关键点
        cutilSafeCall(cudaMalloc(&(imageGPU2.d_keyPointCounter),sizeof(int)));   //设置该图像的特征点个数
        //rgbd1
        cutilSafeCall(cudaMalloc(&dev_color1,sizeof(float)*color_width*color_height));   //设置获取4096个关键点
        cutilSafeCall(cudaMalloc(&dev_depth1,sizeof(float)*depth_width*depth_height));   //设置获取4096个关键点
        //rgbd2
        cutilSafeCall(cudaMalloc(&dev_color2,sizeof(float)*color_width*color_height));   //设置获取4096个关键点
        cutilSafeCall(cudaMalloc(&dev_depth2,sizeof(float)*depth_width*depth_height));   //设置获取4096个关键点
    }
    void freeCUDA(){
        cudaFree(imagePairMatch.d_numMatches);
        cudaFree(imagePairMatch.d_distances);
        cudaFree(imagePairMatch.d_keyPointIndices);

        //imageGPU1
        cudaFree(imageGPU1.d_keyPoints);
        cudaFree(imageGPU1.d_keyPointDescs);
        cudaFree(imageGPU1.d_keyPointCounter);
        //imageGPU2
        cudaFree(imageGPU2.d_keyPoints);
        cudaFree(imageGPU2.d_keyPointDescs);
        cudaFree(imageGPU2.d_keyPointCounter);

        cudaFree(dev_color1);
        cudaFree(dev_depth1);

        cudaFree(dev_color2);
        cudaFree(dev_depth2);

    }

    static void resampleColorToIntensity(float* dev_output, const unsigned char* cpu_input, unsigned int inputWidth, unsigned int inputHeight) {
        float* cpu_output=new float[inputWidth*inputHeight*3];
        unsigned int ncols = inputWidth *3;
        unsigned int nrows = inputHeight ;    //rgb has 3 channels
        for (int i = 0; i < nrows*ncols; i = i + 3) {
            float tmp = (0.299f*cpu_input[i] + 0.587f*cpu_input[i + 1] + 0.114f*cpu_input[i + 2]) / 255.0f;   //输出0-1之间的值
            cpu_output[i / 3] = tmp;
//            std::cout<<tmp<<" ";
//            if(i%2048==0) std::cout<<std::endl;
        }
        cutilSafeCall(cudaMemcpy(dev_output, cpu_output, sizeof(float)*inputWidth*inputHeight, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        delete[] cpu_output;
    }

    static void resampleDepthToFloat(float* dev_output, ushort* cpu_input, unsigned int inputWidth, unsigned int inputHeight, float depth_scale) {
        float* cpu_output=new float[inputWidth*inputHeight];
        unsigned int ncols = inputWidth;
        unsigned int nrows = inputHeight;
        int cnt = 0;
        for (int i = 0; i < nrows*ncols; i++) {
            float tmp = static_cast<float>(cpu_input[i]) / depth_scale;
            cpu_output[i] = tmp;
        }
        cutilSafeCall(cudaMemcpy(dev_output, cpu_output, sizeof(float)*inputWidth*inputHeight, cudaMemcpyHostToDevice));
        delete[] cpu_output;
    }

};


#endif

