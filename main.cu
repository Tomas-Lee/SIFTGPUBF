#include <iostream>
#include "SIFTImageManager.h"

#include <opencv2/opencv.hpp>
#include "glog/logging.h"
using cv::Mat;

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    google::InstallFailureSignalHandler();

    SIFTImageManager siftImageManager(2048,1536);



//	//处理输入数据1
    Mat color1 = cv::imread("../rgb2048.png", -1);	//8UC3
    Mat depth1 = cv::imread("../depth2048.png", -1);	//16UC1
    if (color1.empty() || depth1.empty()) {
        std::cerr << "Read image fail" << std::endl;
        return 0;
    }
    int rows=color1.rows;
    int cols=color1.cols;
    // ushort /uchar to float
    ushort* depth1_ptr = reinterpret_cast<ushort*>(depth1.data);
    uchar* color1_ptr = color1.data;    //default channel order is BGR

//	//处理输入数据2
    Mat color2 = cv::imread("../rgb2048_10.png", -1);	//8UC3
    Mat depth2 = cv::imread("../depth2048_10.png", -1);	//16UC1
    if (color2.empty() || depth2.empty()) {
        std::cerr << "Read image fail" << std::endl;
        return 0;
    }
//    int rows=color1.rows;
//    int cols=color1.cols;
    // ushort /uchar to float
    ushort* depth2_ptr = reinterpret_cast<ushort*>(depth2.data);
    uchar* color2_ptr = color2.data;    //default channel order is BGR


    //run sift1
    siftImageManager.RunSIFT1(color1_ptr,depth1_ptr,1000);
    siftImageManager.RunSIFT2(color2_ptr,depth2_ptr,1000);
    siftImageManager.Match();

    std::cout << "HelloWorld" << std::endl;
}