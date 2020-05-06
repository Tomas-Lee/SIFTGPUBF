#include <iostream>
#include "SIFTImageManager.h"

#include <opencv2/opencv.hpp>
#include "glog/logging.h"
using cv::Mat;

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    google::InstallFailureSignalHandler();

    SIFTImageManager siftImageManager(640,480);



//	//处理输入数据1
    Mat color1 = cv::imread("../color0.png", -1);	//8UC3
    Mat depth1 = cv::imread("../depth0.png", -1);	//16UC1
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
    Mat color2 = cv::imread("../color1.png", -1);	//8UC3
    Mat depth2 = cv::imread("../depth1.png", -1);	//16UC1
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
    siftImageManager.RunSIFT1(color1_ptr,depth1_ptr,5000);
    siftImageManager.RunSIFT2(color2_ptr,depth2_ptr,5000);
    std::vector<cv::DMatch> matches=siftImageManager.Match();

//    cv::Mat kpmat1;
//    cv::drawKeypoints(color1,siftImageManager.vkps1,kpmat1);
//    cv::imwrite("kpmat0.png",kpmat1);
//    cv::imshow("kps1",kpmat1);
//    cv::waitKey();
//
//    cv::Mat kpmat2;
//    cv::drawKeypoints(color2,siftImageManager.vkps2,kpmat2);
//    cv::imwrite("kpmat1.png",kpmat2);
//    cv::imshow("kps2",kpmat2);
//    cv::waitKey();

//    cv::Mat matchMat;
//    cv::drawMatches(color1,siftImageManager.vkps1,color2,siftImageManager.vkps2,matches,matchMat);
//    cv::imwrite("matches01.png",matchMat);
//    cv::imshow("matches",matchMat);
//    cv::waitKey();


    std::cout << "HelloWorld" << std::endl;
}