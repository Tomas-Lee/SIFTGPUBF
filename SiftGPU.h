#ifndef GPU_SIFT_H
#define GPU_SIFT_H

#include <vector>
#include "cuda_runtime.h"
#include "GlobalDefines.h"

struct SIFTKeyPoint {
    float2 pos;
    float scale;
    float depth;
};

struct SIFTKeyPointDesc {
    unsigned char feature[128];
};

struct SIFTImageGPU {
    int* 					d_keyPointCounter;	//single counter value per image (into into global array) 这里我的定义跟原版不同，存储已经提取到的特征点个数
    SIFTKeyPoint*			d_keyPoints;		//array of key points (index into global array)
    SIFTKeyPointDesc*		d_keyPointDescs;	//array of key point descs (index into global array)
};

struct ImagePairMatch {
    int*		d_numMatches;		//single counter value per image, 存储基本参考图像跟索引图像的匹配数目，这只是一个int[1]的数组
    float*		d_distances;		//array of distance (one per match)     //
    uint2*		d_keyPointIndices;	//array of index pair (one per match)   //存储该参考图像的每个点跟索引图像哪个点最匹配，个数为参考图像的关键点个数
};

//correspondence_idx -> image_Idx_i,j
struct EntryJ {
    unsigned int imgIdx_i;
    unsigned int imgIdx_j;
    float3 pos_i;
    float3 pos_j;

    __host__ __device__
    void setInvalid() {
        imgIdx_i = (unsigned int)-1;
        imgIdx_j = (unsigned int)-1;
    }
    __host__ __device__
    bool isValid() const {
        return imgIdx_i != (unsigned int)-1;
    }
};


///////////////////////////////////////////////////////////////////
//clss SiftParam
//description: SIFT parameters SiftParam和GlobalUtil均存储的是sift特征点提取相关信息
////////////////////////////////////////////////////////////////////
class GlobalUtil;
class SiftParam
{
public:
	SiftParam();
	~SiftParam() {
        if (_sigma) { delete[] (_sigma);   (_sigma)=NULL; }
	}

	void ParseSiftParam();  //

	float GetLevelSigma(int lev);
	float GetInitialSmoothSigma(int octave_min);

	std::vector<unsigned int> m_filterWidths;

	float*		_sigma; //sigma0=1.22627
	float		_sigma_skip0; // 1.51987
	float		_sigma_skip1; //0
	
	//sigma of the first level
	float		_sigma0;    //2.01587
	float		_sigman;    //0.5
	int			_sigma_num; //5

	//how many dog_level in an octave
	int			_dog_level_num; //3
	int			_level_num; //6

	//starting level in an octave
	int			_level_min; //-1
	int			_level_max; //4
	int			_level_ds;  //2
	//dog threshold
	float		_dog_threshold; //0.006666667
	//edge elimination
	float		_edge_threshold;    //10
};

class SiftPyramid;
////////////////////////////////////////////////////////////////
//class SIftGPU
//description: Interface of SiftGPU lib
////////////////////////////////////////////////////////////////
class SiftGPU:public SiftParam
{
public:
	typedef struct SiftKeypoint
	{
		float x, y, s, o; //x, y, scale, orientation.
	}SiftKeypoint;
public:
	//constructor, the parameter np is ignored..
	SiftGPU();
	//destructor
	~SiftGPU();


	//Initialize OpenGL and SIFT paremeters, and create the shaders accordingly
	void InitSiftGPU();
	//get the number of SIFT features in current image
	 int	GetFeatureNum();


	//get sift keypoints & descriptors (compute into provided d_keypoints, d_descriptors)
	 unsigned int GetKeyPointsAndDescriptorsCUDA(SIFTImageGPU& siftImage, const float* d_depthData, unsigned int maxNumKeyPoints = (unsigned int)-1);
	//get sift keypoints (compute into provided d_keypoints)
	 void GetKeyPointsCUDA(SiftKeypoint* d_keypoints, float* d_depthData, unsigned int maxNumKeyPoints = (unsigned int)-1);
	//get sift descriptors (compute into provided d_descriptors)
	 void GetDescriptorsCUDA(unsigned char* d_descriptors, unsigned int maxNumKeyPoints = (unsigned int)-1);

	//Copy the SIFT result to two vectors
	// void CopyFeatureVectorToCPU(SiftKeypoint * keys, float * descriptors);
	//parse SiftGPU parameters
	 void SetParams(unsigned int siftWidth, unsigned int siftHeight, bool enableTiming, unsigned int featureCountThreshold, float siftDepthMin, float siftDepthMax);

	int RunSIFT(float* d_colorData, const float* d_depthData);
	//set the active pyramid...dropped function
     void SetActivePyramid(int index) {}
	//allocate pyramid for a given size of image
	 int AllocatePyramid(int width, int height);
	//none of the texture in processing can be larger
	//automatic down-sample is used if necessary. 
	void SetMaxDimension(int sz);

	void EvaluateTimings();
private:
	//when more than one images are specified
	//_current indicates the active one
	int		_current;
	//_initialized indicates if the shaders and OpenGL/SIFT parameters are initialized 标示是否已经初始化
	//they are initialized only once for one SiftGPU inistance 只初始化一次
	//that is, SIFT parameters will not be changed
	int		_initialized;
	//_image_loaded indicates if the current images are loaded
	int		_image_loaded;
	//the SiftPyramid
	SiftPyramid *  _pyramid;
	//print out the command line options
	static void PrintUsage();

};



#endif 
