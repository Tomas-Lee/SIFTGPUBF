#ifndef _PROGRAM_CU_H
#define _PROGRAM_CU_H
//可以理解为这是一个工具函数库，专门实现各种cuda加速功能
class CuTexImage;

class ProgramCU
{
public:
    //GPU FUNCTIONS
	static int  CheckErrorCUDA(const char* location);
    static int  CheckCudaDevice(int device);

	static void InitFilterKernels(const std::vector<float>& sigmas, std::vector<unsigned int>& filterWidths);
    ////SIFTGPU FUNCTIONS
	static void CreateFilterKernel(float sigma, float* kernel, int& width);
	template<int KWIDTH> static void FilterImage(CuTexImage *dst, CuTexImage *src, CuTexImage* buf, unsigned int filterIndex);
	static void FilterImage(CuTexImage *dst, CuTexImage *src, CuTexImage* buf, unsigned int width, unsigned int filterIndex);
	//static void FilterImage(CuTexImage *dst, CuTexImage *src, CuTexImage* buf, float sigma);
	static void ComputeDOG(CuTexImage* gus, CuTexImage* dog, CuTexImage* got);
	static void ComputeKEY(CuTexImage* dog, CuTexImage* key, float Tdog, float Tedge, CuTexImage* featureList, int* d_featureCount, unsigned int featureOctLevelidx, float keyLocScale, float keyLocOffset, const float* d_depthData, float siftDepthMin, float siftDepthMax);
	static void InitHistogram(CuTexImage* key, CuTexImage* hist);
	static void ReduceHistogram(CuTexImage*hist1, CuTexImage* hist2);
	static void GenerateList(CuTexImage* list, CuTexImage* hist);		//TODO get rid of it doesn't seem to be used
	static unsigned int ReshapeFeatureList(CuTexImage* raw, CuTexImage* out, int* d_featureCount, float keyLocScale);	//returns the number of features
	static void ComputeOrientation(CuTexImage*list, CuTexImage* got, CuTexImage*key, float sigma, float sigma_step);
	static void ComputeDescriptor(CuTexImage*list, CuTexImage* got, float* d_outDescriptors, int rect = 0, int stream = 0);
	static void CreateGlobalKeyPointList(CuTexImage* curLevelList, float4* d_outKeypointList, float keyLocScale, float keyLocOffset, const float* d_depthData, int maxNumElements);	//returns the number of features

    //data conversion
	static void SampleImageU(CuTexImage *dst, CuTexImage *src, int log_scale);
	static void SampleImageD(CuTexImage *dst, CuTexImage *src, int log_scale = 1); 
	static void ReduceToSingleChannel(CuTexImage* dst, CuTexImage* src, int convert_rgb);
    static void ConvertByteToFloat(CuTexImage*src, CuTexImage* dst);
    
    //visualization
	static void DisplayConvertDOG(CuTexImage* dog, CuTexImage* out);
	static void DisplayConvertGRD(CuTexImage* got, CuTexImage* out);
	static void DisplayConvertKEY(CuTexImage* key, CuTexImage* dog, CuTexImage* out);
	static void DisplayKeyPoint(CuTexImage* ftex, CuTexImage* out);
	static void DisplayKeyBox(CuTexImage* ftex, CuTexImage* out);
	
	//SIFTMATCH FUNCTIONS	
	static void MultiplyDescriptor(CuTexImage* tex1, CuTexImage* tex2, CuTexImage* texDot, CuTexImage* texCRT);
	static void MultiplyDescriptorG(CuTexImage* texDes1, CuTexImage* texDes2,
		CuTexImage* texLoc1, CuTexImage* texLoc2, CuTexImage* texDot, CuTexImage* texCRT,
		float H[3][3], float hdistmax, float F[3][3], float fdistmax);
	static void GetRowMatch(CuTexImage* texDot, CuTexImage* texMatch, float* d_matchDistances, float distmax, float ratiomax);
	static void GetColMatch(CuTexImage* texCRT, float distmax, float ratiomax, CuTexImage* rowMatch, float* d_matchDistances, uint2* d_outKeyPointIndices, float* d_outMatchDistances, int* d_numMatches, uint2 keyPointOffset, int* numMatches = NULL);

	static void ConvertDescriptorToUChar(float* d_descriptorsFloat, unsigned int numDescriptorElements, unsigned char* d_descriptorsUChar);
};

#endif

