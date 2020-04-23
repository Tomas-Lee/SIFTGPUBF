#ifndef CU_TEX_IMAGE_H
#define CU_TEX_IMAGE_H

struct cudaArray;   //声明有一个叫cudaArray的结构体
struct textureReference;

//using texture2D from linear memory

#define SIFTGPU_ENABLE_LINEAR_TEX2D

class CuTexImage
{
public:
	CuTexImage();
	~CuTexImage();
	friend class ProgramCU;

	void setImageData(int width, int height, int numChannels, void* d_data) {
		_texWidth = _imgWidth = width;
		_texHeight = _imgHeight = height;
		_numChannel = numChannels;
		_numBytes = _numChannel * _imgWidth * _imgHeight * sizeof(float);

		_cuData = d_data;
		m_external = true;
	}

	void SetImageSize(int width, int height);
	void InitTexture(int width, int height, int nchannel = 1);
	void InitTexture2D();
	inline void BindTexture(textureReference& texRef, size_t* offset = NULL);
	inline void BindTexture2D(textureReference& texRef);
	void CopyToTexture2D();
	void CopyToHost(void* buf);
	void CopyToHost(void* buf, int stream);
	void CopyFromHost(const void* buf);
	void CopyToDevice(CuTexImage* other) const;
	static int DebugCopyToTexture2D();
	
	void Memset(int value = 0);
	inline int GetImgWidth(){ return _imgWidth; }
	inline int GetImgHeight(){ return _imgHeight; }
	inline int GetDataSize(){ return _numBytes; }
	inline int GetImgNumChannels(){ return _numChannel; }


private:
	void*		_cuData;    //存储在GPU上的图像数据
	cudaArray*	_cuData2D;  //cudaArray is used to bind to texture memory
	int			_numChannel;
	int			_numBytes;
	int			_imgWidth;
	int			_imgHeight;
	int			_texWidth;
	int			_texHeight;

	bool m_external;
};

//#endif 
#endif // !defined(CU_TEX_IMAGE_H)

