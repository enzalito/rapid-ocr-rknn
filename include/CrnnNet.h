#ifndef __OCR_CRNNNET_H__
#define __OCR_CRNNNET_H__

#include "OcrStruct.h"
#include "rknn_api.h"
#include <opencv2/opencv.hpp>

class CrnnNet {
public:

    ~CrnnNet();

    void initModel(const std::string &pathStr, const std::string &keysPath);

    std::vector<TextLine> getTextLines(std::vector<cv::Mat> &partImg, const char *path, const char *imgName);

private:
    bool isOutputDebugImg = false;

    rknn_context ctx = 0;
    rknn_tensor_attr outputAttr{};
    uint32_t numInputs = 0;
    uint32_t numOutputs = 0;

    const int dstHeight = 48;

    std::vector<std::string> keys;

    TextLine scoreToTextLine(const std::vector<float> &outputData, size_t h, size_t w);

    TextLine getTextLine(const cv::Mat &src);
};


#endif //__OCR_CRNNNET_H__
