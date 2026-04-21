#ifndef __OCR_DBNET_H__
#define __OCR_DBNET_H__

#include "OcrStruct.h"
#include "rknn_api.h"
#include <opencv2/opencv.hpp>

class DbNet {
public:
    ~DbNet();

    void initModel(const std::string &pathStr);

    std::vector<TextBox> getTextBoxes(cv::Mat &src, ScaleParam &s, float boxScoreThresh,
                                      float boxThresh, float unClipRatio);

private:
    rknn_context ctx = 0;
    rknn_tensor_attr inputAttr{};
    rknn_tensor_attr outputAttr{};
    uint32_t numInputs = 0;
    uint32_t numOutputs = 0;
};


#endif //__OCR_DBNET_H__
