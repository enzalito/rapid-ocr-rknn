#ifndef __OCR_ANGLENET_H__
#define __OCR_ANGLENET_H__

#include "OcrStruct.h"
#include "rknn_api.h"
#include <opencv2/opencv.hpp>

class AngleNet {
public:

    ~AngleNet();

    void initModel(const std::string &pathStr);

    std::vector<Angle> getAngles(std::vector<cv::Mat> &partImgs, const char *path,
                                 const char *imgName, bool doAngle, bool mostAngle);

private:
    bool isOutputAngleImg = false;

    rknn_context ctx = 0;
    rknn_tensor_attr outputAttr{};
    uint32_t numInputs = 0;
    uint32_t numOutputs = 0;

    const int dstWidth = 192;
    const int dstHeight = 48;

    Angle getAngle(cv::Mat &src);
};


#endif //__OCR_ANGLENET_H__
