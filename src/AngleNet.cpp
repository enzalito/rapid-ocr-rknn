#include "AngleNet.h"
#include "OcrUtils.h"
#include <numeric>

AngleNet::~AngleNet() {
    if (ctx != 0) {
        rknn_destroy(ctx);
        ctx = 0;
    }
}

void AngleNet::initModel(const std::string &pathStr) {
    std::vector<unsigned char> modelData = loadModelFile(pathStr);
    if (modelData.empty()) {
        printf("AngleNet: failed to load model from %s\n", pathStr.c_str());
        return;
    }

    int ret = rknn_init(&ctx, modelData.data(), modelData.size(), 0, nullptr);
    if (ret != RKNN_SUCC) {
        printf("AngleNet: rknn_init failed, ret=%d\n", ret);
        return;
    }

    // Query input/output counts
    rknn_input_output_num ioNum{};
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &ioNum, sizeof(ioNum));
    if (ret != RKNN_SUCC) {
        printf("AngleNet: rknn_query IN_OUT_NUM failed, ret=%d\n", ret);
        return;
    }
    numInputs = ioNum.n_input;
    numOutputs = ioNum.n_output;

    // Query output attributes for dequantization parameters (zp, scale)
    outputAttr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &outputAttr, sizeof(outputAttr));
    if (ret != RKNN_SUCC) {
        printf("AngleNet: rknn_query OUTPUT_ATTR failed, ret=%d\n", ret);
        return;
    }

    printf("AngleNet: loaded model %s  inputs=%u outputs=%u\n", pathStr.c_str(), numInputs, numOutputs);
    printf("AngleNet: output qnt_type=%d zp=%d scale=%f\n", outputAttr.qnt_type, outputAttr.zp, outputAttr.scale);
}

static Angle scoreToAngle(const std::vector<float> &outputData) {
    int maxIndex = 0;
    float maxScore = 0;
    for (size_t i = 0; i < outputData.size(); i++) {
        if (outputData[i] > maxScore) {
            maxScore = outputData[i];
            maxIndex = static_cast<int>(i);
        }
    }
    return {maxIndex, maxScore};
}

Angle AngleNet::getAngle(cv::Mat &src) {
    // Input is already resized to dstWidth x dstHeight (192x48) by the caller.
    // BGR -> RGB; normalization baked into the model.
    cv::Mat srcRgb;
    cv::cvtColor(src, srcRgb, cv::COLOR_BGR2RGB);

    // AngleNet has a fixed input shape — no dynamic shape call needed.
    rknn_input input{};
    input.index = 0;
    input.buf = srcRgb.data;
    input.size = static_cast<uint32_t>(srcRgb.total() * srcRgb.elemSize());
    input.pass_through = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NHWC;

    int ret = rknn_inputs_set(ctx, 1, &input);
    if (ret != RKNN_SUCC) {
        printf("AngleNet: rknn_inputs_set failed, ret=%d\n", ret);
        return {-1, 0.f};
    }

    ret = rknn_run(ctx, nullptr);
    if (ret != RKNN_SUCC) {
        printf("AngleNet: rknn_run failed, ret=%d\n", ret);
        return {-1, 0.f};
    }

    // Raw INT8 output
    rknn_output output{};
    output.want_float = 0;
    output.is_prealloc = 0;
    output.index = 0;

    ret = rknn_outputs_get(ctx, 1, &output, nullptr);
    if (ret != RKNN_SUCC) {
        printf("AngleNet: rknn_outputs_get failed, ret=%d\n", ret);
        return {-1, 0.f};
    }

    // Dequantize: output is [1, 2] (two angle classes)
    // Total elements = output.size / sizeof(int8_t)
    int numElems = static_cast<int>(output.size / sizeof(int8_t));
    cv::Mat floatMat = dequantizeOutput(
        static_cast<const int8_t *>(output.buf),
        1, numElems,
        outputAttr.zp, outputAttr.scale);

    rknn_outputs_release(ctx, 1, &output);

    std::vector<float> outputData(floatMat.ptr<float>(0),
                                  floatMat.ptr<float>(0) + numElems);
    return scoreToAngle(outputData);
}

std::vector<Angle> AngleNet::getAngles(std::vector<cv::Mat> &partImgs, const char *path,
                                       const char *imgName, bool doAngle, bool mostAngle) {
    size_t size = partImgs.size();
    std::vector<Angle> angles(size);
    if (doAngle) {
        for (size_t i = 0; i < size; ++i) {
            double startAngle = getCurrentTime();
            cv::Mat angleImg;
            cv::resize(partImgs[i], angleImg, cv::Size(dstWidth, dstHeight));
            Angle angle = getAngle(angleImg);
            double endAngle = getCurrentTime();
            angle.time = endAngle - startAngle;

            angles[i] = angle;

            // Output AngleImg if enabled
            if (isOutputAngleImg) {
                std::string angleImgFile = getDebugImgFilePath(path, imgName, i, "-angle-");
                saveImg(angleImg, angleImgFile.c_str());
            }
        }
    } else {
        for (size_t i = 0; i < size; ++i) {
            angles[i] = Angle{-1, 0.f};
        }
    }

    // Most Possible AngleIndex (majority vote)
    if (doAngle && mostAngle) {
        auto angleIndexes = getAngleIndexes(angles);
        double sum = std::accumulate(angleIndexes.begin(), angleIndexes.end(), 0.0);
        double halfPercent = angles.size() / 2.0f;
        int mostAngleIndex;
        if (sum < halfPercent) {
            mostAngleIndex = 0;
        } else {
            mostAngleIndex = 1;
        }
        for (size_t i = 0; i < angles.size(); ++i) {
            Angle angle = angles[i];
            angle.index = mostAngleIndex;
            angles.at(i) = angle;
        }
    }

    return angles;
}
