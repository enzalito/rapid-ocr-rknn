#include "DbNet.h"
#include "OcrUtils.h"

DbNet::~DbNet() {
    if (ctx != 0) {
        rknn_destroy(ctx);
        ctx = 0;
    }
}

void DbNet::initModel(const std::string &pathStr) {
    std::vector<unsigned char> modelData = loadModelFile(pathStr);
    if (modelData.empty()) {
        printf("DbNet: failed to load model from %s\n", pathStr.c_str());
        return;
    }

    int ret = rknn_init(&ctx, modelData.data(), modelData.size(), 0, nullptr);
    if (ret != RKNN_SUCC) {
        printf("DbNet: rknn_init failed, ret=%d\n", ret);
        return;
    }

    // Query input/output counts
    rknn_input_output_num ioNum{};
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &ioNum, sizeof(ioNum));
    if (ret != RKNN_SUCC) {
        printf("DbNet: rknn_query IN_OUT_NUM failed, ret=%d\n", ret);
        return;
    }
    numInputs = ioNum.n_input;
    numOutputs = ioNum.n_output;

    // Query base input attributes (used for reference; actual shape is set dynamically)
    inputAttr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &inputAttr, sizeof(inputAttr));
    if (ret != RKNN_SUCC) {
        printf("DbNet: rknn_query INPUT_ATTR failed, ret=%d\n", ret);
        return;
    }

    // Query base output attributes (zp/scale used for dequantization)
    outputAttr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &outputAttr, sizeof(outputAttr));
    if (ret != RKNN_SUCC) {
        printf("DbNet: rknn_query OUTPUT_ATTR failed, ret=%d\n", ret);
        return;
    }

    printf("DbNet: loaded model %s  inputs=%u outputs=%u\n", pathStr.c_str(), numInputs, numOutputs);
    printf("DbNet: output qnt_type=%d zp=%d scale=%f\n", outputAttr.qnt_type, outputAttr.zp, outputAttr.scale);
}

std::vector<TextBox> findRsBoxes(const cv::Mat &predMat, const cv::Mat &dilateMat, ScaleParam &s,
                                 const float boxScoreThresh, const float unClipRatio) {
    const int longSideThresh = 3;
    const int maxCandidates = 1000;

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(dilateMat, contours, hierarchy, cv::RETR_LIST,
                     cv::CHAIN_APPROX_SIMPLE);

    size_t numContours = contours.size() >= maxCandidates ? maxCandidates : contours.size();

    std::vector<TextBox> rsBoxes;

    for (size_t i = 0; i < numContours; i++) {
        if (contours[i].size() <= 2) {
            continue;
        }
        cv::RotatedRect minAreaRect = cv::minAreaRect(contours[i]);

        float longSide;
        std::vector<cv::Point2f> minBoxes = getMinBoxes(minAreaRect, longSide);

        if (longSide < longSideThresh) {
            continue;
        }

        float boxScore = boxScoreFast(minBoxes, predMat);
        if (boxScore < boxScoreThresh)
            continue;

        //-----unClip-----
        cv::RotatedRect clipRect = unClip(minBoxes, unClipRatio);
        if (clipRect.size.height < 1.001 && clipRect.size.width < 1.001) {
            continue;
        }
        //-----unClip-----

        std::vector<cv::Point2f> clipMinBoxes = getMinBoxes(clipRect, longSide);
        if (longSide < longSideThresh + 2)
            continue;

        std::vector<cv::Point> intClipMinBoxes;

        for (auto &clipMinBox: clipMinBoxes) {
            float x = clipMinBox.x / s.ratioWidth;
            float y = clipMinBox.y / s.ratioHeight;
            int ptX = (std::min)((std::max)(int(x), 0), s.srcWidth - 1);
            int ptY = (std::min)((std::max)(int(y), 0), s.srcHeight - 1);
            cv::Point point{ptX, ptY};
            intClipMinBoxes.push_back(point);
        }
        rsBoxes.push_back(TextBox{intClipMinBoxes, boxScore});
    }
    reverse(rsBoxes.begin(), rsBoxes.end());
    return rsBoxes;
}

// ---- Inference ----

std::vector<TextBox>
DbNet::getTextBoxes(cv::Mat &src, ScaleParam &s, float boxScoreThresh, float boxThresh, float unClipRatio) {
    // 1. Resize to target detection dimensions
    cv::Mat srcResize;
    cv::resize(src, srcResize, cv::Size(s.dstWidth, s.dstHeight));

    // 2. BGR -> RGB (model expects RGB; normalization is baked in)
    cv::Mat srcRgb;
    cv::cvtColor(srcResize, srcRgb, cv::COLOR_BGR2RGB);

    // 3. Set dynamic input shape (NHWC: 1 x H x W x 3)
    rknn_tensor_attr dynInputAttr{};
    dynInputAttr.index = 0;
    dynInputAttr.n_dims = 4;
    dynInputAttr.dims[0] = 1;
    dynInputAttr.dims[1] = static_cast<uint32_t>(s.dstHeight);
    dynInputAttr.dims[2] = static_cast<uint32_t>(s.dstWidth);
    dynInputAttr.dims[3] = 3;
    dynInputAttr.fmt = RKNN_TENSOR_NHWC;
    dynInputAttr.type = RKNN_TENSOR_UINT8;

    int ret = rknn_set_input_shapes(ctx, 1, &dynInputAttr);
    if (ret != RKNN_SUCC) {
        printf("DbNet: rknn_set_input_shapes failed, ret=%d\n", ret);
        return {};
    }

    // 4. Set input data
    rknn_input input{};
    input.index = 0;
    input.buf = srcRgb.data;
    input.size = static_cast<uint32_t>(srcRgb.total() * srcRgb.elemSize());
    input.pass_through = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NHWC;

    ret = rknn_inputs_set(ctx, 1, &input);
    if (ret != RKNN_SUCC) {
        printf("DbNet: rknn_inputs_set failed, ret=%d\n", ret);
        return {};
    }

    // 5. Run inference
    ret = rknn_run(ctx, nullptr);
    if (ret != RKNN_SUCC) {
        printf("DbNet: rknn_run failed, ret=%d\n", ret);
        return {};
    }

    // 6. Query current output shape (dynamic model changes output dims per input shape)
    rknn_tensor_attr curOutputAttr{};
    curOutputAttr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_CURRENT_OUTPUT_ATTR, &curOutputAttr, sizeof(curOutputAttr));
    if (ret != RKNN_SUCC) {
        printf("DbNet: rknn_query CURRENT_OUTPUT_ATTR failed, ret=%d\n", ret);
        return {};
    }

    // 7. Get raw INT8 output
    rknn_output output{};
    output.want_float = 0;
    output.is_prealloc = 0;
    output.index = 0;

    ret = rknn_outputs_get(ctx, 1, &output, nullptr);
    if (ret != RKNN_SUCC) {
        printf("DbNet: rknn_outputs_get failed, ret=%d\n", ret);
        return {};
    }

    // 8. Dequantize to float
    // DBNet output shape is typically [1, 1, H, W] or [1, H, W] -- extract H and W
    int outHeight = s.dstHeight;
    int outWidth = s.dstWidth;
    if (curOutputAttr.n_dims >= 4) {
        outHeight = static_cast<int>(curOutputAttr.dims[curOutputAttr.n_dims - 2]);
        outWidth  = static_cast<int>(curOutputAttr.dims[curOutputAttr.n_dims - 1]);
    } else if (curOutputAttr.n_dims == 3) {
        outHeight = static_cast<int>(curOutputAttr.dims[1]);
        outWidth  = static_cast<int>(curOutputAttr.dims[2]);
    }

    cv::Mat predMat = dequantizeOutput(
        static_cast<const int8_t *>(output.buf),
        outHeight, outWidth,
        curOutputAttr.zp, curOutputAttr.scale);

    rknn_outputs_release(ctx, 1, &output);

    // 9. Build byte mask for thresholding (identical logic to ONNX version)
    size_t area = static_cast<size_t>(outHeight) * static_cast<size_t>(outWidth);
    std::vector<unsigned char> cbufData(area);
    const float *predPtr = predMat.ptr<float>(0);
    for (size_t i = 0; i < area; i++) {
        cbufData[i] = static_cast<unsigned char>(predPtr[i] * 255.0f);
    }
    cv::Mat cBufMat(outHeight, outWidth, CV_8UC1, cbufData.data());

    // 10. Threshold -> dilate -> find contours -> filter boxes
    const double maxValue = 255;
    const double threshold = boxThresh * 255;
    cv::Mat thresholdMat;
    cv::threshold(cBufMat, thresholdMat, threshold, maxValue, cv::THRESH_BINARY);

    cv::Mat dilateMat;
    cv::Mat dilateElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(thresholdMat, dilateMat, dilateElement);

    return findRsBoxes(predMat, dilateMat, s, boxScoreThresh, unClipRatio);
}
