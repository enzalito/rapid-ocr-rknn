#include "CrnnNet.h"
#include "OcrUtils.h"
#include <fstream>

CrnnNet::~CrnnNet() {
    if (ctx != 0) {
        rknn_destroy(ctx);
        ctx = 0;
    }
}

void CrnnNet::initModel(const std::string &pathStr, const std::string &keysPath) {
    std::vector<unsigned char> modelData = loadModelFile(pathStr);
    if (modelData.empty()) {
        printf("CrnnNet: failed to load model from %s\n", pathStr.c_str());
        return;
    }

    int ret = rknn_init(&ctx, modelData.data(), modelData.size(), 0, nullptr);
    if (ret != RKNN_SUCC) {
        printf("CrnnNet: rknn_init failed, ret=%d\n", ret);
        return;
    }

    // Query input/output counts
    rknn_input_output_num ioNum{};
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &ioNum, sizeof(ioNum));
    if (ret != RKNN_SUCC) {
        printf("CrnnNet: rknn_query IN_OUT_NUM failed, ret=%d\n", ret);
        return;
    }
    numInputs = ioNum.n_input;
    numOutputs = ioNum.n_output;

    // Query output attributes for dequantization parameters (zp, scale)
    outputAttr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &outputAttr, sizeof(outputAttr));
    if (ret != RKNN_SUCC) {
        printf("CrnnNet: rknn_query OUTPUT_ATTR failed, ret=%d\n", ret);
        return;
    }

    printf("CrnnNet: loaded model %s  inputs=%u outputs=%u\n", pathStr.c_str(), numInputs, numOutputs);
    printf("CrnnNet: output qnt_type=%d zp=%d scale=%f\n", outputAttr.qnt_type, outputAttr.zp, outputAttr.scale);

    // Load character dictionary
    std::ifstream in(keysPath.c_str());
    std::string line;
    if (in) {
        while (getline(in, line)) {
            keys.push_back(line);
        }
    } else {
        printf("CrnnNet: keys.txt file not found: %s\n", keysPath.c_str());
        return;
    }
    keys.insert(keys.begin(), "#");
    keys.emplace_back(" ");
    printf("CrnnNet: total keys size(%lu)\n", keys.size());
}

template<class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}

TextLine CrnnNet::scoreToTextLine(const std::vector<float> &outputData, size_t h, size_t w) {
    auto keySize = keys.size();
    auto dataSize = outputData.size();
    std::string strRes;
    std::vector<float> scores;
    size_t lastIndex = 0;
    size_t maxIndex;
    float maxValue;

    for (size_t i = 0; i < h; i++) {
        size_t start = i * w;
        size_t stop = (i + 1) * w;
        if (stop > dataSize - 1) {
            stop = (i + 1) * w - 1;
        }
        maxIndex = int(argmax(&outputData[start], &outputData[stop]));
        maxValue = float(*std::max_element(&outputData[start], &outputData[stop]));

        if (maxIndex > 0 && maxIndex < keySize && (!(i > 0 && maxIndex == lastIndex))) {
            scores.emplace_back(maxValue);
            strRes.append(keys[maxIndex]);
        }
        lastIndex = maxIndex;
    }
    return {strRes, scores};
}

TextLine CrnnNet::getTextLine(const cv::Mat &src) {
    // Scale to fixed height=48, proportional width
    float scale = static_cast<float>(dstHeight) / static_cast<float>(src.rows);
    int dstWidth = static_cast<int>(static_cast<float>(src.cols) * scale);

    cv::Mat srcResize;
    cv::resize(src, srcResize, cv::Size(dstWidth, dstHeight));

    // BGR -> RGB; normalization is baked into the model
    cv::Mat srcRgb;
    cv::cvtColor(srcResize, srcRgb, cv::COLOR_BGR2RGB);

    // Set dynamic input shape (NHWC: 1 x dstHeight x dstWidth x 3)
    rknn_tensor_attr dynInputAttr{};
    dynInputAttr.index = 0;
    dynInputAttr.n_dims = 4;
    dynInputAttr.dims[0] = 1;
    dynInputAttr.dims[1] = static_cast<uint32_t>(dstHeight);
    dynInputAttr.dims[2] = static_cast<uint32_t>(dstWidth);
    dynInputAttr.dims[3] = 3;
    dynInputAttr.fmt = RKNN_TENSOR_NHWC;
    dynInputAttr.type = RKNN_TENSOR_UINT8;

    int ret = rknn_set_input_shapes(ctx, 1, &dynInputAttr);
    if (ret != RKNN_SUCC) {
        printf("CrnnNet: rknn_set_input_shapes failed, ret=%d\n", ret);
        return {"", {}};
    }

    rknn_input input{};
    input.index = 0;
    input.buf = srcRgb.data;
    input.size = static_cast<uint32_t>(srcRgb.total() * srcRgb.elemSize());
    input.pass_through = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NHWC;

    ret = rknn_inputs_set(ctx, 1, &input);
    if (ret != RKNN_SUCC) {
        printf("CrnnNet: rknn_inputs_set failed, ret=%d\n", ret);
        return {"", {}};
    }

    ret = rknn_run(ctx, nullptr);
    if (ret != RKNN_SUCC) {
        printf("CrnnNet: rknn_run failed, ret=%d\n", ret);
        return {"", {}};
    }

    // Query current output shape (sequence_length x num_classes varies with input width)
    rknn_tensor_attr curOutputAttr{};
    curOutputAttr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_CURRENT_OUTPUT_ATTR, &curOutputAttr, sizeof(curOutputAttr));
    if (ret != RKNN_SUCC) {
        printf("CrnnNet: rknn_query CURRENT_OUTPUT_ATTR failed, ret=%d\n", ret);
        return {"", {}};
    }

    // Get raw INT8 output
    rknn_output output{};
    output.want_float = 0;
    output.is_prealloc = 0;
    output.index = 0;

    ret = rknn_outputs_get(ctx, 1, &output, nullptr);
    if (ret != RKNN_SUCC) {
        printf("CrnnNet: rknn_outputs_get failed, ret=%d\n", ret);
        return {"", {}};
    }

    // CRNN output shape is typically [1, seq_len, num_classes] (3-dim)
    // or [seq_len, 1, num_classes] — extract the two meaningful dims.
    size_t seqLen = 1;
    size_t numClasses = 1;
    if (curOutputAttr.n_dims >= 3) {
        // Assume last dim is num_classes, second-to-last is seq_len
        seqLen    = curOutputAttr.dims[curOutputAttr.n_dims - 2];
        numClasses = curOutputAttr.dims[curOutputAttr.n_dims - 1];
    } else if (curOutputAttr.n_dims == 2) {
        seqLen    = curOutputAttr.dims[0];
        numClasses = curOutputAttr.dims[1];
    }

    // Dequantize: treat as [seqLen, numClasses] 2D layout
    cv::Mat floatMat = dequantizeOutput(
        static_cast<const int8_t *>(output.buf),
        static_cast<int>(seqLen),
        static_cast<int>(numClasses),
        curOutputAttr.zp, curOutputAttr.scale);

    rknn_outputs_release(ctx, 1, &output);

    std::vector<float> outputData(floatMat.ptr<float>(0),
                                  floatMat.ptr<float>(0) + seqLen * numClasses);
    return scoreToTextLine(outputData, seqLen, numClasses);
}

std::vector<TextLine> CrnnNet::getTextLines(std::vector<cv::Mat> &partImg, const char *path, const char *imgName) {
    int size = static_cast<int>(partImg.size());
    std::vector<TextLine> textLines(size);
    for (int i = 0; i < size; ++i) {
        if (isOutputDebugImg) {
            std::string debugImgFile = getDebugImgFilePath(path, imgName, i, "-debug-");
            saveImg(partImg[i], debugImgFile.c_str());
        }

        double startCrnnTime = getCurrentTime();
        TextLine textLine = getTextLine(partImg[i]);
        double endCrnnTime = getCurrentTime();
        textLine.time = endCrnnTime - startCrnnTime;
        textLines[i] = textLine;
    }
    return textLines;
}
