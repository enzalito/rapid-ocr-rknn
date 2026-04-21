# rapid-ocr-rknn

A C++ OCR library ported from [RapidOcrOnnx](https://github.com/RapidAI/RapidOcrOnnx) to run on Rockchip NPU hardware via the RKNN runtime. It implements the standard three-stage OCR pipeline — text detection (DBNet), orientation classification (AngleNet), and text recognition (CRNN) — entirely on the NPU.

Version: **0.1.0**

## Prerequisites

### Build host

- CMake ≥ 3.14
- C++14 compiler (GCC or Clang)
- `libopencv-dev` (version 4.x recommended)

### Target hardware / platform

Any Rockchip SoC with a compatible RKNN driver ([list](https://github.com/rockchip-linux/rknn-toolkit2#support-platform)).  

The `librknnrt.so` for each supported platform / arch is bundled under `vendor/rknn/`.

| Platform | Arch |
|----------|------|
| `Linux` | `aarch64` |
| `Linux` | `armhf` |
| `Linux` | `armhf-uclibc` |
| `Android` | `arm64-v8a` |
| `Android` | `armeabi-v7a` |

## Building

On the target device:

```sh
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DPLATFORM=<plat> \
    -DARCH=<arch>

cmake --build build --parallel
```

For cross-compilation, add the following flag:

```sh
    -DCMAKE_TOOLCHAIN_FILE=/path/to/aarch64-toolchain.cmake \
```

## Model conversion

The library expects **RKNN** models. They can be obtained with the help of the provided ONNX to RKNN conversion utility:

```sh
cd convert
uv run main.py --target <soc> --output <out_dir> # default configuation
```

To list all the available flags:

```sh
uv run main.py -h
```

### Quantization

INT8 quantization requires a set of representative images. By default the ICDAR 2015 dataset is used, but you can specify your own to potentially improve the accuracy of the models for your usecase.  

Specifying the input image resolution(s) can also help in that regard.

## Usage

Link `librapid_ocr_rknn.a`, `librknnrt.so`, and the OpenCV libraries into your application, then:

```cpp
#include <iostream>
#include <OcrLite.h>

OcrLite ocr;
ocr.initModels("det.rknn", "cls.rknn", "rec.rknn", "ppocr_keys_v1.txt");

// from file 
OcrResult file_res = ocr.detect(
    "/images/",  // dirPath
    "photo.jpg", // fileName
    50,          // padding
    960,         // maxSideLen
    0.5f,        // boxScoreThresh
    0.3f,        // boxThresh
    2.0f,        // unclipRatio
    true,        // doAngle
    true,        // mostAngle
);
std::cout << file_res.strRes << std::endl;

// or from a cv::Mat
cv::Mat img = cv::imread("photo.jpg");
OcrResult cv_res = ocr.detect(img, 50, 960, 0.5f, 0.3f, 2.0f, true, true);
std::cout << cv_res.strRes << std::endl;
```

## Runtime notes

- **Input format:** The C++ code converts each OpenCV BGR image to RGB before passing it to the NPU. Normalisation is applied inside the RKNN graph (baked in during conversion).
- **Quantisation:** All three models run as INT8. Outputs are manually dequantized on the CPU using `cv::Mat::convertTo` with the `zp`/`scale` values queried from the RKNN runtime.
- **Dynamic shapes:** The detection and recognition models use `rknn_set_input_shapes()` before each inference call. This is supported from RKNN runtime 1.5+ and requires the model to have been exported with dynamic shape support.
- **Multi-core NPU (RK3588):** The RKNN context defaults to `RKNN_NPU_CORE_AUTO`. To pin to a specific core or split across cores, call `rknn_set_core_mask()` on the underlying context after model initialisation.
- **`librknnrt.so` deployment:** The shared library must be present on the target device at runtime. Either install it system-wide or set `LD_LIBRARY_PATH` to its directory. You can find it in the `vendor` directory.

## License

Apache 2.0 — see [LICENSE](LICENSE).
