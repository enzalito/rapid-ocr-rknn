from dataclasses import field
from enum import StrEnum
import logging
from pathlib import Path
from typing import Protocol, cast

from rapidocr import RapidOCR
from rapidocr.inference_engine.base import dataclass
from rapidocr.inference_engine.onnxruntime import OrtInferSession
from rknn.api import RKNN

from datasets import DatasetsIndexes


logger = logging.getLogger(__name__)


class OCRVersion(StrEnum):
    V2 = "PP-OCRv2"
    V3 = "PP-OCRv3"
    V4 = "PP-OCRv4"
    V5 = "PP-OCRv5"


class Models(Protocol):
    @property
    def det_path(self) -> Path: ...

    @property
    def cls_path(self) -> Path: ...

    @property
    def rec_path(self) -> Path: ...

    @property
    def ocr_version(self) -> OCRVersion: ...


@dataclass
class LocalModels:
    det_path: Path
    cls_path: Path
    rec_path: Path
    ocr_version: OCRVersion


class LibModels:
    def __init__(self, rapid_ocr: RapidOCR) -> None:
        self.rapid_ocr = rapid_ocr

    def _model_path(self, attr: str) -> Path:
        model_path = cast(
            OrtInferSession, getattr(self.rapid_ocr, attr).session
        ).session._model_path
        if not model_path:
            raise ValueError(f"Could not find RapidOCR {attr} model path.")
        return Path(model_path)

    @property
    def det_path(self) -> Path:
        return self._model_path("text_det")

    @property
    def cls_path(self) -> Path:
        return self._model_path("text_cls")

    @property
    def rec_path(self) -> Path:
        return self._model_path("text_rec")

    @property
    def ocr_version(self) -> OCRVersion:
        return OCRVersion(self.rapid_ocr.cfg.Det["ocr_version"].value)


@dataclass
class OnnxToRknnConfig:
    """
    :param input: Source ONNX models (det, cls, rec paths and OCR version).
    :param out_path: Directory where the converted .rknn files will be written.
    :param target: RKNN target platform (e.g. ``"rk3588"``).
    :param datasets: Calibration dataset index files used for int8 quantization.
    :param det_resolutions: Sample resolutions used for det. model quatization.
    """

    DEFAULT_DET_RESOLUTIONS = [
        # 9:16
        (720, 1280),  # HD
        (1080, 1920),  # Full HD
        (1440, 2560),  # QHD
        # 4:3
        (768, 1024),  # XGA
        (1050, 1400),  # SXGA+
        (1536, 2048),  # QXGA
    ]

    input: Models
    out_path: Path
    target: str
    datasets: DatasetsIndexes
    det_resolutions: list[tuple[int, int]] = field(
        default_factory=lambda: OnnxToRknnConfig.DEFAULT_DET_RESOLUTIONS
    )


def onnx_to_rknn(cfg: OnnxToRknnConfig) -> LocalModels:
    """
    Convert ONNX models to RKNN with int8 quantization.

    :param config: Conversion config.
    :returns: Resulting RKNN models.
    """
    cfg.out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Converting det. model...")
    out_det = _det_onnx_to_rknn(cfg)
    logger.info("Det. model conversion complete.")

    logger.info("Converting cls. model...")
    out_cls = _cls_onnx_to_rknn(cfg)
    logger.info("Cls. model conversion complete.")

    logger.info("Converting rec. model...")
    out_rec = _rec_onnx_to_rknn(cfg)
    logger.info("Rec. model conversion complete.")

    return LocalModels(out_det, out_cls, out_rec, cfg.input.ocr_version)


def _make_out_model_path(in_path: Path, out_path: Path):
    return out_path / (in_path.name.removesuffix(".onnx") + ".rknn")


def _align_input_shape(res: tuple[int, int]) -> tuple[int, int]:
    height = max((res[0] // 32) * 32, 32)
    width = max((res[1] // 32) * 32, 32)
    return height, width


def _res_to_input_shapes(
    resolutions: list[tuple[int, int]],
) -> list[list[list[int]]]:
    shapes: list[list[list[int]]] = []
    for res in resolutions:
        h, w = _align_input_shape(res)
        shapes.append([[1, 3, h, w]])

    return shapes


def _det_onnx_to_rknn(cfg: OnnxToRknnConfig) -> Path:
    out = _make_out_model_path(cfg.input.det_path, cfg.out_path)
    dynamic_input = _res_to_input_shapes(cfg.det_resolutions)

    det_kit = RKNN(verbose=False)
    det_kit.config(
        mean_values=[[123.675, 116.28, 103.53]],  # ImageNet mean (RGB order)
        std_values=[[58.395, 57.12, 57.375]],  # ImageNet std  (RGB order)
        dynamic_input=dynamic_input,
        target_platform=cfg.target,
        quant_img_RGB2BGR=False,
    )
    det_kit.load_onnx(
        model=str(cfg.input.det_path),
    )

    det_kit.build(
        do_quantization=True,
        dataset=str(cfg.datasets.det_path),
        rknn_batch_size=1,
    )
    det_kit.export_rknn(str(out))

    det_kit.release()

    return out


def _cls_onnx_to_rknn(cfg: OnnxToRknnConfig) -> Path:
    out = _make_out_model_path(cfg.input.cls_path, cfg.out_path)
    input_height = 32 if cfg.input.ocr_version == OCRVersion.V2 else 48

    cls_kit = RKNN(verbose=False)
    cls_kit.config(
        mean_values=[[127.5, 127.5, 127.5]],
        std_values=[[127.5, 127.5, 127.5]],
        target_platform=cfg.target,
        quant_img_RGB2BGR=False,
    )
    cls_kit.load_onnx(
        model=str(cfg.input.cls_path),
        inputs=["x"],
        input_size_list=[[1, 3, input_height, 192]],
    )

    cls_kit.build(do_quantization=True, dataset=str(cfg.datasets.cls_path))
    cls_kit.export_rknn(str(out))

    cls_kit.release()

    return out


def _rec_onnx_to_rknn(cfg: OnnxToRknnConfig) -> Path:
    out = _make_out_model_path(cfg.input.rec_path, cfg.out_path)

    rec_kit = RKNN(verbose=False)
    rec_kit.config(
        mean_values=[[127.5, 127.5, 127.5]],
        std_values=[[127.5, 127.5, 127.5]],
        dynamic_input=[
            [[1, 3, 48, 1280]],
            [[1, 3, 48, 640]],
            [[1, 3, 48, 320]],
            [[1, 3, 48, 160]],
        ],
        target_platform=cfg.target,
        quant_img_RGB2BGR=False,
    )
    rec_kit.load_onnx(
        model=str(cfg.input.rec_path),
    )

    rec_kit.build(do_quantization=True, dataset=str(cfg.datasets.rec_path))
    rec_kit.export_rknn(str(out))

    rec_kit.release()

    return out
