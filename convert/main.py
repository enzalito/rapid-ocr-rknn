import logging
from pathlib import Path
from typing import Literal, Optional

import kagglehub
from rapidocr.main import RapidOCR
from typed_argparse import arg, Parser, TypedArgs

from datasets import generate_datasets, DatasetsGenerationConfig
from convert import (
    onnx_to_rknn,
    LocalModels,
    OCRVersion,
    OnnxToRknnConfig,
    LibModels,
)

logging.basicConfig(level=logging.INFO)

Targets = Literal[
    "rv1103",
    "rv1103b",
    "rv1106",
    "rv1106b",
    "rv1126b",
    "rk2118",
    "rk3562",
    "rk3566",
    "rk3568",
    "rk3576",
    "rk3588",
]


def resolutions_to_str(l: list[tuple[int, int]]) -> str:  # noqa: E741
    return ",".join(f"{w}x{h}" for w, h in l)


class Args(TypedArgs):
    target: Targets = arg(
        "-t", "--target", help="Target architecture of the output model."
    )
    output: Path = arg(
        "-o", "--output", help="Output directory for the converted models."
    )
    det_model: Optional[Path] = arg(
        help="Detection model path. Uses the RapidOCR library default if omitted."
    )
    cls_model: Optional[Path] = arg(
        help="Classification model path. Uses the RapidOCR library default if omitted."
    )
    rec_model: Optional[Path] = arg(
        help="Recognition model path. Uses the RapidOCR library default if omitted."
    )
    models_version: Optional[OCRVersion] = arg(help="Version of the passed models.")
    quant_dataset: Optional[Path] = arg(
        help="""
        Directory containing full-scene text images for quantization.
        ICAR 2015 will be downloaded and used if omitted.
        """
    )
    quant_imgs_glob: str = arg(
        help="""
        Glob pattern to match images in the dataset directory.
        Do not set if `--quant-dataset` is ommited.
        """,
        default=DatasetsGenerationConfig.DEFAULT_IMG_GLOB,
    )
    quant_max_imgs: int = arg(
        help="Maximum number of images to use for quantization.",
        default=DatasetsGenerationConfig.DEFAULT_MAX_IMGS,
    )
    quant_det_resolutions: str = arg(
        help="Sample resolutions used for det. model quatization.",
        default=resolutions_to_str(OnnxToRknnConfig.DEFAULT_DET_RESOLUTIONS),
    )


def parse_datasets_gen_config(args: Args) -> DatasetsGenerationConfig:
    out_dir = Path(".cache/rapid-ocr-rknn/datasets")

    in_dir = args.quant_dataset
    if in_dir is None:
        imgs_path = kagglehub.dataset_download(
            "hafizshehbazali/icdar2015",
        )
        in_dir = Path(imgs_path) / "ch4_test_images"

    return DatasetsGenerationConfig(
        in_dir,
        out_dir,
        args.quant_imgs_glob,
        args.quant_max_imgs,
    )


def parse_resolutions(s: str) -> list[tuple[int, int]]:
    l: list[tuple[int, int]] = []  # noqa: E741
    for sub_str in s.split(","):
        h, w = sub_str.split("x")
        l.append((int(h), int(w)))

    return l


def parse_models(args: Args) -> Optional[LocalModels]:
    match (args.det_model, args.cls_model, args.rec_model, args.models_version):
        case (None, None, None, None):
            return None
        case (Path() as d, Path() as c, Path() as r, OCRVersion() as v):
            return LocalModels(d, c, r, v)
        case _:
            raise Exception(
                "invalid flag combination: models and version must be exhaustive or omitted."
            )


if __name__ == "__main__":
    parser = Parser(
        Args,
        description="A utility to convert RapidOCR ONNX models to the RKNN format.",
    )

    def main(args: Args):
        try:
            datasets_gen_cfg = parse_datasets_gen_config(args)
            det_resolutions = parse_resolutions(args.quant_det_resolutions)
            models = parse_models(args)
        except Exception as e:
            parser._argparse_parser.error(str(e))

        ocr = RapidOCR()

        datasets = generate_datasets(ocr, datasets_gen_cfg)

        convert_cfg = OnnxToRknnConfig(
            models or LibModels(ocr),
            args.output,
            args.target,
            datasets,
            det_resolutions,
        )
        onnx_to_rknn(convert_cfg)

    parser.bind(main).run()
