from dataclasses import dataclass
import logging
import shutil
from typing import Optional
from pathlib import Path

import cv2
from cv2.typing import MatLike
import numpy as np
from rapidocr import RapidOCR

logger = logging.getLogger(__name__)


@dataclass
class DatasetsIndexes:
    det_path: Path
    cls_path: Path
    rec_path: Path


@dataclass
class DatasetsGenerationConfig:
    """
    :param in_path: Path of the directory that contains the input full-scene text images.
    :param out_path: Path of the output directory for the generated datasets.
    :param img_glob: Glob pattern used to find images in the input directory.
    :param max_imgs: Maximum images in each generated dataset.
    """

    DEFAULT_IMG_GLOB = "*.jpg"
    DEFAULT_MAX_IMGS = 200

    in_path: Path
    out_path: Path
    img_glob: str = DEFAULT_IMG_GLOB
    max_imgs: int = DEFAULT_MAX_IMGS


def generate_datasets(ocr: RapidOCR, cfg: DatasetsGenerationConfig) -> DatasetsIndexes:
    """
    Generate the datasets needed for quantization of the models.

    :param ocr: RapidOCR instance used generate the cls and det datasets.
    :param cfg: Generation config.
    :returns: Index file paths for the generated datasets.
    """

    logger.info("Starting quantization datasets generation...")

    det_out = _mkdir_clean(cfg.out_path / "det")
    cls_out = _mkdir_clean(cfg.out_path / "cls")
    rec_out = _mkdir_clean(cfg.out_path / "rec")

    det_img_paths: list[Path] = []
    cls_img_paths: list[Path] = []
    rec_img_paths: list[Path] = []

    for img_path in sorted(cfg.in_path.glob(cfg.img_glob)):
        if (
            len(det_img_paths) >= cfg.max_imgs
            and len(cls_img_paths) >= cfg.max_imgs
            and len(rec_img_paths) >= cfg.max_imgs
        ):
            break

        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Failed to load {img_path}, skipped.")
            continue

        if len(det_img_paths) < cfg.max_imgs:
            det_img_paths.append(_write_img(img, det_out, len(det_img_paths)))

        # if len(rec_img_names) >= cfg.max_imgs the same is true for cls_img_names
        if len(rec_img_paths) >= cfg.max_imgs:
            continue

        det_res = ocr.text_det(img)
        if det_res.boxes is None:
            continue

        raw_crops = []
        for box in det_res.boxes:
            crop = _perspective_crop(img, box)
            if crop is None or crop.size <= 0:
                continue

            raw_crops.append(crop)

            if len(cls_img_paths) < cfg.max_imgs:
                cls_img_paths.append(_write_img(img, cls_out, len(cls_img_paths)))

        cls_res = ocr.text_cls(raw_crops)
        if cls_res.img_list is None:
            continue

        for crop in cls_res.img_list:
            if crop is None or crop.size <= 0:
                continue

            rec_img_paths.append(_write_img(img, rec_out, len(rec_img_paths)))

            if len(rec_img_paths) >= cfg.max_imgs:
                break

    det_idx = _write_index(det_out, det_img_paths)
    logger.info(f"{str(det_out)}: {len(det_img_paths)} images.")
    cls_idx = _write_index(cls_out, cls_img_paths)
    logger.info(f"{str(cls_out)}: {len(cls_img_paths)} images.")
    rec_idx = _write_index(rec_out, rec_img_paths)
    logger.info(f"{str(rec_out)}: {len(rec_img_paths)} images.")

    logger.info("Generation complete.")

    return DatasetsIndexes(det_idx, cls_idx, rec_idx)


def _mkdir_clean(dir: Path):
    if dir.exists():
        shutil.rmtree(dir)
    dir.mkdir(parents=True, exist_ok=True)

    return dir


def _write_img(img: MatLike, out: Path, idx: int) -> Path:
    gen_img_path = out / f"{idx + 1:04d}.jpg"
    cv2.imwrite(str(gen_img_path), img)
    return gen_img_path


def _write_index(dir: Path, imgs: list[Path]) -> Path:
    """
    :param dir: Path of the directory where the index will be generated.
    :param imgs: Paths of the images to reference in the index.
    :returns: Path of the newly created index file.
    """

    dest = dir / "index.txt"
    content = ""
    for img in imgs:
        content += str(img.resolve()) + "\n"

    dest.write_text(content)

    return dest


def _perspective_crop(img: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
    """
    :param img: Base image.
    :param box: Coordinates of the bounding box: shape (4, 2), TL→TR→BR→BL, float32.
    """

    tl, tr, br, bl = box.astype("float32")
    width = max(int(np.linalg.norm(tr - tl)), int(np.linalg.norm(br - bl)))
    height = max(int(np.linalg.norm(bl - tl)), int(np.linalg.norm(br - tr)))
    if width <= 0 or height <= 0:
        return None

    out_coords = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    trans_mat = cv2.getPerspectiveTransform(box.astype("float32"), out_coords)
    corrected = cv2.warpPerspective(
        img,
        trans_mat,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if corrected.shape[0] >= corrected.shape[1] * 1.5:
        corrected = cv2.rotate(corrected, cv2.ROTATE_90_CLOCKWISE)

    return corrected
