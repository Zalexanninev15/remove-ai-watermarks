"""Rebuild synthetic silhouettes for additional AI disclosure layouts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.render_vendor_silhouettes import render  # noqa: E402

log = logging.getLogger(__name__)
ASSETS = ROOT / "src" / "remove_ai_watermarks" / "assets"
CJK = {"font": "/System/Library/Fonts/Hiragino Sans GB.ttc", "font_index": 2}


def centered_lines(
    top: str, bottom: str, *, top_scale: float, bottom_height: float = 1.0, shear: float = 0.0
) -> np.ndarray:
    upper, lower = render(top, opts=CJK), render(bottom, opts=CJK)
    if bottom_height != 1.0:
        lower = cv2.resize(lower, (upper.shape[1], round(upper.shape[0] * bottom_height)))
    else:
        upper = cv2.resize(upper, (round(lower.shape[1] * top_scale), round(lower.shape[0] * 0.95)))
    lines = [upper, lower]
    result = np.zeros((sum(line.shape[0] for line in lines) + 2, max(line.shape[1] for line in lines)), np.uint8)
    y = 0
    for line in lines:
        x = (result.shape[1] - line.shape[1]) // 2
        result[y : y + line.shape[0], x : x + line.shape[1]] = line
        y += line.shape[0] + 2
    if shear:
        h, w = result.shape
        extra = round(abs(shear) * h)
        result = cv2.warpAffine(result, np.float32([[1, shear, extra], [0, 1, 0]]), (w + extra, h))
    return result


def silhouettes() -> dict[str, np.ndarray]:
    return {
        "workbuddy_alpha.png": centered_lines("AI生成", "WORKBUDDY>_", top_scale=1.0, bottom_height=0.45),
        "yuanbao_compact_alpha.png": centered_lines("元宝", "AI生成", top_scale=0.65, shear=-0.15),
        "yuanbao_compact_gray_alpha.png": centered_lines("元宝", "AI生成", top_scale=0.65, shear=-0.15),
        "doubao_outline_alpha.png": render("豆包AI生成", opts=CJK),
        "samsung_ko_alpha.png": render(
            "AI로 생성한 콘텐츠", opts={"font": "/System/Library/Fonts/AppleSDGothicNeo.ttc"}
        ),
        "microsoft_text_alpha.png": render(
            "Made with AI", opts={"font": "/System/Library/Fonts/Supplemental/Arial.ttf"}
        ),
        "jimeng_label_alpha.png": render("AI生成", opts={"font": "/System/Library/Fonts/STHeiti Medium.ttc"}),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    for name, alpha in silhouettes().items():
        Image.fromarray(alpha).save(ASSETS / name)
        log.info("Rendered %s, %s x %s", name, alpha.shape[1], alpha.shape[0])


if __name__ == "__main__":
    main()
