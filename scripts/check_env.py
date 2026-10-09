"""Smoke test for the pixi environment (`pixi run check`).

Checks that CUDA is visible, that the compiled extensions (detectron2, pytorch3d,
torch_scatter, libs/pointops2) load and run, and that the main Python
dependencies plus the spaCy model / NLTK corpus are present.
"""

from __future__ import annotations

import importlib
import sys
import traceback

OK, FAIL, WARN = "[ ok ]", "[FAIL]", "[warn]"
failures: list[str] = []


def step(name: str, fn, *, warn_only: bool = False):
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - report everything
        tag = WARN if warn_only else FAIL
        print(f"{tag} {name}: {type(exc).__name__}: {exc}")
        if not warn_only:
            failures.append(name)
            traceback.print_exc(limit=2)
        return
    print(f"{OK} {name}" + (f": {detail}" if detail else ""))


def versions():
    out = []
    for mod in ("torch", "torchvision", "detectron2", "pytorch3d", "torch_scatter",
                "transformers", "peft", "accelerate", "datasets", "spacy", "numpy", "cv2"):
        m = importlib.import_module(mod)
        out.append(f"{mod}={getattr(m, '__version__', '?')}")
    return ", ".join(out)


def cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    names = [f"{torch.cuda.get_device_name(i)} (sm_{''.join(map(str, torch.cuda.get_device_capability(i)))})"
             for i in range(torch.cuda.device_count())]
    return f"torch cuda {torch.version.cuda}, {len(names)} GPU(s): {names}"


def detectron2_ops():
    import torch
    from detectron2 import _C  # compiled ops  # noqa: F401
    from detectron2.layers import nms_rotated

    boxes = torch.tensor([[10, 10, 20, 20, 0.0], [10, 10, 20, 20, 0.0]], device="cuda")
    keep = nms_rotated(boxes, torch.tensor([0.9, 0.8], device="cuda"), 0.5)
    return f"nms_rotated kept {keep.tolist()}"


def pytorch3d_ops():
    import torch
    from pytorch3d.ops import knn_points

    pts = torch.rand(1, 64, 3, device="cuda")
    knn = knn_points(pts, pts, K=3)
    return f"knn_points dists shape {tuple(knn.dists.shape)}"


def scatter_ops():
    import torch
    from torch_scatter import scatter_mean

    src = torch.rand(8, 4, device="cuda")
    index = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device="cuda")
    return f"scatter_mean -> {tuple(scatter_mean(src, index, dim=0).shape)}"


def pointops2_ops():
    import torch
    import libs.pointops2.functions.pointops as pointops  # needs PYTHONPATH=<repo root>

    xyz = torch.rand(256, 3, device="cuda")
    new_xyz = torch.rand(32, 3, device="cuda")
    feat = torch.rand(256, 8, device="cuda")
    offset = torch.tensor([256], dtype=torch.int32, device="cuda")
    new_offset = torch.tensor([32], dtype=torch.int32, device="cuda")
    out = pointops.interpolation(xyz, new_xyz, feat, offset, new_offset)
    return f"pointops.interpolation -> {tuple(out.shape)}"


def spacy_model():
    import spacy

    nlp = spacy.load("en_core_web_sm")
    return f"en_core_web_sm {nlp.meta['version']}"


def nltk_stopwords():
    from nltk.corpus import stopwords

    return f"{len(stopwords.words('english'))} english stopwords"


def qwen3d_package():
    import qwen3d  # noqa: F401  (imports the model + dataset mappers)

    return "import qwen3d"


if __name__ == "__main__":
    print(f"python {sys.version.split()[0]} ({sys.executable})")
    step("library versions", versions)
    step("CUDA", cuda)
    step("detectron2 compiled ops", detectron2_ops)
    step("pytorch3d compiled ops", pytorch3d_ops)
    step("torch_scatter compiled ops", scatter_ops)
    step("libs/pointops2 compiled ops", pointops2_ops)
    step("spaCy model", spacy_model)
    step("NLTK stopwords (run `pixi run setup` if missing)", nltk_stopwords, warn_only=True)
    step("qwen3d package", qwen3d_package)
    if failures:
        print(f"\n{len(failures)} check(s) failed: {', '.join(failures)}")
        sys.exit(1)
    print("\nEnvironment looks good.")
