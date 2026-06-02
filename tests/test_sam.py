"""Tests for SAM2 polygon post-processing."""

import os
import subprocess
import sys
import types
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest
import shapely
import torch
from shapely import wkt
from shapely.geometry import box

from deepforest import get_data, utilities
from deepforest.main import deepforest
from deepforest.scripts.sam import (
    Sam2PolygonModel,
    convert_boxes_to_polygons,
    load_sam2_model,
    process_image_group,
)

SAM_SCRIPT = files("deepforest.scripts").joinpath("sam.py")


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeSam2Outputs:
    def __init__(self, num_objects: int):
        self.pred_masks = torch.zeros((1, num_objects, 3, 4, 4))
        self.iou_scores = torch.full((1, num_objects, 3), 0.9)


class _FakeSam2Processor:
    _cached_prompts = None
    _prompt_mode = "box"

    @classmethod
    def from_pretrained(cls, model_name, token=None):
        _ = (model_name, token)
        return cls()

    def __call__(
        self,
        images=None,
        input_boxes=None,
        input_points=None,
        input_labels=None,
        return_tensors="pt",
    ):
        _ = (input_labels, return_tensors)
        if input_boxes is not None:
            _FakeSam2Processor._prompt_mode = "box"
            _FakeSam2Processor._cached_prompts = input_boxes[0]
        elif input_points is not None:
            _FakeSam2Processor._prompt_mode = "point"
            _FakeSam2Processor._cached_prompts = [
                (point_group[0][0], point_group[0][1]) for point_group in input_points[0]
            ]
        else:
            raise ValueError("Fake processor expects input_boxes or input_points")

        return _FakeBatch(
            {
                "original_sizes": torch.tensor([[images.height, images.width]]),
            }
        )

    def post_process_masks(
        self,
        pred_masks,
        original_sizes,
        binarize=False,
        **kwargs,
    ):
        _ = (pred_masks, binarize, kwargs)
        height, width = original_sizes[0].tolist()
        mask_sets = []
        for prompt in _FakeSam2Processor._cached_prompts:
            if _FakeSam2Processor._prompt_mode == "box":
                x1, y1, x2, y2 = [int(v) for v in prompt]
            else:
                x, y = [int(v) for v in prompt]
                half = 8
                x1, y1, x2, y2 = x - half, y - half, x + half, y + half

            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height))
            if x2 <= x1:
                x2 = min(width, x1 + 1)
            if y2 <= y1:
                y2 = min(height, y1 + 1)

            mask = np.zeros((height, width), dtype=np.float32)
            mask[y1:y2, x1:x2] = 1.0
            corner_h = max(1, (y2 - y1) // 3)
            corner_w = max(1, (x2 - x1) // 3)
            mask[y1 : y1 + corner_h, x1 : x1 + corner_w] = 0.0
            mask_sets.append([mask, mask, mask])

        return [mask_sets]


class _FakeSam2Model:
    @classmethod
    def from_pretrained(cls, model_name, token=None):
        _ = (model_name, token)
        return cls()

    def to(self, device):
        _ = device
        return self

    def __call__(self, **kwargs):
        _ = kwargs
        num_objects = len(_FakeSam2Processor._cached_prompts or [])
        return _FakeSam2Outputs(num_objects=num_objects)


@pytest.fixture()
def fake_sam2(monkeypatch):
    fake_module = types.SimpleNamespace(
        Sam2Model=_FakeSam2Model, Sam2Processor=_FakeSam2Processor
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_module)


@pytest.mark.slow
def test_load_sam2_model():
    """Test SAM2 model successfully loads."""
    from transformers import Sam2Model, Sam2Processor

    model, processor = load_sam2_model("facebook/sam2.1-hiera-small", device="cpu")

    assert isinstance(model, Sam2Model)
    assert isinstance(processor, Sam2Processor)


@pytest.mark.slow
def test_process_image_group():
    """Test processing a single image with detections."""
    test_csv = get_data("OSBS_029.csv")
    test_image_dir = os.path.dirname(get_data("OSBS_029.tif"))

    df = pd.read_csv(test_csv)

    model, processor = load_sam2_model("facebook/sam2.1-hiera-small", device="cpu")

    polygons = process_image_group(
        image_path="OSBS_029.tif",
        detections=df,
        model=model,
        processor=processor,
        device="cpu",
        image_root=test_image_dir,
        box_batch_size=2,
    )

    assert len(polygons) == len(df)

    for poly_wkt in polygons:
        poly = wkt.loads(poly_wkt)
        assert poly is not None


@pytest.mark.slow
def test_convert_boxes_to_polygons(tmp_path):
    """Test the main conversion function directly."""
    test_csv = get_data("OSBS_029.csv")
    test_image_dir = os.path.dirname(get_data("OSBS_029.tif"))
    output_csv = tmp_path / "polygons.csv"
    viz_dir = tmp_path / "viz"

    input_df = pd.read_csv(test_csv)

    convert_boxes_to_polygons(
        input_csv=test_csv,
        output_csv=str(output_csv),
        image_root=test_image_dir,
        box_batch_size=2,
        visualize=True,
        viz_output_dir=str(viz_dir),
        device="cpu",
    )

    assert output_csv.exists()

    result_df = pd.read_csv(output_csv)
    assert "polygon_geometry" in result_df.columns
    assert len(result_df) == len(input_df)

    for poly_wkt in result_df["polygon_geometry"]:
        poly = wkt.loads(poly_wkt)
        assert poly is not None

    assert viz_dir.exists()
    viz_files = list(viz_dir.glob("*.png"))
    assert len(viz_files) > 0, "No visualization files were created"


@pytest.mark.slow
def test_polygon_box_overlap():
    """Test that output polygons overlap with input bounding boxes."""
    test_csv = get_data("OSBS_029.csv")
    test_image_dir = os.path.dirname(get_data("OSBS_029.tif"))

    df = pd.read_csv(test_csv)

    model, processor = load_sam2_model("facebook/sam2.1-hiera-small", device="cpu")

    polygons = process_image_group(
        image_path="OSBS_029.tif",
        detections=df,
        model=model,
        processor=processor,
        device="cpu",
        image_root=test_image_dir,
        box_batch_size=2,
    )

    for idx, (_, row) in enumerate(df.iterrows()):
        poly = wkt.loads(polygons[idx])

        if poly.is_empty:
            continue

        bbox = box(row["xmin"], row["ymin"], row["xmax"], row["ymax"])
        intersection = poly.intersection(bbox)
        assert intersection.area > 0, f"Polygon {idx} has no overlap with its bounding box"


@pytest.mark.slow
def test_sam_cli_end_to_end(tmp_path):
    """Test complete deepforest-sam CSV workflow with visualization."""
    test_csv = get_data("OSBS_029.csv")
    test_image_dir = os.path.dirname(get_data("OSBS_029.tif"))

    df = pd.read_csv(test_csv)

    output_csv = tmp_path / "polygons.csv"
    viz_dir = tmp_path / "viz"

    args = [
        sys.executable,
        str(SAM_SCRIPT),
        test_csv,
        "-o",
        str(output_csv),
        "--image-root",
        test_image_dir,
        "--box-batch",
        "2",
        "--device",
        "cpu",
        "--mask-threshold",
        "0.5",
        "--iou-threshold",
        "0.5",
        "--visualize",
        "--viz-output-dir",
        str(viz_dir),
    ]

    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert output_csv.exists(), f"Expected output file not found: {output_csv}"

    df_out = pd.read_csv(output_csv)
    assert "polygon_geometry" in df_out.columns
    assert len(df_out) == len(df)

    valid_polygons = 0
    for poly_wkt in df_out["polygon_geometry"]:
        poly = wkt.loads(poly_wkt)
        assert poly is not None
        if not poly.is_empty:
            valid_polygons += 1

    assert valid_polygons > 0, "All polygons are empty"

    assert viz_dir.exists()
    viz_files = list(viz_dir.glob("*.png"))
    assert len(viz_files) > 0, "No visualization files were created"


def test_load_sam2_model_with_mock(fake_sam2):
    sam = Sam2PolygonModel.load_model(model_name="facebook/sam2.1-hiera-small")
    assert sam.model is not None
    assert sam.processor is not None


def test_mask_to_polygon_accepts_device_tensor():
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("Requires CUDA or MPS")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    mask = torch.zeros((32, 32), device=device)
    mask[5:20, 5:20] = 1
    mask_uint8 = (mask.detach().cpu().numpy() > 0.5).astype(np.uint8)
    polygon = utilities.mask_to_polygon(mask_uint8)
    assert polygon is not None
    assert not polygon.is_empty


def test_predict_polygons_from_predict_image(fake_sam2):
    model = deepforest()
    model.load_model(model_name="weecology/deepforest-tree")
    image_path = get_data("OSBS_029.png")
    results = model.predict_image(path=image_path)

    polygons = model.predict_polygons(
        results=results,
        path=image_path,
        model_name="facebook/sam2.1-hiera-small",
        prompt_mode="box",
    )

    assert polygons is not None
    assert len(polygons) > 0
    assert "geometry" in polygons.columns
    assert utilities.determine_geometry_type(polygons) == "polygon"


def test_predict_polygons_from_predict_tile(fake_sam2):
    model = deepforest()
    model.load_model(model_name="weecology/deepforest-tree")
    tile_path = get_data("OSBS_029.tif")
    results = model.predict_tile(path=tile_path, patch_size=300, patch_overlap=0.25)

    polygons = model.predict_polygons(
        results=results,
        path=tile_path,
        model_name="facebook/sam2.1-hiera-small",
        prompt_mode="box",
    )

    assert polygons is not None
    assert len(polygons) > 0
    assert "geometry" in polygons.columns
    assert utilities.determine_geometry_type(polygons) == "polygon"


def test_predict_polygons_point_workflow(fake_sam2):
    model = deepforest()
    model.load_model(model_name="weecology/deepforest-tree")
    image_path = get_data("OSBS_029.png")
    results = model.predict_image(path=image_path).copy()
    results["x"] = (results["xmin"] + results["xmax"]) / 2.0
    results["y"] = (results["ymin"] + results["ymax"]) / 2.0
    results["geometry"] = [
        shapely.geometry.Point(x, y)
        for x, y in zip(results["x"], results["y"], strict=False)
    ]
    results = results.drop(columns=["xmin", "ymin", "xmax", "ymax"])
    results = utilities.__pandas_to_geodataframe__(results)

    polygons = model.predict_polygons(
        results=results,
        path=image_path,
        model_name="facebook/sam2.1-hiera-small",
        prompt_mode="point",
    )

    assert polygons is not None
    assert len(polygons) > 0
    assert "geometry" in polygons.columns
    assert utilities.determine_geometry_type(polygons) == "polygon"
