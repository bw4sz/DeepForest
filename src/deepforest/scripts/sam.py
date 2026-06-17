"""SAM2 polygon post-processing for DeepForest predictions.

The core box-to-polygon SAM2 workflow in this module was contributed by
Josh Veitch-Michaelis in https://github.com/weecology/DeepForest/pull/1158
(issue #460). Extensions include point prompts, ``main.deepforest`` API
integration, and the ``deepforest sam2-polygons`` CLI subcommand.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from shapely import wkt
from shapely.geometry import Polygon
from tqdm import tqdm
from transformers import Sam2Model, Sam2Processor

from deepforest import utilities
from deepforest.utilities import mask_to_polygon
from deepforest.visualize import plot_results

logger = logging.getLogger(__name__)

# SAM2 mask quality degrades when too many point prompts are passed at once.
DEFAULT_MAX_POINT_PROMPTS = 12


def resolve_device(device: str | None) -> str:
    """Resolve device string for SAM2 inference."""
    if device is None or device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device


def load_sam2_model(
    model_name: str,
    device: str,
    hf_token: str | None = None,
) -> tuple[Sam2Model, Sam2Processor]:
    """Load SAM2 model and processor from Hugging Face.

    Args:
        model_name: Name of the SAM2 model on Hugging Face
        device: Device to load model on ('cuda', 'mps', or 'cpu')
        hf_token: Optional Hugging Face token for gated models

    Returns:
        Tuple of (model, processor)
    """
    token = hf_token or os.getenv("HF_TOKEN")
    load_kwargs = {"token": token} if token else {}
    processor = Sam2Processor.from_pretrained(model_name, **load_kwargs)
    model = Sam2Model.from_pretrained(model_name, **load_kwargs)
    model = model.to(device)
    return model, processor


def _point_coordinates_from_detections(detections: pd.DataFrame) -> np.ndarray:
    if "x" in detections.columns and "y" in detections.columns:
        x = detections["x"].astype(float).to_numpy()
        y = detections["y"].astype(float).to_numpy()
    elif "geometry" in detections.columns:
        x = detections.geometry.x.astype(float).to_numpy()
        y = detections.geometry.y.astype(float).to_numpy()
    else:
        raise ValueError("Point prompts require x/y columns or point geometry.")

    return np.column_stack([x, y])


def _negative_point_indices(
    coordinates: np.ndarray,
    focal_idx: int,
    *,
    max_point_prompts: int,
) -> list[int]:
    """Select other detections to use as negative point prompts for one focal
    tree."""
    other_indices = [idx for idx in range(len(coordinates)) if idx != focal_idx]
    max_negative = max_point_prompts - 1
    if max_negative <= 0 or len(other_indices) <= max_negative:
        return other_indices

    focal = coordinates[focal_idx]
    distances = np.sum((coordinates[other_indices] - focal) ** 2, axis=1)
    nearest_order = np.argsort(distances)
    return [other_indices[idx] for idx in nearest_order[:max_negative]]


def _build_focal_point_prompts(
    coordinates: np.ndarray,
    focal_indices: list[int],
    *,
    use_negative_point_prompts: bool,
    max_point_prompts: int = DEFAULT_MAX_POINT_PROMPTS,
) -> tuple[list[list[list[list[float]]]], list[list[list[int]]]]:
    """Build SAM2 point prompts for one or more focal detections."""
    objects: list[list[list[float]]] = []
    labels: list[list[int]] = []

    for focal_idx in focal_indices:
        if use_negative_point_prompts and len(coordinates) > 1:
            object_points = [
                [float(coordinates[focal_idx][0]), float(coordinates[focal_idx][1])]
            ]
            object_labels = [1]
            for other_idx in _negative_point_indices(
                coordinates,
                focal_idx,
                max_point_prompts=max_point_prompts,
            ):
                object_points.append(
                    [float(coordinates[other_idx][0]), float(coordinates[other_idx][1])]
                )
                object_labels.append(0)
        else:
            object_points = [
                [float(coordinates[focal_idx][0]), float(coordinates[focal_idx][1])]
            ]
            object_labels = [1]

        objects.append(object_points)
        labels.append(object_labels)

    return [objects], [labels]


def _mask_to_polygon_result(
    mask,
    mask_threshold: float,
    iou_threshold: float,
    best_iou: float,
) -> tuple[Polygon, float]:
    if best_iou < iou_threshold:
        return Polygon(), best_iou

    mask_np = mask.numpy() if hasattr(mask, "numpy") else np.asarray(mask)
    if isinstance(mask_np, torch.Tensor):
        mask_np = mask_np.cpu().numpy()
    mask_uint8 = (mask_np > mask_threshold).astype(np.uint8)
    return mask_to_polygon(mask_uint8), best_iou


def _run_sam2_prompt_batch(
    image: Image.Image,
    model: Sam2Model,
    processor: Sam2Processor,
    device: str,
    *,
    input_boxes: list[list[float]] | None = None,
    input_points: list[list[list[list[float]]]] | None = None,
    input_labels: list[list[list[int]]] | None = None,
    mask_threshold: float,
    iou_threshold: float,
    num_objects: int,
) -> list[tuple[Polygon, float]]:
    processor_kwargs = {"images": image, "return_tensors": "pt"}
    if input_boxes is not None:
        processor_kwargs["input_boxes"] = [input_boxes]
    else:
        processor_kwargs["input_points"] = input_points
        processor_kwargs["input_labels"] = input_labels

    inputs = processor(**processor_kwargs).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"],
        binarize=False,
        mask_interpolation_mode="nearest",
    )[0]
    iou_scores = outputs.iou_scores.cpu()

    results: list[tuple[Polygon, float]] = []
    for idx, mask_set in enumerate(masks):
        if idx >= num_objects:
            break
        best_idx = iou_scores[0, idx].argmax().item()
        best_iou = iou_scores[0, idx, best_idx].item()
        polygon, score = _mask_to_polygon_result(
            mask_set[best_idx],
            mask_threshold=mask_threshold,
            iou_threshold=iou_threshold,
            best_iou=best_iou,
        )
        results.append((polygon, score))

    return results


def _process_box_prompt_chunk(
    image: Image.Image,
    detections: pd.DataFrame,
    model: Sam2Model,
    processor: Sam2Processor,
    device: str,
    mask_threshold: float,
    iou_threshold: float,
) -> list[tuple[Polygon, float]]:
    """Run SAM2 on a batch of box prompts for one image."""
    boxes = detections[["xmin", "ymin", "xmax", "ymax"]].astype(float).values.tolist()
    return _run_sam2_prompt_batch(
        image=image,
        model=model,
        processor=processor,
        device=device,
        input_boxes=boxes,
        mask_threshold=mask_threshold,
        iou_threshold=iou_threshold,
        num_objects=len(detections),
    )


def _process_point_prompts(
    image: Image.Image,
    detections: pd.DataFrame,
    model: Sam2Model,
    processor: Sam2Processor,
    device: str,
    prompt_batch_size: int,
    mask_threshold: float,
    iou_threshold: float,
    use_negative_point_prompts: bool,
    max_point_prompts: int = DEFAULT_MAX_POINT_PROMPTS,
) -> list[tuple[Polygon, float]]:
    """Run SAM2 for each point detection, optionally with other points as
    negatives."""
    coordinates = _point_coordinates_from_detections(detections)
    if len(coordinates) == 0:
        return []

    all_results: list[tuple[Polygon, float]] = []
    for start in range(0, len(coordinates), prompt_batch_size):
        focal_indices = list(
            range(start, min(start + prompt_batch_size, len(coordinates)))
        )
        input_points, input_labels = _build_focal_point_prompts(
            coordinates,
            focal_indices,
            use_negative_point_prompts=use_negative_point_prompts,
            max_point_prompts=max_point_prompts,
        )
        batch_results = _run_sam2_prompt_batch(
            image=image,
            model=model,
            processor=processor,
            device=device,
            input_points=input_points,
            input_labels=input_labels,
            mask_threshold=mask_threshold,
            iou_threshold=iou_threshold,
            num_objects=len(focal_indices),
        )
        all_results.extend(batch_results)

    return all_results


def process_image_group(
    image_path: str,
    detections: pd.DataFrame,
    model: Sam2Model,
    processor: Sam2Processor,
    device: str,
    image_root: str = "",
    box_batch_size: int = 32,
    prompt_mode: str = "box",
    mask_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    use_negative_point_prompts: bool = True,
    max_point_prompts: int = DEFAULT_MAX_POINT_PROMPTS,
    viz_output_dir: str | None = None,
) -> list[str]:
    """Process all detections for a single image.

    Args:
        image_path: Path to the image file
        detections: DataFrame of detections for this image
        model: SAM2 model
        processor: SAM2 processor
        device: Device to run inference on
        image_root: Root directory to prepend to image_path if needed
        box_batch_size: Maximum number of prompts per forward pass
        prompt_mode: ``box`` or ``point`` prompts
        mask_threshold: Threshold for binarizing SAM2 mask outputs
        iou_threshold: Minimum IoU score to accept a polygon
        use_negative_point_prompts: For point mode, mark other detections as negative prompts
        max_point_prompts: Maximum SAM2 point prompts per focal tree (including the positive)
        viz_output_dir: Directory to save visualizations (if not None)

    Returns:
        List of WKT polygon strings (empty polygon when below IoU threshold)
    """
    full_path = os.path.join(image_root, image_path) if image_root else image_path
    image = Image.open(full_path).convert("RGB")

    if prompt_mode == "box":
        required = {"xmin", "ymin", "xmax", "ymax"}
        missing = required.difference(detections.columns)
        if missing:
            raise ValueError(f"Missing box columns for SAM2 prompts: {sorted(missing)}")
    else:
        _point_coordinates_from_detections(detections)

    all_polygons: list[str] = []
    if prompt_mode == "box":
        for start in range(0, len(detections), box_batch_size):
            chunk = detections.iloc[start : start + box_batch_size]
            chunk_results = _process_box_prompt_chunk(
                image=image,
                detections=chunk,
                model=model,
                processor=processor,
                device=device,
                mask_threshold=mask_threshold,
                iou_threshold=iou_threshold,
            )
            all_polygons.extend(polygon.wkt for polygon, _ in chunk_results)
    else:
        chunk_results = _process_point_prompts(
            image=image,
            detections=detections,
            model=model,
            processor=processor,
            device=device,
            prompt_batch_size=box_batch_size,
            mask_threshold=mask_threshold,
            iou_threshold=iou_threshold,
            use_negative_point_prompts=use_negative_point_prompts,
            max_point_prompts=max_point_prompts,
        )
        all_polygons.extend(polygon.wkt for polygon, _ in chunk_results)

    if viz_output_dir is not None:
        viz_df = detections.copy()
        viz_df["polygon_geometry"] = all_polygons
        viz_df["geometry"] = viz_df["polygon_geometry"].apply(
            lambda value: wkt.loads(value) if pd.notna(value) else None
        )
        viz_df = viz_df[
            viz_df["geometry"].apply(lambda geom: geom is not None and not geom.is_empty)
        ]

        if len(viz_df) > 0:
            if "label" not in viz_df.columns:
                viz_df["label"] = "Tree"
            if "score" not in viz_df.columns:
                viz_df["score"] = 1.0

            with Image.open(full_path) as img:
                width, height = img.size

            image_name = Path(image_path).stem
            viz_path = os.path.join(viz_output_dir, f"{image_name}_polygons.png")
            plot_results(
                results=viz_df,
                image=full_path,
                savedir=os.path.dirname(viz_path),
                basename=os.path.splitext(os.path.basename(viz_path))[0],
                height=height,
                width=width,
                show=False,
            )

    return all_polygons


def process_detections_dataframe(
    image: Image.Image,
    detections: pd.DataFrame,
    model: Sam2Model,
    processor: Sam2Processor,
    device: str,
    prompt_mode: str,
    prompt_batch_size: int = 32,
    mask_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    use_negative_point_prompts: bool = True,
    max_point_prompts: int = DEFAULT_MAX_POINT_PROMPTS,
) -> pd.DataFrame:
    """Convert detections for one image into polygon rows."""
    if prompt_mode == "box":
        required = {"xmin", "ymin", "xmax", "ymax"}
        missing = required.difference(detections.columns)
        if missing:
            raise ValueError(f"Missing box columns for SAM2 prompts: {sorted(missing)}")
    else:
        _point_coordinates_from_detections(detections)

    if len(detections) == 0:
        return detections.iloc[0:0].copy()

    rows = []
    if prompt_mode == "box":
        for start in range(0, len(detections), prompt_batch_size):
            chunk = detections.iloc[start : start + prompt_batch_size]
            chunk_results = _process_box_prompt_chunk(
                image=image,
                detections=chunk,
                model=model,
                processor=processor,
                device=device,
                mask_threshold=mask_threshold,
                iou_threshold=iou_threshold,
            )
            for idx, (polygon, best_iou) in enumerate(chunk_results):
                if polygon.is_empty:
                    continue
                row = chunk.iloc[idx].to_dict()
                row["geometry"] = polygon
                row["score"] = best_iou
                rows.append(row)
    else:
        point_results = _process_point_prompts(
            image=image,
            detections=detections,
            model=model,
            processor=processor,
            device=device,
            prompt_batch_size=prompt_batch_size,
            mask_threshold=mask_threshold,
            iou_threshold=iou_threshold,
            use_negative_point_prompts=use_negative_point_prompts,
            max_point_prompts=max_point_prompts,
        )
        for idx, (polygon, best_iou) in enumerate(point_results):
            if polygon.is_empty:
                continue
            row = detections.iloc[idx].to_dict()
            row["geometry"] = polygon
            row["score"] = best_iou
            rows.append(row)

    if not rows:
        return detections.iloc[0:0].copy()

    return pd.DataFrame(rows)


def convert_boxes_to_polygons(
    input_csv: str,
    output_csv: str,
    model_name: str = "facebook/sam2.1-hiera-small",
    box_batch_size: int = 32,
    image_root: str = "",
    visualize: bool = False,
    viz_output_dir: str = ".",
    mask_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    device: str | None = None,
    hf_token: str | None = None,
) -> None:
    """Convert DeepForest bounding boxes to polygons using SAM2.

    Args:
        input_csv: Path to input CSV with DeepForest predictions
        output_csv: Path to save output CSV with polygons
        model_name: Hugging Face model name for SAM2
        box_batch_size: Maximum number of boxes to process per forward pass
        image_root: Root directory to prepend to image paths in CSV
        visualize: Whether to create visualization images
        viz_output_dir: Directory to save visualization images
        mask_threshold: Threshold for binarizing SAM2 mask outputs
        iou_threshold: Minimum IoU score to accept a polygon
        device: Device to use ('cuda', 'mps', or 'cpu'). Auto-detects if None.
        hf_token: Optional Hugging Face token
    """
    df = pd.read_csv(input_csv)

    required_cols = ["xmin", "ymin", "xmax", "ymax", "image_path"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    resolved_device = resolve_device(device)
    logger.info("Using device: %s", resolved_device)
    logger.info("Loading SAM2 model: %s", model_name)
    model, processor = load_sam2_model(model_name, resolved_device, hf_token=hf_token)

    grouped = df.groupby("image_path")
    total_images = len(grouped)

    all_polygons: list[str] = []

    if visualize:
        os.makedirs(viz_output_dir, exist_ok=True)

    for image_path, group in tqdm(grouped, desc="Processing images", total=total_images):
        polygons = process_image_group(
            image_path,
            group,
            model,
            processor,
            resolved_device,
            image_root,
            box_batch_size,
            prompt_mode="box",
            mask_threshold=mask_threshold,
            iou_threshold=iou_threshold,
            viz_output_dir=viz_output_dir if visualize else None,
        )
        all_polygons.extend(polygons)

    df["polygon_geometry"] = all_polygons

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Saved results to %s", output_csv)


class Sam2PolygonModel:
    """Cached SAM2 model for ``deepforest.predict_polygons``."""

    def __init__(
        self, model: Sam2Model, processor: Sam2Processor, device: str, model_name: str
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.model_name = model_name

    @classmethod
    def load_model(
        cls,
        model_name: str = "facebook/sam2.1-hiera-small",
        hf_token: str | None = None,
        device: str = "auto",
    ):
        """Load SAM2 model and processor from Hugging Face."""
        resolved_device = resolve_device(device)
        try:
            model, processor = load_sam2_model(
                model_name=model_name,
                device=resolved_device,
                hf_token=hf_token,
            )
        except Exception as exc:
            raise ValueError(
                f"Unable to load SAM2 model '{model_name}' from Hugging Face."
            ) from exc

        return cls(
            model=model,
            processor=processor,
            device=resolved_device,
            model_name=model_name,
        )

    @staticmethod
    def _normalize_input(results):
        if results is None:
            raise ValueError("results cannot be None")
        if len(results) == 0:
            return utilities.__pandas_to_geodataframe__(results.copy())
        return results.copy()

    @staticmethod
    def _resolve_prompt_mode(results: pd.DataFrame, prompt_mode: str) -> str:
        if prompt_mode not in {"auto", "box", "point"}:
            raise ValueError("prompt_mode must be one of: auto, box, point")

        inferred_mode = utilities.determine_geometry_type(results)
        if inferred_mode not in {"box", "point"}:
            raise ValueError(
                f"SAM2 polygon prompts require box or point input, got geometry type "
                f"'{inferred_mode}'"
            )

        if prompt_mode == "auto":
            return inferred_mode
        if prompt_mode != inferred_mode:
            raise ValueError(
                f"prompt_mode='{prompt_mode}' does not match results geometry '{inferred_mode}'"
            )
        return prompt_mode

    def predict_polygons(
        self,
        results,
        image: np.ndarray | None = None,
        path: str | None = None,
        root_dir: str | None = None,
        prompt_mode: str = "auto",
        mask_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        prompt_batch_size: int = 32,
        use_negative_point_prompts: bool = True,
        max_point_prompts: int = DEFAULT_MAX_POINT_PROMPTS,
    ):
        """Convert DeepForest box/point predictions into polygon
        predictions."""
        results = self._normalize_input(results)
        if len(results) == 0:
            return utilities.__pandas_to_geodataframe__(results)

        resolved_prompt_mode = self._resolve_prompt_mode(results, prompt_mode=prompt_mode)

        if path is not None:
            image_obj = Image.open(path).convert("RGB")
            image_name = os.path.basename(path)
            if "image_path" in results.columns:
                selected = results[results.image_path == image_name]
                if len(selected) == 0:
                    selected = results
            else:
                selected = results
            polygon_df = process_detections_dataframe(
                image=image_obj,
                detections=selected,
                model=self.model,
                processor=self.processor,
                device=self.device,
                prompt_mode=resolved_prompt_mode,
                prompt_batch_size=prompt_batch_size,
                mask_threshold=mask_threshold,
                iou_threshold=iou_threshold,
                use_negative_point_prompts=use_negative_point_prompts,
                max_point_prompts=max_point_prompts,
            )
            gdf = utilities.__pandas_to_geodataframe__(polygon_df)
            gdf.root_dir = os.path.dirname(path)
            return gdf

        if image is not None:
            image_obj = Image.fromarray(image.astype(np.uint8)).convert("RGB")
            polygon_df = process_detections_dataframe(
                image=image_obj,
                detections=results,
                model=self.model,
                processor=self.processor,
                device=self.device,
                prompt_mode=resolved_prompt_mode,
                prompt_batch_size=prompt_batch_size,
                mask_threshold=mask_threshold,
                iou_threshold=iou_threshold,
                use_negative_point_prompts=use_negative_point_prompts,
                max_point_prompts=max_point_prompts,
            )
            gdf = utilities.__pandas_to_geodataframe__(polygon_df)
            gdf.root_dir = None
            return gdf

        if "image_path" not in results.columns:
            raise ValueError(
                "results must include image_path when image/path are not provided"
            )

        inferred_root = root_dir or getattr(results, "root_dir", None)
        if inferred_root is None:
            raise ValueError(
                "No image root found. Pass root_dir or provide results with results.root_dir."
            )

        output_groups = []
        for image_path, group in results.groupby("image_path"):
            full_path = os.path.join(inferred_root, image_path)
            image_obj = Image.open(full_path).convert("RGB")
            output_groups.append(
                process_detections_dataframe(
                    image=image_obj,
                    detections=group,
                    model=self.model,
                    processor=self.processor,
                    device=self.device,
                    prompt_mode=resolved_prompt_mode,
                    prompt_batch_size=prompt_batch_size,
                    mask_threshold=mask_threshold,
                    iou_threshold=iou_threshold,
                    use_negative_point_prompts=use_negative_point_prompts,
                    max_point_prompts=max_point_prompts,
                )
            )

        if output_groups:
            polygon_df = pd.concat(output_groups, ignore_index=True)
        else:
            polygon_df = results.iloc[0:0].copy()

        gdf = utilities.__pandas_to_geodataframe__(polygon_df)
        gdf.root_dir = inferred_root
        return gdf


def sam2_polygons(
    config: DictConfig,
    input_path: str | None = None,
    predictions_csv: str | None = None,
    output_path: str | None = None,
    root_dir: str | None = None,
    mode: str = "single",
    prompt_mode: str = "auto",
    model_name: str = "facebook/sam2.1-hiera-small",
    mask_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    prompt_batch_size: int = 32,
    use_negative_point_prompts: bool = True,
    max_point_prompts: int = DEFAULT_MAX_POINT_PROMPTS,
):
    """Convert DeepForest point/box predictions to polygons with SAM2."""
    from deepforest.main import deepforest as deepforest_model

    m = deepforest_model(config=config)
    m.create_trainer(logger=False)

    if predictions_csv is not None:
        if root_dir is None:
            root_dir = config.validation.root_dir
        if root_dir is None:
            root_dir = os.path.dirname(predictions_csv)
        results = utilities.read_file(predictions_csv, root_dir=root_dir)
        path_for_image = input_path
    else:
        if input_path is None:
            raise ValueError(
                "input_path is required when predictions_csv is not provided"
            )
        path_for_image = input_path
        if mode == "single":
            results = m.predict_image(path=input_path)
        elif mode == "tile":
            results = m.predict_tile(
                path=input_path,
                patch_size=config.patch_size,
                patch_overlap=config.patch_overlap,
                iou_threshold=config.nms_thresh,
            )
        elif mode == "csv":
            if root_dir is None:
                root_dir = config.validation.root_dir
            if root_dir is None:
                root_dir = os.path.dirname(input_path)
            results = m.predict_file(csv_file=input_path, root_dir=root_dir)
            path_for_image = None
        else:
            raise ValueError(f"Invalid mode: {mode}. Pick one of single/tile/csv.")

    if results is None:
        raise ValueError("No predictions found to convert to polygons.")

    polygons = m.predict_polygons(
        results=results,
        path=path_for_image,
        root_dir=root_dir,
        prompt_mode=prompt_mode,
        model_name=model_name,
        mask_threshold=mask_threshold,
        iou_threshold=iou_threshold,
        prompt_batch_size=prompt_batch_size,
        use_negative_point_prompts=use_negative_point_prompts,
        max_point_prompts=max_point_prompts,
    )

    if output_path is not None:
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if output_path.endswith(".shp") or output_path.endswith(".gpkg"):
            geo = utilities.image_to_geo_coordinates(polygons)
            geo.to_file(output_path)
        else:
            polygons.to_csv(output_path, index=False)

    return polygons


def main() -> None:
    """CLI entrypoint for CSV box-to-polygon conversion (``deepforest-
    sam``)."""
    parser = argparse.ArgumentParser(
        description="Convert DeepForest bounding boxes to polygons using SAM2"
    )
    parser.add_argument("input", help="Path to input CSV with DeepForest predictions")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output CSV (default: input with '_polygons' suffix)",
    )
    parser.add_argument(
        "--model",
        default="facebook/sam2.1-hiera-small",
        help="SAM2 model name from HuggingFace (default: facebook/sam2.1-hiera-small)",
    )
    parser.add_argument(
        "--box-batch",
        type=int,
        default=32,
        help="Maximum number of boxes to process per forward pass (default: 32)",
    )
    parser.add_argument(
        "--image-root",
        default="",
        help="Root directory to prepend to image paths in CSV",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualization images with polygons overlaid",
    )
    parser.add_argument(
        "--viz-output-dir",
        default=".",
        help="Directory to save visualization images (default: current directory)",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Threshold for binarizing SAM2 mask outputs (default: 0.5)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Minimum IoU score to accept a polygon (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Device to use for inference (default: auto-detect mps > cuda > cpu)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.output is None:
        input_path = Path(args.input)
        output_path = input_path.parent / f"{input_path.stem}_polygons{input_path.suffix}"
        args.output = str(output_path)

    convert_boxes_to_polygons(
        args.input,
        args.output,
        model_name=args.model,
        box_batch_size=args.box_batch,
        image_root=args.image_root,
        visualize=args.visualize,
        viz_output_dir=args.viz_output_dir,
        mask_threshold=args.mask_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
