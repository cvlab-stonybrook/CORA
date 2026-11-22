"""Objects of an image: connected components of the semantic class masks (supplement Sec. 4).

"To isolate individual objects within an image, we first decompose each category mask into separate
connected components." Every component is one region with the id ``"<class id>_<k>"`` (supplement
Fig. 3 / Tables 4-5, e.g. ``"11_2"`` = second person region, ``"5_20"`` = twentieth pole region), where
``k`` counts from 1 per class in the raster order of the component's first pixel (top to bottom, then left
to right), so ids are deterministic. Touching same-class instances therefore merge into one region.

Components smaller than ``min_area`` pixels get no region (the remaining ones are numbered contiguously).
Regions are stored per image as a uint16 region map (``0`` = no region, value = ``Region.index``) plus a
JSON table. Polygons (supplement Sec. 7, "mask polygons") can be extracted from region masks with
:func:`mask_polygons`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import ndimage

from cora import constants as C
from cora.config import Config
from cora.data.adapters import DatasetAdapter, atomic_write

MAX_REGIONS = np.iinfo(np.uint16).max  # the region map is uint16
FORMAT_VERSION = 2


@dataclass
class Region:
    """One connected component of a class mask."""

    index: int  # 1-based value in the region map
    region_id: str  # constants.region_id(label_id, k), k = 1-based order within the class
    label_id: int
    label_name: str
    area: int
    bbox: Tuple[int, int, int, int]  # x0, y0, x1, y1 with x1 / y1 exclusive
    centroid: Tuple[float, float]  # x, y


@dataclass(eq=False)
class ImageObjects:
    """All regions of one image."""

    image_id: str
    height: int
    width: int
    region_map: np.ndarray  # (H, W) uint16, 0 where no region
    regions: List[Region]  # sorted by index
    connectivity: Optional[int] = None  # parameters used for the extraction (recorded to detect stale files)
    min_area: Optional[int] = None
    _by_id: Dict[str, Region] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.region_map.shape != (self.height, self.width):
            raise ValueError(f"{self.image_id}: region map {self.region_map.shape} does not match {(self.height, self.width)}")
        self._by_id = {r.region_id: r for r in self.regions}
        if len(self._by_id) != len(self.regions):
            raise ValueError(f"{self.image_id}: duplicate region ids")

    def get(self, region_id: str) -> Region:
        """Region with this id; ``KeyError`` naming the image and the known ids when absent."""
        try:
            return self._by_id[region_id]
        except KeyError:
            raise KeyError(f"{self.image_id}: no region {region_id!r}; image has {sorted(self._by_id)}") from None

    def mask(self, region_ids: Sequence[str]) -> np.ndarray:
        """(H, W) bool mask, the union of the named regions (empty selection -> all False)."""
        indices = [self.get(rid).index for rid in region_ids]
        if not indices:
            return np.zeros(self.region_map.shape, dtype=bool)
        if len(indices) == 1:
            return self.region_map == indices[0]
        return np.isin(self.region_map, indices)

    def of_class(self, label_id: int) -> List[Region]:
        """Regions of one class, in id order."""
        return [r for r in self.regions if r.label_id == label_id]


# ----------------------------------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------------------------------
def _components_in_raster_order(mask: np.ndarray, connectivity: int):
    """Label ``mask`` and order the components by the raster position of their first pixel.

    Returns (labels, order, stats, centroids) where ``order[j]`` is the cv2 label of the component that is
    ``j``-th in raster order. The order is computed from the pixels, not assumed from the labelling
    algorithm of OpenCV.
    """
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=connectivity, ltype=cv2.CV_32S
    )
    if count <= 1:
        return labels, np.zeros(0, dtype=np.int64), stats, centroids
    flat = labels.ravel()
    positions = np.flatnonzero(flat)
    found, first = np.unique(flat[positions], return_index=True)  # first occurrence = smallest flat index
    order = found[np.argsort(positions[first], kind="stable")]
    return labels, order, stats, centroids


def extract_objects(
    image_id: str,
    label: np.ndarray,
    class_names: Sequence[str],
    *,
    connectivity: int = 8,
    min_area: int = 1,
    ignore_label: int = C.IGNORE_LABEL,
) -> ImageObjects:
    """Decompose every class mask of a label map into connected components (supplement Sec. 4).

    ``label`` is an (H, W) integer map of class ids; ``ignore_label`` pixels belong to no region. Every
    other class id present must have an entry in ``class_names``.
    """
    if label.ndim != 2 or not np.issubdtype(label.dtype, np.integer):
        raise ValueError(f"{image_id}: label must be a 2-D integer array, got {label.dtype} {label.shape}")
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    if min_area < 1:
        raise ValueError(f"min_area must be at least 1, got {min_area}")
    height, width = label.shape
    present = [int(v) for v in np.unique(label) if int(v) != ignore_label]
    bad = [v for v in present if not 0 <= v < len(class_names)]
    if bad:
        raise ValueError(f"{image_id}: label map holds class ids {bad} outside 0..{len(class_names) - 1} (ignore label {ignore_label})")

    region_map = np.zeros((height, width), dtype=np.uint16)
    regions: List[Region] = []
    for class_id in present:
        mask = label == class_id
        labels, order, stats, centroids = _components_in_raster_order(mask, connectivity)
        remap = np.zeros(len(stats), dtype=np.int64)
        k = 0
        for cv_label in order:
            left, top, w, h, area = (int(v) for v in stats[cv_label])
            if area < min_area:
                continue
            k += 1
            if len(regions) + 1 > MAX_REGIONS:
                raise ValueError(f"{image_id}: more than {MAX_REGIONS} regions do not fit the uint16 region map")
            index = len(regions) + 1
            remap[cv_label] = index
            cx, cy = (float(v) for v in centroids[cv_label])
            regions.append(
                Region(index, C.region_id(class_id, k), class_id, str(class_names[class_id]), area,
                       (left, top, left + w, top + h), (cx, cy))
            )
        inside = labels > 0
        region_map[inside] = remap[labels[inside]].astype(np.uint16)
    return ImageObjects(image_id, height, width, region_map, regions, connectivity, min_area)


# ----------------------------------------------------------------------------------------------
# Polygons
# ----------------------------------------------------------------------------------------------
def _outer_polygons(mask: np.ndarray, tolerance: float) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if tolerance > 0:
        contours = [_simplified(c, tolerance) for c in contours]
    return [c.reshape(-1, 2) for c in contours]


def _simplified(contour: np.ndarray, tolerance: float) -> np.ndarray:
    """``approxPolyDP`` of a contour; the exact contour is kept when simplification would leave fewer than 3 vertices."""
    approx = cv2.approxPolyDP(contour, tolerance, True)
    return approx if len(approx) >= 3 or len(contour) < 3 else contour


def _component_polygons(mask: np.ndarray, tolerance: float) -> List[np.ndarray]:
    """Outer polygon of every 8-connected component plus the polygon of each enclosed hole."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    polygons: List[np.ndarray] = []
    for label in range(1, count):
        left, top, w, h, _area = (int(v) for v in stats[label])
        comp = np.pad(labels[top : top + h, left : left + w] == label, 1)
        offset = np.array([left - 1, top - 1])
        polygons.extend(p + offset for p in _outer_polygons(comp, tolerance))
        holes = ndimage.binary_fill_holes(comp) & ~comp
        if holes.any():
            n_holes, hole_labels = cv2.connectedComponents(holes.astype(np.uint8), connectivity=4)
            for hole in range(1, n_holes):
                polygons.extend(p + offset for p in _outer_polygons(hole_labels == hole, tolerance))
    return polygons


def mask_polygons(mask: np.ndarray, tolerance: float = 0.0, *, include_holes: bool = False) -> List[List[List[int]]]:
    """Contours of a boolean mask as polygons ``[[x, y], ...]`` in pixel coordinates (supplement Sec. 7).

    By default one polygon per outermost contour (``cv2.RETR_EXTERNAL``). For the mask of a single region
    that is its outer contour; for a union of regions, a component lying inside a hole of another component
    is omitted. ``tolerance > 0`` simplifies every polygon with ``cv2.approxPolyDP`` (maximum deviation in
    pixels); a polygon that would collapse to fewer than 3 vertices keeps its exact contour. With
    ``include_holes`` the outer polygon of every 8-connected component is returned together with one
    polygon per enclosed hole, so nested components are kept and :func:`polygons_to_mask` (even-odd rule)
    reproduces the mask. Polygons are ordered by the raster position of their topmost-leftmost point.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    if tolerance < 0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance}")
    mask = mask.astype(bool)
    if not mask.any():
        return []
    polygons = _component_polygons(mask, tolerance) if include_holes else _outer_polygons(mask, tolerance)
    polygons.sort(key=lambda p: min((int(y), int(x)) for x, y in p))
    return [[[int(x), int(y)] for x, y in p] for p in polygons]


def polygons_to_mask(polygons: Sequence[Sequence[Sequence[int]]], shape: Tuple[int, int]) -> np.ndarray:
    """Rasterise polygons from :func:`mask_polygons` with the even-odd rule (holes cancel their parent)."""
    out = np.zeros(shape, dtype=bool)
    for polygon in polygons:
        layer = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(layer, [np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)], 1)
        out ^= layer.astype(bool)
    return out


# ----------------------------------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------------------------------
def _paths(directory: Path, image_id: str) -> Tuple[Path, Path]:
    parts = image_id.split("/")
    if not image_id or image_id.startswith("/") or "\\" in image_id or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"invalid image id {image_id!r}")
    base = Path(directory).joinpath(*parts)
    return base.with_name(base.name + ".png"), base.with_name(base.name + ".json")


def _map_digest(region_map: np.ndarray) -> str:
    """Content hash of a uint16 region map (little-endian bytes), stored in the JSON next to the PNG."""
    return hashlib.blake2b(np.ascontiguousarray(region_map, dtype="<u2").tobytes(), digest_size=16).hexdigest()


def save_objects(objects: ImageObjects, directory: Path) -> None:
    """Write ``<directory>/<image_id>.png`` (uint16 region map) and ``.json`` (region table), atomically.

    The JSON file is written last, so a folder entry without it is treated as missing by :func:`load_objects`.
    It records a digest of the region map, so a PNG replaced while an older JSON is still in place (or the
    reverse) is reported by :func:`load_objects` instead of being read as a consistent pair.
    """
    png_path, json_path = _paths(directory, objects.image_id)
    region_map = objects.region_map.astype(np.uint16)
    ok, encoded = cv2.imencode(".png", region_map)
    if not ok:
        raise RuntimeError(f"{objects.image_id}: PNG encoding of the region map failed")
    table = {
        "version": FORMAT_VERSION,
        "image_id": objects.image_id,
        "height": objects.height,
        "width": objects.width,
        "connectivity": objects.connectivity,
        "min_area": objects.min_area,
        "map_digest": _map_digest(region_map),
        "regions": [asdict(r) for r in objects.regions],
    }
    atomic_write(png_path, encoded.tobytes())
    atomic_write(json_path, json.dumps(table).encode())


def objects_exist(directory: Path, image_id: str) -> bool:
    png_path, json_path = _paths(directory, image_id)
    return png_path.is_file() and json_path.is_file()


def load_objects(directory: Path, image_id: str) -> ImageObjects:
    """Read objects written by :func:`save_objects`; ``FileNotFoundError`` if absent, ``ValueError`` if inconsistent."""
    png_path, json_path = _paths(directory, image_id)
    if not (png_path.is_file() and json_path.is_file()):
        raise FileNotFoundError(f"no stored objects for {image_id!r} in {directory}")
    table = json.loads(json_path.read_text())
    if table.get("version") != FORMAT_VERSION or table.get("image_id") != image_id:
        raise ValueError(f"{json_path}: unexpected format version or image id")
    region_map = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if region_map is None or region_map.ndim != 2 or region_map.dtype != np.uint16:
        raise ValueError(f"{png_path}: not a single-channel uint16 region map")
    if table.get("map_digest") != _map_digest(region_map):
        raise ValueError(f"{png_path}: region map does not match {json_path.name} (files from different writes)")
    regions = [
        Region(r["index"], r["region_id"], r["label_id"], r["label_name"], r["area"], tuple(r["bbox"]), tuple(r["centroid"]))
        for r in table["regions"]
    ]
    if int(region_map.max(initial=0)) > len(regions):
        raise ValueError(f"{png_path}: region map holds an index above the {len(regions)} listed regions")
    return ImageObjects(image_id, table["height"], table["width"], region_map, regions,
                        table.get("connectivity"), table.get("min_area"))


def ensure_objects(cfg: Config, adapter: DatasetAdapter, split: str, image_id: str) -> ImageObjects:
    """Stored objects of an image, extracted from its label map and saved first when missing.

    Extraction uses ``cfg.data.connectivity`` and ``cfg.data.min_region_area``. Stored objects made with
    other values raise ``ValueError`` instead of being reused or overwritten, because instruction sets
    refer to region ids.
    """
    directory = cfg.objects_dir(split)
    if objects_exist(directory, image_id):
        stored = load_objects(directory, image_id)
        if (stored.connectivity, stored.min_area) != (cfg.data.connectivity, cfg.data.min_region_area):
            raise ValueError(
                f"{image_id}: stored objects use connectivity={stored.connectivity}, min_area={stored.min_area} but the "
                f"config asks for {cfg.data.connectivity} / {cfg.data.min_region_area}; delete {directory} or fix the config"
            )
        return stored
    objects = extract_objects(
        image_id, adapter.load_label(image_id), adapter.class_names,
        connectivity=cfg.data.connectivity, min_area=cfg.data.min_region_area,
    )
    save_objects(objects, directory)
    return objects


def count_regions(objects: ImageObjects) -> Dict[int, int]:
    """Number of regions per class id."""
    counts: Dict[int, int] = {}
    for region in objects.regions:
        counts[region.label_id] = counts.get(region.label_id, 0) + 1
    return counts
