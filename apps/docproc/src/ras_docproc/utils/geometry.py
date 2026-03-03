"""Bounding box and geometry utilities."""

from __future__ import annotations

from ras_docproc.schema import BBox


def bbox_overlap(a: BBox, b: BBox) -> float:
    """Compute IoU (Intersection over Union) of two bounding boxes."""
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)

    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0

    intersection = (ix1 - ix0) * (iy1 - iy0)
    union = a.area + b.area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def bbox_contains(outer: BBox, inner: BBox) -> bool:
    """Check if outer bbox fully contains inner bbox."""
    return outer.x0 <= inner.x0 and outer.y0 <= inner.y0 and outer.x1 >= inner.x1 and outer.y1 >= inner.y1


def is_in_zone(bbox: BBox, page_height: float, zone_top_frac: float, zone_bottom_frac: float) -> bool:
    """Check if a bbox's vertical center falls within a zone defined by fractions of page height."""
    center_y = (bbox.y0 + bbox.y1) / 2
    zone_top = page_height * zone_top_frac
    zone_bottom = page_height * zone_bottom_frac
    return zone_top <= center_y <= zone_bottom


def vertical_distance(a: BBox, b: BBox) -> float:
    """Compute vertical distance between bottom of a and top of b."""
    return b.y0 - a.y1


def bbox_to_pixel_coords(bbox: BBox, page_rect_width: float, page_rect_height: float, pixmap_width: int, pixmap_height: int) -> BBox:
    """Convert PDF coordinate bbox to pixel coordinates for a rendered pixmap."""
    scale_x = pixmap_width / page_rect_width
    scale_y = pixmap_height / page_rect_height
    return BBox(
        x0=bbox.x0 * scale_x,
        y0=bbox.y0 * scale_y,
        x1=bbox.x1 * scale_x,
        y1=bbox.y1 * scale_y,
    )
