"""Tests of cora.data.transforms: SAM / CLIP preprocessing and the anchor crop / box (Sec. 3.1, 3.3; supplement Sec. 5)."""
import cv2
import numpy as np
import pytest
import torch

from cora.data import transforms as T


def gradient_image(h, w, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# ----------------------------------------------------------------------------------------------
# SAM view
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("h,w", [(100, 400), (400, 100), (256, 256), (37, 91)])
def test_sam_preprocess_shape_and_padding(h, w):
    image = gradient_image(h, w)
    x, (rh, rw) = T.sam_preprocess(image, 128)
    assert x.shape == (3, 128, 128) and x.dtype == torch.float32
    assert max(rh, rw) == 128
    assert (rh, rw) == T.sam_resize_shape(h, w, 128)
    # padding is zero at the bottom and right, the content region is not all zero
    assert torch.all(x[:, rh:, :] == 0) and torch.all(x[:, :, rw:] == 0)
    assert x[:, :rh, :rw].abs().sum() > 0
    # normalised range of 0..255 pixels
    assert x.min() >= (0 - 123.675) / 58.395 - 1e-4 and x.max() <= (255 - 103.53) / 57.375 + 1e-4


def test_sam_preprocess_matches_segment_anything():
    transforms = pytest.importorskip("segment_anything.utils.transforms")
    image = gradient_image(120, 333, seed=3)
    x, (rh, rw) = T.sam_preprocess(image, 256)
    expected = transforms.ResizeLongestSide(256).apply_image(image)
    assert expected.shape[:2] == (rh, rw)
    mean = torch.tensor(T.SAM_PIXEL_MEAN).view(3, 1, 1)
    std = torch.tensor(T.SAM_PIXEL_STD).view(3, 1, 1)
    ref = (torch.from_numpy(expected.copy()).permute(2, 0, 1).float() - mean) / std
    assert torch.allclose(x[:, :rh, :rw], ref, atol=1e-5)


def test_sam_preprocess_rejects_bad_input():
    with pytest.raises(ValueError, match="uint8"):
        T.sam_preprocess(np.zeros((8, 8, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        T.sam_preprocess(np.zeros((8, 8), dtype=np.uint8))


# ----------------------------------------------------------------------------------------------
# CLIP view
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", T.CLIP_MODES)
@pytest.mark.parametrize("h,w", [(100, 200), (200, 100), (64, 64)])
def test_clip_preprocess_shape(mode, h, w):
    x = T.clip_preprocess(gradient_image(h, w), 56, mode)
    assert x.shape == (3, 56, 56) and x.dtype == torch.float32 and torch.isfinite(x).all()


def test_clip_pad_mode_pads_centred_with_mean_colour():
    image = np.full((40, 120, 3), 255, dtype=np.uint8)  # wide, white
    x = T.clip_preprocess(image, 120, "pad")
    mean_colour = torch.tensor([int(c * 255) for c in T.CLIP_MEAN], dtype=torch.float32) / 255.0
    expected = ((mean_colour - torch.tensor(T.CLIP_MEAN)) / torch.tensor(T.CLIP_STD)).view(3, 1, 1)
    top, bottom = x[:, :40, :], x[:, 80:, :]  # 40 rows of padding above and below the 40-row image
    assert torch.allclose(top, expected.expand_as(top), atol=1e-5)
    assert torch.allclose(bottom, expected.expand_as(bottom), atol=1e-5)
    white = ((1.0 - torch.tensor(T.CLIP_MEAN)) / torch.tensor(T.CLIP_STD)).view(3, 1, 1)
    middle = x[:, 45:75, :]
    assert torch.allclose(middle, white.expand_as(middle), atol=1e-4)


def test_clip_pad_mode_tall_image_pads_left_and_right():
    image = np.full((120, 40, 3), 255, dtype=np.uint8)
    x = T.clip_preprocess(image, 120, "pad")
    white = ((1.0 - torch.tensor(T.CLIP_MEAN)) / torch.tensor(T.CLIP_STD)).view(3, 1, 1)
    centre = x[:, :, 45:75]
    assert torch.allclose(centre, white.expand_as(centre), atol=1e-4)
    assert not torch.allclose(x[:, :, :35], white.expand_as(x[:, :, :35]), atol=1e-2)


def test_clip_center_crop_drops_the_sides_and_squash_keeps_them():
    image = np.zeros((50, 150, 3), dtype=np.uint8)
    image[:, :40] = 255  # white strip on the far left, outside the central square
    cropped = T.clip_preprocess(image, 50, "center_crop")
    squashed = T.clip_preprocess(image, 50, "squash")
    black = ((0.0 - torch.tensor(T.CLIP_MEAN)) / torch.tensor(T.CLIP_STD)).view(3, 1, 1)
    assert torch.allclose(cropped, black.expand_as(cropped), atol=1e-4)
    assert squashed.max() > 1.5  # the white strip survives


def test_clip_preprocess_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown CLIP"):
        T.clip_preprocess(gradient_image(8, 8), 8, "stretch")


def test_clip_center_crop_matches_hf_processor():
    from PIL import Image

    try:
        from transformers import CLIPImageProcessor

        processor = CLIPImageProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    except OSError:
        pytest.skip("LLaVA image processor not in the local Hugging Face cache")
    image = gradient_image(150, 300, seed=5)
    image = cv2.GaussianBlur(image, (0, 0), 3)  # smooth content, so resampling kernels agree closely
    reference = processor(Image.fromarray(image), return_tensors="pt")["pixel_values"][0]
    ours = T.clip_preprocess(image, 336, "center_crop")
    assert reference.shape == ours.shape
    assert (reference - ours).abs().mean() < 0.02


# ----------------------------------------------------------------------------------------------
# Anchor crop and box
# ----------------------------------------------------------------------------------------------
def test_anchor_crop_zeroes_outside_crops_and_pads_top_left():
    image = gradient_image(60, 80, seed=1) | 1  # no zero pixels inside the mask
    mask = np.zeros((60, 80), dtype=bool)
    mask[10:20, 30:60] = True  # 10 x 30 box
    mask[12, 35] = False  # a hole inside the box
    crop = T.anchor_crop(image, mask)
    assert crop.shape == (30, 30, 3) and crop.dtype == np.uint8
    assert np.array_equal(crop[:10, :30][mask[10:20, 30:60]], image[10:20, 30:60][mask[10:20, 30:60]])
    assert np.all(crop[2, 5] == 0)  # the hole (row 12, col 35) is zero
    assert np.all(crop[10:] == 0)  # zero padding at the bottom
    mask_tall = np.zeros((60, 80), dtype=bool)
    mask_tall[5:35, 10:15] = True
    tall = T.anchor_crop(image, mask_tall)
    assert tall.shape == (30, 30, 3) and np.all(tall[:, 5:] == 0)  # padding at the right
    assert np.array_equal(tall[:, :5], image[5:35, 10:15])


def test_anchor_crop_errors():
    image = gradient_image(10, 10)
    with pytest.raises(ValueError, match="empty"):
        T.anchor_crop(image, np.zeros((10, 10), dtype=bool))
    with pytest.raises(ValueError, match="differs"):
        T.anchor_crop(image, np.ones((9, 10), dtype=bool))


def test_anchor_box_format_and_round_trip():
    mask = np.zeros((512, 1024), dtype=bool)
    mask[100:180, 300:700] = True  # rows 100..179, cols 300..699
    box = T.anchor_box(mask, 1024)  # scale 1.0
    assert box.shape == (4,) and box.dtype == torch.float32
    y0, x0, y1, x1 = (box * 1024).round().long().tolist()
    assert (y0, x0, y1, x1) == (100, 300, 179, 699)  # [y0, x0, y1, x1], inclusive, divided by size


@pytest.mark.parametrize("h,w", [(1024, 2048), (500, 333), (300, 1200)])
def test_anchor_box_round_trip_through_sam_frame(h, w):
    rng = np.random.default_rng(h + w)
    mask = np.zeros((h, w), dtype=bool)
    y0, x0 = int(rng.integers(0, h // 2)), int(rng.integers(0, w // 2))
    y1, x1 = y0 + int(rng.integers(5, h // 3)), x0 + int(rng.integers(5, w // 3))
    mask[y0 : y1 + 1, x0 : x1 + 1] = True
    box = T.anchor_box(mask, 1024)
    scale = 1024 / max(h, w)
    ry0, rx0, ry1, rx1 = (box * 1024).tolist()
    # undoing the SAM resize recovers the original extent to within one source pixel at either end
    assert abs(ry0 / scale - y0) <= 1 and abs(rx0 / scale - x0) <= 1
    assert abs((ry1 + 1) / scale - (y1 + 1)) <= 1 and abs((rx1 + 1) / scale - (x1 + 1)) <= 1
    assert 0 <= box.min() and box.max() < 1.0
    # the box covers exactly the extent of the SAM-resized mask
    new_h, new_w = T.sam_resize_shape(h, w, 1024)
    resized = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_AREA) > 0
    rows, cols = np.flatnonzero(resized.any(1)), np.flatnonzero(resized.any(0))
    assert abs(ry0 - rows[0]) <= 1 and abs(ry1 - rows[-1]) <= 1 and abs(rx0 - cols[0]) <= 1 and abs(rx1 - cols[-1]) <= 1


def test_anchor_box_thin_mask_survives_downscale():
    mask = np.zeros((1024, 2048), dtype=bool)
    mask[500, 700] = True  # a single pixel
    box = T.anchor_box(mask, 1024)
    y0, x0, y1, x1 = (box * 1024).tolist()
    assert y0 <= y1 and x0 <= x1 and (y0, x0) == (250, 350)


def test_anchor_box_empty_mask_raises():
    with pytest.raises(ValueError, match="empty"):
        T.anchor_box(np.zeros((8, 8), dtype=bool))


# ----------------------------------------------------------------------------------------------
# Real image
# ----------------------------------------------------------------------------------------------
@pytest.mark.data
def test_real_cityscapes_image(cityscapes_root):
    from cora.data.cityscapes import CityscapesAdapter

    adapter = CityscapesAdapter(cityscapes_root)
    image_id = adapter.partition("1_30", "labeled")[0]
    image, label = adapter.load_image(image_id), adapter.load_label(image_id)
    sam, (h, w) = T.sam_preprocess(image, 1024)
    assert image.shape == (1024, 2048, 3) and (h, w) == (512, 1024) and sam.shape == (3, 1024, 1024)
    assert torch.all(sam[:, 512:] == 0)
    for mode in T.CLIP_MODES:
        assert T.clip_preprocess(image, 336, mode).shape == (3, 336, 336)
    car = label == 13
    if not car.any():
        car = label == label[label != 255][0]
    crop = T.anchor_crop(image, car)
    assert crop.shape[0] == crop.shape[1] and crop.max() > 0
    box = T.anchor_box(car, 1024)
    assert 0 <= box.min() and box.max() < 1 and box[2] >= box[0] and box[3] >= box[1]
