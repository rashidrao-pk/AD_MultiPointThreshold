import os
import json
import time
import argparse
from collections import deque, OrderedDict

import cv2
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import torch
import torchvision.transforms as transforms

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image as CompressedImage
from cv_bridge import CvBridge

from distrimuse_ros2_api.msg import RulexAreaScore, RulexDetectionResult

import utils as ut
import utils.scripts.utils_model as utmc
from utils.scripts.utils_model import Encoder, Decoder, Discriminator


ALL_SAFETY_AREAS = ["PLeft", "PRight", "RoboArm", "ConvBelt"]

THRESHOLD_CMAP_UNEXPECTED = LinearSegmentedColormap.from_list(
    "custom_threshold_cmap",
    [
        (0.0, "white"),
        (0.25, "lightblue"),
        (0.35, "coral"),
        (0.50, "red"),
        (1.0, "purple"),
    ],
)

AREA_DISPLAY_NAMES = {
    "PLeft": "Pallet Left",
    "PRight": "Pallet Right",
    "RoboArm": "Robo Arm",
    "ConvBelt": "Conveyor Belt",
}

AREA_NAME_TO_ENUM = {
    "RoboArm": RulexAreaScore.AREA_A,
    "ConvBelt": RulexAreaScore.AREA_B,
    "PLeft": RulexAreaScore.AREA_C,
    "PRight": RulexAreaScore.AREA_D,
}


def ordered_area_list(areas):
    order_map = {name: i for i, name in enumerate(ALL_SAFETY_AREAS)}
    return sorted(list(areas), key=lambda x: order_map.get(x, 999))


def create_union_mask(area_inputs, frame_shape_hw):
    h, w = frame_shape_hw
    union_mask = np.zeros((h, w), dtype=np.uint8)

    for area_name in ordered_area_list(area_inputs.keys()):
        info = area_inputs[area_name]
        mask_bin = info.get("mask_bin")
        if mask_bin is None:
            continue
        if mask_bin.shape[:2] != (h, w):
            mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
        union_mask = np.maximum(union_mask, mask_bin)

    return union_mask


def overlay_outside_safety_blur(frame_bgr, area_inputs, blur_ksize=31, darken_factor=0.35):
    if len(area_inputs) == 0:
        return frame_bgr.copy()

    union_mask = create_union_mask(area_inputs, frame_bgr.shape[:2])

    blurred = cv2.GaussianBlur(frame_bgr, (blur_ksize, blur_ksize), 0)
    darkened = (blurred.astype(np.float32) * darken_factor).clip(0, 255).astype(np.uint8)

    union_mask_3 = cv2.cvtColor(union_mask, cv2.COLOR_GRAY2BGR)
    out = np.where(union_mask_3 > 0, frame_bgr, darkened)
    return out


def resize_and_center(image, target_w, target_h, bg_color=(0, 0, 0)):
    if image is None:
        return np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def scale_contours(contours, scale, x_off, y_off):
    scaled = []
    for cnt in contours:
        cnt_scaled = cnt.astype(np.float32).copy()
        cnt_scaled[:, 0, 0] = x_off + cnt_scaled[:, 0, 0] * scale
        cnt_scaled[:, 0, 1] = y_off + cnt_scaled[:, 0, 1] * scale
        scaled.append(cnt_scaled.astype(np.int32))
    return scaled


def colorize_anomaly_map(dist_map, vmin=0.0, vmax=2.0):
    if dist_map is None:
        return None

    dm = dist_map.astype(np.float32)
    denom = max(float(vmax) - float(vmin), 1e-6)
    dm = np.clip((dm - float(vmin)) / denom, 0.0, 1.0)

    rgb = (THRESHOLD_CMAP_UNEXPECTED(dm)[..., :3] * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def unletterbox_patch(patch_bgr, resize_meta):
    if patch_bgr is None or resize_meta is None:
        return patch_bgr

    x_off = int(resize_meta.get("x_off", 0))
    y_off = int(resize_meta.get("y_off", 0))
    new_w = int(resize_meta.get("new_w", patch_bgr.shape[1]))
    new_h = int(resize_meta.get("new_h", patch_bgr.shape[0]))

    if new_w <= 0 or new_h <= 0:
        return patch_bgr

    h, w = patch_bgr.shape[:2]
    x1 = max(0, x_off)
    y1 = max(0, y_off)
    x2 = min(w, x_off + new_w)
    y2 = min(h, y_off + new_h)

    if x2 <= x1 or y2 <= y1:
        return patch_bgr

    cropped = patch_bgr[y1:y2, x1:x2]
    if cropped.size == 0:
        return patch_bgr
    return cropped


def paste_area_result_in_full_frame(
    target_canvas,
    patch_bgr,
    bbox,
    mask_bin,
    resize_meta=None,
    keep_background=False,
    background_canvas=None,
):
    if patch_bgr is None or bbox is None or mask_bin is None:
        return target_canvas

    x1, y1, x2, y2 = bbox
    crop_w = x2 - x1 + 1
    crop_h = y2 - y1 + 1

    if crop_w <= 0 or crop_h <= 0:
        return target_canvas

    if resize_meta is not None:
        patch_bgr = unletterbox_patch(patch_bgr, resize_meta)

    if patch_bgr is None or patch_bgr.size == 0:
        return target_canvas

    patch_resized = cv2.resize(patch_bgr, (crop_w, crop_h), interpolation=cv2.INTER_AREA)
    mask_crop = mask_bin[y1:y2 + 1, x1:x2 + 1]
    mask_crop_3 = cv2.cvtColor(mask_crop, cv2.COLOR_GRAY2BGR)

    roi = target_canvas[y1:y2 + 1, x1:x2 + 1]

    if keep_background and background_canvas is not None:
        bg_roi = background_canvas[y1:y2 + 1, x1:x2 + 1]
        blended = np.where(mask_crop_3 > 0, patch_resized, bg_roi)
    else:
        blended = np.where(mask_crop_3 > 0, patch_resized, roi)

    target_canvas[y1:y2 + 1, x1:x2 + 1] = blended
    return target_canvas


def draw_text_table(panel, results, frame_id=None, corr_frame_id=None, corr_stamp=None):
    panel[:] = (245, 245, 245)
    
    title_y = 35
    cv2.putText(panel, "Details", (panel.shape[1] // 2 - 50, title_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)

    y = 70
    cv2.line(panel, (20, y), (panel.shape[1] - 20, y), (40, 40, 40), 2)
    y += 35

    if frame_id is not None:
        if corr_stamp is not None:
            txt = f"Frame: {frame_id} CFID: {corr_frame_id} @ {corr_stamp.sec}.{corr_stamp.nanosec}"
        else:
            txt = f"Frame: {frame_id}"
        cv2.putText(panel, txt, (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
        y += 20
        cv2.line(panel, (20, y), (panel.shape[1] - 20, y), (40, 40, 40), 1)
        y += 35

    headers = ["Safety Area", "Raw Score", "Threshold", "Norm Score", "Status"]
    col_x = [30, 240, 370, 510, 670]

    for i, hdr in enumerate(headers):
        cv2.putText(panel, hdr, (col_x[i], y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)

    y += 20
    cv2.line(panel, (20, y), (panel.shape[1] - 20, y), (40, 40, 40), 1)
    y += 35

    for area_name in ordered_area_list(results.keys()):
        r = results[area_name]
        raw_score = r.get("score", None)
        thr = r.get("threshold", None)
        norm = r.get("norm_score", None)
        status = r.get("status", "unknown")
        is_anom = bool(r.get("is_anomalous", False))

        color = (0, 0, 180) if is_anom else (0, 140, 0)
        vals = [
            AREA_DISPLAY_NAMES.get(area_name, area_name),
            "-" if raw_score is None else f"{raw_score:.3f}",
            "-" if thr is None else f"{thr:.3f}",
            "-" if norm is None else f"{norm:.3f}",
            status,
        ]

        for i, val in enumerate(vals):
            cv2.putText(panel, str(val), (col_x[i], y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color if i >= 3 else (30, 30, 30),
                        2 if i == 4 else 1,
                        cv2.LINE_AA)

        y += 20
        cv2.line(panel, (20, y), (panel.shape[1] - 20, y), (120, 120, 120), 1)
        y += 35

    return panel


def build_full_recon_and_anom(frame_bgr, area_inputs):
    recon_full = np.zeros_like(frame_bgr)
    anom_full = np.full_like(frame_bgr, 255)

    for area_name in ordered_area_list(area_inputs.keys()):
        info = area_inputs[area_name]
        bbox = info.get("bbox")
        mask_bin = info.get("mask_bin")
        resize_meta = info.get("resize_meta")
        recon_patch = info.get("recon_patch_bgr")
        anom_patch = info.get("anom_patch_bgr")

        recon_full = paste_area_result_in_full_frame(
            recon_full, recon_patch, bbox, mask_bin, resize_meta=resize_meta
        )
        anom_full = paste_area_result_in_full_frame(
            anom_full, anom_patch, bbox, mask_bin, resize_meta=resize_meta
        )

    return recon_full, anom_full


def draw_dashboard_panel(frame_bgr, area_inputs, latest_results, frame_id=None,
                         width=1600, height=1000, corr_frame_id=None, corr_stamp=None):
    canvas = np.full((height, width, 3), 235, dtype=np.uint8)

    pad = 16
    panel_w = (width - 3 * pad) // 2
    panel_h = (height - 3 * pad) // 2

    tl = (pad, pad, pad + panel_w, pad + panel_h)
    tr = (2 * pad + panel_w, pad, width - pad, pad + panel_h)
    bl = (pad, 2 * pad + panel_h, pad + panel_w, height - pad)
    br = (2 * pad + panel_w, 2 * pad + panel_h, width - pad, height - pad)

    def draw_panel_title(title, box):
        x1, y1, x2, y2 = box
        cv2.putText(canvas, title, (x1 + 12, y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 20, 20), 1)

    draw_panel_title("Input Image with detections", tl)
    draw_panel_title("Anomaly Map", tr)
    draw_panel_title("Reconstructed Image", bl)
    draw_panel_title("Details", br)

    inner_margin = 12
    title_h = 40

    def inner_box(box):
        x1, y1, x2, y2 = box
        return (x1 + inner_margin, y1 + title_h, x2 - inner_margin, y2 - inner_margin)

    tl_in = inner_box(tl)
    tr_in = inner_box(tr)
    bl_in = inner_box(bl)
    br_in = inner_box(br)

    h, w = frame_bgr.shape[:2]

    input_vis = overlay_outside_safety_blur(frame_bgr, area_inputs)
    input_full = input_vis.copy()

    for area_name in ordered_area_list(area_inputs.keys()):
        info = area_inputs[area_name]
        input_full = paste_area_result_in_full_frame(
            input_full,
            info.get("orig_patch_bgr"),
            info.get("bbox"),
            info.get("mask_bin"),
            resize_meta=info.get("resize_meta"),
            keep_background=True,
            background_canvas=input_vis,
        )

    tl_w = tl_in[2] - tl_in[0]
    tl_h = tl_in[3] - tl_in[1]
    scale = min(tl_w / w, tl_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    tl_img = cv2.resize(input_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x_off = tl_in[0] + (tl_w - new_w) // 2
    y_off = tl_in[1] + (tl_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = tl_img

    for area_name in ordered_area_list(area_inputs.keys()):
        info = area_inputs[area_name]
        rr = latest_results.get(area_name, {})
        color = (0, 0, 255) if bool(rr.get("is_anomalous", False)) else (255, 255, 255)
        scaled = scale_contours(info.get("contours", []), scale, x_off, y_off)
        if len(scaled) > 0:
            cv2.drawContours(canvas, scaled, -1, color, 2)
            pt = scaled[0][0][0]
            label = f"{AREA_DISPLAY_NAMES.get(area_name, area_name)}: {rr.get('norm_score', 0):.2f}" if "norm_score" in rr else area_name
            cv2.putText(canvas, label, (int(pt[0]), max(20, int(pt[1]) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    recon_full, anom_full = build_full_recon_and_anom(frame_bgr, area_inputs)

    tr_w = tr_in[2] - tr_in[0]
    tr_h = tr_in[3] - tr_in[1]
    anom_disp = resize_and_center(anom_full, tr_w, tr_h, bg_color=(255, 255, 255))
    canvas[tr_in[1]:tr_in[1] + tr_h, tr_in[0]:tr_in[0] + tr_w] = anom_disp

    scale_tr = min(tr_w / w, tr_h / h)
    new_w_tr = max(1, int(w * scale_tr))
    new_h_tr = max(1, int(h * scale_tr))
    x_off_tr = tr_in[0] + (tr_w - new_w_tr) // 2
    y_off_tr = tr_in[1] + (tr_h - new_h_tr) // 2

    for area_name in ordered_area_list(area_inputs.keys()):
        info = area_inputs[area_name]
        rr = latest_results.get(area_name, {})
        color = (0, 0, 255) if bool(rr.get("is_anomalous", False)) else (0, 128, 0)
        scaled = scale_contours(info.get("contours", []), scale_tr, x_off_tr, y_off_tr)
        if len(scaled) > 0:
            cv2.drawContours(canvas, scaled, -1, color, 2)
            pt = scaled[0][0][0]
            label = f"{rr.get('status', '')}: {rr.get('norm_score', 0):.2f}" if "norm_score" in rr else area_name
            cv2.putText(canvas, label, (int(pt[0]), int(pt[1]) + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    bl_w = bl_in[2] - bl_in[0]
    bl_h = bl_in[3] - bl_in[1]
    recon_disp = resize_and_center(recon_full, bl_w, bl_h, bg_color=(0, 0, 0))
    canvas[bl_in[1]:bl_in[1] + bl_h, bl_in[0]:bl_in[0] + bl_w] = recon_disp

    scale_bl = min(bl_w / w, bl_h / h)
    new_w_bl = max(1, int(w * scale_bl))
    new_h_bl = max(1, int(h * scale_bl))
    x_off_bl = bl_in[0] + (bl_w - new_w_bl) // 2
    y_off_bl = bl_in[1] + (bl_h - new_h_bl) // 2

    for area_name in ordered_area_list(area_inputs.keys()):
        info = area_inputs[area_name]
        rr = latest_results.get(area_name, {})
        color = (0, 0, 255) if bool(rr.get("is_anomalous", False)) else (0, 180, 0)
        scaled = scale_contours(info.get("contours", []), scale_bl, x_off_bl, y_off_bl)
        if len(scaled) > 0:
            cv2.drawContours(canvas, scaled, -1, color, 2)

    details_panel = np.full((br_in[3] - br_in[1], br_in[2] - br_in[0], 3), 245, dtype=np.uint8)
    details_panel = draw_text_table(details_panel, latest_results, frame_id=frame_id,
                                    corr_frame_id=corr_frame_id, corr_stamp=corr_stamp)
    canvas[br_in[1]:br_in[3], br_in[0]:br_in[2]] = details_panel

    return canvas, recon_full, anom_full


def _ensure_gray(mask):
    if mask is None:
        return None
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def _prepare_binary_mask(mask, frame_shape_hw):
    h, w = frame_shape_hw
    mask = _ensure_gray(mask)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask_bin


def _extract_mask_contours(mask_gray, frame_shape_hw):
    mask_bin = _prepare_binary_mask(mask_gray, frame_shape_hw)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask_bin


def _crop_with_mask(frame, mask_gray):
    mask_bin = _prepare_binary_mask(mask_gray, frame.shape[:2])
    masked_full = cv2.bitwise_and(frame, frame, mask=mask_bin)

    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, None, masked_full

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cropped = masked_full[y_min:y_max + 1, x_min:x_max + 1]
    bbox = (x_min, y_min, x_max, y_max)
    return cropped, bbox, masked_full


def _resize_128(image, keep_aspect=True, target=(128, 128), return_meta=False):
    target_w, target_h = target

    if image is None:
        return (None, None) if return_meta else None

    if not keep_aspect:
        out = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        meta = {
            "new_w": target_w, "new_h": target_h, "x_off": 0, "y_off": 0,
            "target_w": target_w, "target_h": target_h,
            "orig_h": image.shape[0], "orig_w": image.shape[1],
        }
        return (out, meta) if return_meta else out

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return (None, None) if return_meta else None

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    meta = {
        "new_w": new_w, "new_h": new_h, "x_off": x_off, "y_off": y_off,
        "target_w": target_w, "target_h": target_h, "orig_h": h, "orig_w": w,
    }
    return (canvas, meta) if return_meta else canvas


def tensor_to_hwc_float32(t: torch.Tensor) -> np.ndarray:
    return (t.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32) * 0.5 + 0.5)


def _compute_distance_offset_np(imgA: np.ndarray, imgB: np.ndarray, offset: int) -> np.ndarray:
    H, W, _ = imgA.shape
    dist = np.full((H, W), np.inf, dtype=np.float32)
    for di in range(-offset, offset + 1):
        for dj in range(-offset, offset + 1):
            i0a = max(0, di)
            i1a = min(H, H + di)
            i0b = max(0, -di)
            i1b = min(H, H - di)

            j0a = max(0, dj)
            j1a = min(W, W + dj)
            j0b = max(0, -dj)
            j1b = min(W, W - dj)

            diff = imgA[i0a:i1a, j0a:j1a] - imgB[i0b:i1b, j0b:j1b]
            d = np.sqrt((diff ** 2).sum(axis=2)).astype(np.float32)
            dist[i0a:i1a, j0a:j1a] = np.minimum(dist[i0a:i1a, j0a:j1a], d)
    return dist


def compute_anomaly_score_pair(imgA, imgB, offset=2, quantile=0.995):
    dist = _compute_distance_offset_np(imgA, imgB, offset)
    return float(np.quantile(dist, quantile)), dist


def load_threshold(threshold_dir: str, safety_area: str) -> float:
    json_path = os.path.join(threshold_dir, safety_area, f"threshold_{safety_area}.json")
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            d = json.load(f)
        return float(d["threshold"])
    raise FileNotFoundError(f"Threshold file not found: {json_path}")


def build_suffix_for_area(area, args):
    class P:
        pass

    params = P()
    params.subgroup = area
    params.latent_dims = args.latent_dims
    params.z_dim = args.latent_dims
    params.dataset_type = args.dataset_source_name
    params.subgroup_mask = args.subgroup_mask
    params.target_size = (128, 128)

    if not hasattr(args, "save_figures"):
        args.save_figures = False
    if not hasattr(args, "train"):
        args.train = False
    if not hasattr(args, "test"):
        args.test = False
    if not hasattr(args, "inference"):
        args.inference = True

    paths = P()
    cwd = os.getcwd()
    paths.path_codes = cwd
    paths.path_codes_local = cwd
    paths.path_results_local = cwd
    paths.path_results_cloud = cwd
    paths.path_models = os.path.join(cwd, args.checkpoints)

    suffix, _ = ut.get_create_results_path(
        area, params, args, paths,
        save_path_type=args.save_path_type,
        dir="scripts/results",
        verbose=False,
    )
    return suffix


def load_models_for_areas(args, device, checkpoint_root=None, log_fn=None):
    """Load encoder/decoder checkpoints and thresholds for one or more safety areas."""
    checkpoint_root = checkpoint_root or args.checkpoints
    area_names = args.area_names if str(args.safety_area).upper() == "ALL" else [args.safety_area]
    models = {}
    thresholds = {}

    for area in area_names:
        enc = Encoder(z_size=args.latent_dims).to(device)
        dec = Decoder(z_size=args.latent_dims).to(device)
        dis = Discriminator().to(device)

        optED, optD = utmc.get_optimizers(enc, dec, dis, verbose=False)
        suffix = build_suffix_for_area(area, args)

        if log_fn:
            log_fn(2, f"[model-load] area={area} suffix={suffix}")
            log_fn(2, f"[model-load] checkpoint_root={checkpoint_root}")

        history, config = utmc.load_model(
            enc, dec, dis, optED, optD,
            checkpoint_root, suffix, device=device, verbose=False,
            model_variant=args.model_variant,
        )

        if not history:
            raise RuntimeError(f"No checkpoint found for area={area}, suffix={suffix}")

        enc.eval()
        dec.eval()
        tau = load_threshold(args.threshold_dir, area)

        models[area] = {"encoder": enc, "decoder": dec, "suffix": suffix, "config": config}
        thresholds[area] = tau

        if log_fn:
            log_fn(1, f"[loaded] {area}: suffix={suffix}, tau={tau:.6f}")

    return models, thresholds


# Backwards-compatible alias for older notebook/code references.
load_models = load_models_for_areas



def parse_args():
    p = argparse.ArgumentParser("Live ROS anomaly inference v5")

    p.add_argument("--camera_topic", default="/camera/back_view/image_raw")
    p.add_argument("--rulex_topic", default="/rulex/data")
    p.add_argument("--publish_rulex", action="store_true")
    p.add_argument("--attach_image_on_anomaly", action="store_true")

    p.add_argument("--publish_anomaly_map", action="store_true")
    p.add_argument("--anomaly_map_topic", default="/rulex/anomaly_map")

    p.add_argument("--publish_dashboard", action="store_true")
    p.add_argument("--dashboard_topic", default="/rulex/dashboard")

    p.add_argument("--publish_timeline", action="store_true")
    p.add_argument("--timeline_topic", default="/rulex/timeline")

    p.add_argument("--dataset", default="Distrimuse_UniGra",
                   choices=["MVtec", "Robotics_Hazards", "Distrimuse_UniGra"],
                   help="Dataset to use for inference")
    p.add_argument("--safety_area", default="ALL")
    p.add_argument("--area_names", nargs="+", default=["PLeft", "PRight", "RoboArm", "ConvBelt"])
    p.add_argument("--static_mask_paths", nargs="+", required=True)

    p.add_argument("--threshold_dir", required=True)
    p.add_argument("--checkpoints", default="scripts/results/models")
    p.add_argument("--latent_dims", type=int, default=64)

    p.add_argument("--offset", type=int, default=2)
    p.add_argument("--quantile", type=float, default=0.995)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--dataset_source_name", default="SR")
    p.add_argument("--subgroup_mask", default="MASK")
    p.add_argument("--save_path_type", default="local")
    p.add_argument("--save_figures", action="store_true", default=False)

    p.add_argument("--verbose_level", type=int, default=2)
    p.add_argument("--log_every_n", type=int, default=1)
    p.add_argument("--process_period", type=float, default=0.05)

    p.add_argument("--show_timeline", action="store_true")
    p.add_argument("--timeline_history", type=int, default=200)
    p.add_argument("--timeline_width", type=int, default=1000)
    p.add_argument("--timeline_height", type=int, default=500)

    p.add_argument("--show_model_input", action="store_true")
    p.add_argument("--model_input_width", type=int, default=1600)
    p.add_argument("--model_input_height", type=int, default=1000)
    p.add_argument("--model_variant", default="old")
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = LiveRosAnomalyInfer(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopped by user.")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
