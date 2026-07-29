"""抓取顺序可视化与优先级重排工具。

把 _draw_priority_overlay 与 rank_detections_by_priority 从
box_object_pointcloud_sam_completion_template_icp_with_yolo2 抽出，便于复用。
rank_detections_by_priority 内部延迟导入 box_object 模块中的依赖，
以避免与本模块形成循环依赖。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def _draw_priority_overlay(image_bgr, detections):
    """在图像副本上标注优先级抓取顺序：rank 编号（中心圆）+ OBB 多边形 + 分数 + 遮挡标签。

    返回标注后的 BGR 图（不修改原图）。rank=1 用红色，之后渐变到绿色，
    直观表达"先抓谁"。遮挡标签：红字"遮挡"(class_id=0) / 绿字"无遮挡"(class_id=1)。
    供 rank_detections_by_priority(show=True) 调用。
    """
    vis = image_bgr.copy()
    total = max(len(detections), 1)
    for det in detections:
        rank = int(det.get("priority_rank", 0))
        score = float(det.get("priority_score", 0.0))
        bbox = det.get("bbox")
        corners = det.get("corners_px")
        # 颜色：rank 1 红(0) -> 末位 绿(约120)
        hue = 0 if total <= 1 else int(120.0 * (rank - 1) / max(total - 1, 1))
        hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        color = tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
        # OBB 多边形
        if corners is not None:
            pts = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
        # 中心点 / 分数锚点（仅用 bbox 求中心与锚点，不再画外接水平框）
        if bbox is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            score_anchor = (x1, max(y1 - 8, 12))
        elif corners is not None:
            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]
            cx, cy = int(round(sum(xs) / len(xs))), int(round(sum(ys) / len(ys)))
            score_anchor = (int(round(min(xs))), max(int(round(min(ys))) - 8, 12))
        else:
            cx, cy = 0, 0
            score_anchor = (12, 12)
        # 中心 rank 编号（填充圆 + 白边 + 白字）+ 遮挡标签（写在序号下方、框中心）
        if cx or cy:
            r = 22
            cv2.circle(vis, (cx, cy), r, color, thickness=-1)
            cv2.circle(vis, (cx, cy), r, (255, 255, 255), thickness=2)
            label = str(rank)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 1.0, 3)
            cv2.putText(vis, label, (cx - tw // 2, cy + th // 2), font, 1.0,
                        (255, 255, 255), 3, cv2.LINE_AA)
            # 遮挡标签：class_id 0=遮挡(红字) / 1=无遮挡(绿字)，居中写在序号下方
            occluded = int(det.get("class_id", 1)) == 0
            occ_label = "遮挡" if occluded else "无遮挡"
            occ_color = (0, 0, 255) if occluded else (0, 200, 0)
            (ow, oh), _ = cv2.getTextSize(occ_label, font, 0.6, 2)
            cv2.putText(vis, occ_label, (cx - ow // 2, cy + r + oh + 4),
                        font, 0.6, occ_color, 2, cv2.LINE_AA)
        # 分数文本
        cv2.putText(vis, f"{score:.3f}", score_anchor, cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA)
    # 顶部标题
    cv2.putText(vis, "Priority grasp order (red = 1st)", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    return vis


def rank_detections_by_priority(image_bgr, depth_mm, detections, args, show):
    """用优先级网络对 YOLO 检测候选重排，返回按优先级降序的新 detections 列表。

    每个 detection 会附带 priority_score / priority_rank 字段。
    若未启用、无深度图或候选<=1，则原样返回（不改顺序）。
    """
    # 延迟导入，避免与 box_object 模块形成循环依赖
    from yanjiuyuan.box_object_pointcloud_sam_completion_template_icp_with_yolo2 import (
        log,
        _load_priority_model_cached,
        _detection_to_priority_object,
        _build_priority_batch,
    )
    if not getattr(args, "priority_order", True):
        return detections
    if depth_mm is None:
        log("[box_object] 无深度图，跳过优先级排序，保留原（面积）顺序")
        return detections
    if len(detections) <= 1:
        return detections

    try:
        model, config, device, helpers, prototype = _load_priority_model_cached(args)
    except Exception as exc:  # noqa: BLE001
        log(f"[box_object] 优先级模型加载失败，回退到面积顺序: {exc}")
        return detections

    image_size = int(config["image_size"])
    depth_near_mm = float(config["depth_near_mm"])
    depth_far_mm = float(config["depth_far_mm"])
    mode = config["mode"]

    objects = [_detection_to_priority_object(d, i + 1) for i, d in enumerate(detections)]
    try:
        batch = _build_priority_batch(image_bgr, depth_mm, objects, image_size,
                                      depth_near_mm, depth_far_mm, helpers)
        batch = helpers["move_batch"](batch, device)
        with torch.inference_mode():
            output = model(batch)
            if mode == "full_ranking":
                scores = output["scores"].detach().float().cpu().numpy().reshape(-1).tolist()
            else:
                if prototype is None:
                    log("[box_object] top1_only 模式缺少 prototype，回退到面积顺序")
                    return detections
                emb = output["embeddings"]
                emb = emb / emb.norm(dim=1, keepdim=True)
                scores = (emb @ prototype).detach().float().cpu().numpy().reshape(-1).tolist()
    except Exception as exc:  # noqa: BLE001
        log(f"[box_object] 优先级推理失败，回退到面积顺序: {exc}")
        return detections

    paired = sorted(zip(detections, scores), key=lambda x: -x[1])
    for rank, (det, score) in enumerate(paired, start=1):
        det["priority_score"] = float(score)
        det["priority_rank"] = rank
    reordered = [det for det, _ in paired]
    log("[box_object] 优先级排序完成（%d 个候选），按优先级降序: %s" % (
        len(reordered),
        ", ".join(f"#{d['priority_rank']} score={d['priority_score']:.3f}" for d in reordered),
    ))
    if show:
        try:
            vis = _draw_priority_overlay(image_bgr, reordered)
            # 最佳努力保存到 RGB 同级目录 / output_dir（真实相机流 args.image 可能为 None）
            _save_dir = None
            _rgb = getattr(args, "image", None)
            if _rgb is not None:
                _save_dir = Path(_rgb).parent
            elif getattr(args, "output_dir", None) is not None:
                _save_dir = Path(args.output_dir)
            if _save_dir is not None:
                try:
                    _save_dir.mkdir(parents=True, exist_ok=True)
                    _sp = _save_dir / "priority_order_vis.png"
                    cv2.imwrite(str(_sp), vis)
                    log(f"[box_object] 优先级顺序可视化已保存: {_sp}")
                except Exception as exc:  # noqa: BLE001
                    log(f"[box_object] 优先级顺序可视化保存失败: {exc}")
            # 弹窗显示（headless 环境会抛错，已捕获；窗口停留 3 秒后自动关闭，不阻塞流水线）
            try:
                cv2.imshow("Priority Order (YOLO detections)", vis)
                log("[box_object] 优先级顺序可视化：窗口停留 3 秒或按任意键关闭...")
                cv2.waitKey(0)
            except Exception as exc:  # noqa: BLE001
                log(f"[box_object] 无法弹窗显示（headless 环境？），已保存 PNG: {exc}")
            finally:
                try:
                    cv2.destroyWindow("Priority Order (YOLO detections)")
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001
            log(f"[box_object] 优先级顺序可视化失败: {exc}")

    return reordered
