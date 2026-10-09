#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
photo2scan.py —— 拍摄的文档照片 → 扫描件（黑白 / 彩色）

一条流水线，7 步：
  1) 纸张定位：多策略色度/亮度分割 + 灰度梯度亚像素吸附 → 四角精定位
  2) 透视校正：四点透视变换到标准纸张尺寸（A4/Letter/A5 @ 指定 DPI）
  3) 光照归一化：只用纯纸面像素估计光照场并相除 → 均匀光线、无阴影
  4) 表格拉直：自动检出长直线（表格/表单外框），非线性网格变形把弯曲的边拉直
  5) 纸面提纯：纸面 → 纯白；大片淡灰打印底色（表单底纹）→ 按实测灰度还原
  6) 黑白化：2 倍超采样 Sauvola 局部阈值 → 降采样，得到抗锯齿的黑白稿
  7) 输出：灰度 PNG + 黑白 PDF + 彩色 PDF（多张照片自动合成多页 PDF）

用法示例：
  python photo2scan.py -i 照片.jpg -o out -n 转正述职报告
  python photo2scan.py -i "photos/*.jpg" -o out -n 合同 --mode bw
  python photo2scan.py -i a.jpg -o out --corners "120,80 1180,40 1345,1775 60,1810"

依赖：opencv-python(-headless)、numpy、pillow
"""
from __future__ import annotations

import argparse
import glob as _glob
import os
import sys

import numpy as np

try:
    import cv2
except Exception as _exc:  # pragma: no cover
    sys.exit(
        "[photo2scan] 缺少 opencv，请先装依赖：\n"
        "  /Users/CC/.workbuddy/binaries/python/envs/default/bin/pip install "
        "opencv-python-headless pillow numpy\n"
        f"原始错误：{_exc}"
    )

from PIL import Image, ImageOps

# ---------------------------------------------------------------- 常量
PAPER_MM = {"a4": (210.0, 297.0), "letter": (215.9, 279.4), "a5": (148.0, 210.0)}
LEVEL = 232.0          # 光照归一化后「纸面」的目标灰度（后续二值化的基准）
RULE_REF = 0.88        # 长直线检测的参考灰度 = RULE_REF * LEVEL
SS_DEFAULT = 2         # 二值化超采样倍数
K_DEFAULT = 0.18       # Sauvola k（越小笔画越粗、框线越连续）
                       # 实测：0.22 对「小字密排」的原稿会把笔画削断（墨量比 0.18 少 5%，
                       # 且字腔发空）；0.18 在 5 张测试图上均无碎点、笔画完整。
                       # 原稿字大笔画粗 / 底噪多 / 网点稿 → 上调 0.22~0.30；
                       # 小字细笔画仍偏断 → 下调 0.14~0.16。
WIN_DEFAULT = 61       # Sauvola 窗口（原图尺度，像素）
MIN_AREA_DEFAULT = 40  # SS 域内 < 该面积的碎点删除
MAX_HOLE_DEFAULT = 110 # SS 域内 <= 该面积的笔画空洞填充
INK_LEVEL = 130.0      # 光照归一化后「实墨」的灰度：灰度渲染时 (LEVEL-INK_LEVEL) 作为满墨跨度
                       # 取 130 而非更低值，是为了让手机拍的淡墨/细笔画也能压到接近纯黑
TONE_REF_SIGMA = 20.0              # 灰度渲染的「局部纸面基准」平滑尺度（跟住纸面慢变化，不跟笔画）
TONE_LO, TONE_HI = 0.16, 0.85      # 灰度渲染的软膝：覆盖率 <LO 一律纯白（吃掉纸面噪点），>HI 一律实黑
TONE_GAMMA = 0.85                  # 灰度渲染的对比曲线（<1 把笔画压深一点）
TONE_DENOISE = 3                   # 灰度渲染前的 medianBlur 核（去纸面/JPEG 噪点，保笔画）
TONE_AUTO_TEXT_MM = 2.8            # --tone auto 的判据：正文字高中位小于该物理尺寸（≈8pt）→ 用灰度
                                   # 等价于 200 DPI 下 22px；按 DPI 换算，换分辨率不改变判断结果

SHADE_LO, SHADE_HI = 0.75, 0.97    # 底纹判定：相对纸面亮度比区间（覆盖「淡灰」到「中灰」印刷底色）
SHADE_MIN_FRAC = 0.002             # 底纹连通域最小面积 / 纸张面积
SHADE_MAX_FRAC = 0.50              # 超过此占比判为光照残留，不做底纹还原
RULE_MIN_SPAN = 0.50               # 认定为表格框线所需的最小跨度（占页宽/页高）
                                   # 注意：受拍摄弯曲影响，一条真框线经方向性开运算后往往
                                   # 只剩 55%~90% 的连续响应，门限设 0.60 会把真框线误挡掉，
                                   # 真正区分「框线 vs 文字行」的是下面的强度门限。
RULE_GROUP_GAP = 0.004             # 行（列）分组的「断口容忍」= 该比例 × 页长边（最小 4 行）
                                   # 弯曲的粗框线在响应图上会断成上下两段（中间几行响应弱），
                                   # gap 太小 → 同一条线被拆成两段、每段只覆盖半幅宽 → span 腰斩被误挡。
RULE_MIN_STRENGTH = 12.0           # 认定为表格框线所需的最小平均响应（区分「真框线」与文字行）
RULE_EDGE_MARGIN = 0.02            # 距页面边缘这个比例以内的线视为纸张边缘渐晕/背景残留，不作框线
QUAD_MIN_EDGE_SUPPORT = 0.25       # 候选四边形每条边至少有这么大比例的采样点能对上真实图像边缘
                                   # （防止「阴影/亮度接近」导致的错误分割被当成纸）


# ================================================================ 基础
def log(*a):
    print(*a, flush=True)


# 输出分辨率相对 200 DPI 的比值。脚本里大量「绝对像素」参数（Sauvola 窗口、形态学核、
# 高斯尺度、碎点面积…）都是按 200 DPI 调的；换 --dpi 时必须一起缩放，否则同一张照片
# 在不同 DPI 下会得到不同的二值化结果（实测 300 DPI 量出的字高只有 200 DPI 的 0.79 倍）。
SCALE = 1.0


def px(v, lo=1):
    """按输出分辨率缩放一个「核尺寸」参数。返回奇数——OpenCV 的中值/形态学核要求奇数。"""
    n = max(int(lo), int(round(v * SCALE)))
    return n + 1 if n % 2 == 0 else n


def pxf(v, lo=0.5):
    """按输出分辨率缩放一个「高斯尺度」参数"""
    return max(lo, float(v) * SCALE)


def median_flat(a, k):
    """一维序列的中值平滑。

    `cv2.medianBlur` 对 float32 只支持 ksize ≤ 5，所以大核改用滑窗取中值。
    """
    k = int(k)
    if k < 3 or a.size <= k:
        return a.astype(np.float32)
    if k % 2 == 0:
        k += 1
    if k <= 5:
        return cv2.medianBlur(a.astype(np.float32).reshape(-1, 1), k).ravel()
    pad = k // 2
    b = np.pad(a.astype(np.float32), pad, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(b, k)
    return np.nanmedian(win, axis=-1).astype(np.float32)


def median_big(g, ks):
    """大窗口中值滤波。

    ⚠ OpenCV 5.0 的 8U medianBlur 在大核上是「数据相关」的：同一个 ksize，随机图能过，
    换成真实照片就报 `(-215) k < 16 in function 'medianBlur_8u_O1'`。所以这里在报错时
    退化为「先降采样 → 小核中值 → 再升采样」，等效窗口基本不变（降采样倍数由 ks 反推）。
    """
    ks = int(ks)
    if ks % 2 == 0:
        ks += 1
    if ks <= 15:
        return cv2.medianBlur(g, ks)
    try:
        return cv2.medianBlur(g, ks)
    except cv2.error:
        h, w = g.shape[:2]
        f = max(2, int(np.ceil(ks / 15.0)))
        small = cv2.resize(g, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
        k = max(3, min(15, int(round(ks / float(f))) | 1))
        return cv2.resize(cv2.medianBlur(small, k), (w, h), interpolation=cv2.INTER_LINEAR)


def group_consec(idx, gap=3):
    """把相邻的下标聚成组"""
    out = []
    for i in idx:
        if out and i - out[-1][-1] <= gap:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def load_image(path):
    """读图并应用 EXIF 方向（手机照片常带旋转标记）"""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def save_pdf(pages, path, dpi, title=None):
    """pages: list of HxW (灰度) 或 HxWx3 (BGR) 的 uint8 数组"""
    imgs = []
    for p in pages:
        if p.ndim == 3:
            p = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
        imgs.append(Image.fromarray(p))
    kw = dict(resolution=float(dpi))
    if title:
        kw["title"] = title
    if len(imgs) == 1:
        imgs[0].save(path, "PDF", **kw)
    else:
        imgs[0].save(path, "PDF", save_all=True, append_images=imgs[1:], **kw)


# ================================================================ 1. 纸张定位
def order_pts(pts):
    """四角排序为 左上 → 右上 → 右下 → 左下"""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def _illum_norm_bright(img):
    """先除以大尺度光照场把亮度拉平，再 Otsu —— 纸与背景亮度接近时这一招很有效"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    s = max(8.0, 0.12 * max(g.shape))
    bg = cv2.GaussianBlur(g, (0, 0), s)
    gn = np.clip(g / np.maximum(bg, 1.0) * 200.0, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(gn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gn > t


def _mask_candidates(img):
    """多种「纸面 vs 背景」判据。纸与背景亮度接近时，色度（b*）或去光照后再阈值才是有效判据。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b = lab[:, :, 2].astype(np.int16) - 128
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1].astype(np.int16), hsv[:, :, 2].astype(np.int16)
    out = {}
    for t in (-2, -4, -6, -8):
        out["b*<%d" % t] = b < t
    # 注意用 <=：图像接近无色时 S 的分位数为 0，用 < 会让该判据整幅失效
    out["亮+低饱和"] = (V > np.percentile(V, 70)) & (S <= max(1, np.percentile(S, 60)))
    _t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out["Otsu亮区"] = g > _t
    out["去光照+Otsu"] = _illum_norm_bright(img)
    return out, g


def quad_edge_support(gray, quad, sample=70, span=13.0, min_grad=8.0):
    """四条边贴不贴「真实图像边缘」——每边等距采样，看法向 ±span 内有没有强梯度。

    返回 4 个支撑率（0~1），顺序同 order_pts（上/右/下/左）。
    用途：亮度/色度分割在「纸与背景亮度接近」或「纸有阴影」时可能切出错误的四边形，
    此时它的边会落在阴影边界而非纸边——用这个指标即可把这类错误候选筛掉。
    """
    H, W = gray.shape
    out = []
    for i in range(4):
        A, B = quad[i], quad[(i + 1) % 4]
        d = B - A
        L = float(np.linalg.norm(d))
        if L < 10:
            out.append(0.0)
            continue
        dn = d / L
        n = np.array([-dn[1], dn[0]])
        ts = np.arange(-span, span + 1e-9, 1.0)
        hit = tot = 0
        for f in np.linspace(0.04, 0.96, sample):
            p = A + f * d
            xs, ys = p[0] + ts * n[0], p[1] + ts * n[1]
            if xs.min() < 1 or ys.min() < 1 or xs.max() > W - 2 or ys.max() > H - 2:
                continue
            prof = cv2.remap(gray, xs.astype(np.float32).reshape(1, -1),
                             ys.astype(np.float32).reshape(1, -1),
                             cv2.INTER_LINEAR).ravel().astype(np.float64)
            if prof.size != ts.size:
                continue
            tot += 1
            if np.abs(np.gradient(prof, ts)).max() >= min_grad:
                hit += 1
        out.append(hit / tot if tot else 0.0)
    return out


def _quad_from_mask(m):
    """掩膜 → 最大连通域 → 凸包 → 四边形。返回 (quad, 面积占比) 或 None"""
    m = (m.astype(np.uint8) * 255)
    # 注意：这里作用在「源图」上，核尺寸按源图定，不能跟输出 DPI 缩放
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8), iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((21, 21), np.uint8), iterations=2)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(st[1:, 4]))
    frac = st[i, 4] / float(m.size)
    if frac < 0.05:
        return None
    mm = np.where(lab == i, 255, 0).astype(np.uint8)
    ff = mm.copy()
    cv2.floodFill(ff, np.zeros((mm.shape[0] + 2, mm.shape[1] + 2), np.uint8), (0, 0), 255)
    mm = mm | (~ff)                                    # 填内部空洞（文字造成）
    cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.05, 0.08, 0.12):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            return order_pts(ap.reshape(-1, 2)), frac
    return None


def find_paper_quad(img, verbose=True):
    """多策略找纸张四边形；返回 quad 或 None"""
    cands, g = _mask_candidates(img)
    gmean = float(g.mean())
    gray_f = g.astype(np.float32)
    best, best_info = None, None
    for name, m in cands.items():
        r = _quad_from_mask(m)
        if r is None:
            continue
        q, frac = r
        if not (0.08 <= frac <= 0.97):
            continue
        inside = cv2.mean(g, mask=cv2.fillPoly(np.zeros(g.shape, np.uint8),
                                               [q.astype(np.int32)], 255))[0]
        if inside < gmean * 0.95:      # 纸面通常比背景亮
            continue
        side = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
        if min(side) < 0.08 * max(img.shape[:2]):
            continue
        sup = quad_edge_support(gray_f, q)
        smin = float(min(sup))
        if smin < QUAD_MIN_EDGE_SUPPORT:   # 边不贴真边缘 → 分割切错了（阴影/亮度接近）
            if verbose:
                log("    候选 %-10s 面积占比 %.3f → 四边支撑 %s 过低，弃用"
                    % (name, frac, ["%.2f" % v for v in sup]))
            continue
        score = frac * min(inside / max(gmean, 1e-3), 1.6) * (0.4 + 0.6 * smin)
        if verbose:
            log("    候选 %-10s 面积占比 %.3f 亮度比 %.3f 边支撑 %s → 得分 %.3f"
                % (name, frac, inside / max(gmean, 1e-3),
                   ["%.2f" % v for v in sup], score))
        if best is None or score > best:
            best, best_info = score, (name, q, frac)
    if best_info and verbose:
        log("    选中候选：%s（得分 %.3f）" % (best_info[0], best))
    return None if best_info is None else best_info[1]


def fit_line(pts):
    pts = np.asarray(pts, np.float64)
    mu = pts.mean(axis=0)
    return mu, np.linalg.svd(pts - mu)[2][0]


def intersect(l1, l2):
    (p1, d1), (p2, d2) = l1, l2
    return p1 + np.linalg.solve(np.array([d1, -d2]).T, p2 - p1)[0] * d1


def snap_side(gray, A, B, sample=90, span=18.0, min_grad=6.0):
    """沿一条边采样，用法向灰度梯度找真实纸边（亚像素）"""
    H, W = gray.shape
    d = B - A
    L = float(np.linalg.norm(d))
    if L < 10:
        return []
    dn = d / L
    n = np.array([-dn[1], dn[0]])
    ts = np.arange(-span, span + 1e-9, 0.5)
    res = []
    for f in np.linspace(0.05, 0.95, sample):
        p = A + f * d
        xs, ys = p[0] + ts * n[0], p[1] + ts * n[1]
        if xs.min() < 1 or ys.min() < 1 or xs.max() > W - 2 or ys.max() > H - 2:
            continue
        prof = cv2.remap(gray,
                         xs.astype(np.float32).reshape(1, -1),
                         ys.astype(np.float32).reshape(1, -1),
                         cv2.INTER_LINEAR).ravel().astype(np.float64)
        if prof.size != ts.size:
            continue
        prof = cv2.GaussianBlur(prof.reshape(1, -1), (0, 0), 1.5).ravel()
        gr = np.abs(np.gradient(prof, ts))
        k = int(np.argmax(gr))
        if gr[k] < min_grad:
            continue
        off = 0.0
        if 0 < k < gr.size - 1:
            y0, y1, y2 = gr[k - 1], gr[k], gr[k + 1]
            den = y0 - 2 * y1 + y2
            if abs(den) > 1e-9:
                off = float(np.clip(0.5 * (y0 - y2) / den, -1, 1))
        res.append((p + (ts[k] + off * 0.5) * n, abs(ts[k])))
    return res


def refine_quad(gray, quad, iters=4):
    """迭代：每边梯度吸附 → 拟合直线 → 相邻边求交得到新角点"""
    quad = np.asarray(quad, np.float32)
    for _ in range(iters):
        lines = []
        for i in range(4):
            A, B = quad[i], quad[(i + 1) % 4]
            cand = snap_side(gray, A, B)
            if len(cand) < 15:
                lines.append(fit_line([A, B]))
                continue
            offs = np.array([c[1] for c in cand])
            pts = np.array([cand[j][0] for j in range(len(cand))
                            if offs[j] <= np.percentile(offs, 75)])
            mu, d = fit_line(pts)
            res = np.abs((pts - mu) @ np.array([-d[1], d[0]]))
            keep = res <= max(2.0, np.percentile(res, 80))
            lines.append(fit_line(pts[keep]) if keep.sum() >= 10 else (mu, d))
        new = np.array([intersect(lines[(i - 1) % 4], lines[i]) for i in range(4)], np.float32)
        done = np.abs(new - quad).max() < 0.15
        quad = new
        if done:
            break
    return quad


def parse_corners(s):
    """'x1,y1 x2,y2 x3,y3 x4,y4' → 4x2 float32（左上/右上/右下/左下）"""
    nums = [float(v) for v in s.replace(",", " ").split()]
    if len(nums) != 8:
        raise ValueError("--corners 需要 8 个数字：x1,y1 x2,y2 x3,y3 x4,y4")
    return np.array(nums, np.float32).reshape(4, 2)


def pick_page_size(quad, paper, dpi):
    """决定输出像素尺寸；paper='auto' 时按四边形长宽比在 A4/Letter 里选"""
    if paper != "auto":
        wmm, hmm = PAPER_MM[paper]
        return int(round(wmm / 25.4 * dpi)), int(round(hmm / 25.4 * dpi))
    s = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)]
    a = (s[0] + s[2]) / 2.0
    b = (s[1] + s[3]) / 2.0
    ar = max(a, b) / max(min(a, b), 1e-6)
    name = "letter" if abs(ar - 1.294) < abs(ar - 1.414) else "a4"
    wmm, hmm = PAPER_MM[name]
    if a > b:                      # 横向
        wmm, hmm = hmm, wmm
    return int(round(wmm / 25.4 * dpi)), int(round(hmm / 25.4 * dpi))


def warp_to_page(img, quad, size):
    W, H = size
    M = cv2.getPerspectiveTransform(quad.astype(np.float32),
                                    np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]))
    warp = cv2.warpPerspective(img, M, (W, H), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), M, (W, H),
                               cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warp, mask


# ================================================================ 3. 光照归一化
def make_field(gray, m, s1=None, s2=None, level=LEVEL):
    """用掩膜 m 上的像素做归一化卷积，估出纸面背景场并相除"""
    s1 = pxf(45.0) if s1 is None else s1
    s2 = pxf(25.0) if s2 is None else s2
    mm = m.astype(np.float32)
    den = np.maximum(cv2.GaussianBlur(mm, (0, 0), s1), 1e-3)
    bg = cv2.GaussianBlur(gray * mm, (0, 0), s1) / den
    bg = np.maximum(cv2.GaussianBlur(bg, (0, 0), s2), 1.0)
    return np.clip(gray / bg * level, 0, 255), bg


def paper_pixels(gray, core, med=None, delta=8.0):
    """「纸面像素」判据：不比局部中位暗太多（自适应光照，且能排除笔画）"""
    loc = median_big(gray.astype(np.uint8), px(31) if med is None else med).astype(np.float32)
    return core & (gray > loc - delta)


# ================================================================ 4. 表格拉直
def _rules(resp, axis, min_span, rel):
    """在长结构响应图上找线：返回 [{a,b,c,strength,span}]"""
    prof = resp.sum(axis=1) if axis == 0 else resp.sum(axis=0)
    if prof.max() <= 0:
        return []
    idx = np.where(prof > prof.max() * rel)[0]
    gap = max(4, int(round(RULE_GROUP_GAP * max(resp.shape))))
    out = []
    for gg in group_consec(idx, gap=gap):
        a, b = gg[0], gg[-1]
        sl = slice(a, b + 1)
        band = resp[sl].max(axis=0) if axis == 0 else resp[:, sl].max(axis=1)
        out.append(dict(a=int(a), b=int(b), c=(a + b) / 2.0,
                        strength=float(prof[gg].max()),
                        span=float((band > 0).mean())))
    return [x for x in out if x["span"] >= min_span]


def detect_rules(norm, min_span=0.45, rel=0.25):
    """检出长横线/竖线（文字会被方向性开运算滤掉）"""
    H, W = norm.shape
    dark = dark_map(norm)
    Hl = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((1, max(31, W // 8)), np.uint8))
    Vl = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((max(31, H // 8), 1), np.uint8))
    return Hl, Vl, _rules(Hl, 0, min_span, rel), _rules(Vl, 1, min_span, rel)


def trace_curve(resp, axis, seed, lo, hi, step=None, win=None):
    """在长结构响应图（文字已被滤掉）上跟踪一条线，不会漂到文字上"""
    step = px(4, 2) if step is None else step
    win = px(14, 3) if win is None else win
    idx = np.arange(lo, hi, step)
    cur, ys = float(seed), []
    for i in idx:
        a = max(0, int(round(cur)) - win)
        b = min(resp.shape[axis], int(round(cur)) + win + 1)
        prof = (resp[a:b, i] if axis == 0 else resp[i, a:b]).astype(np.float64)
        if prof.size == 0 or prof.max() <= 0:
            ys.append(np.nan)
            continue
        k = int(np.argmax(prof))
        p = a + k
        if 0 < k < prof.size - 1:
            v0, v1, v2 = prof[k - 1], prof[k], prof[k + 1]
            den = v0 - 2 * v1 + v2
            if abs(den) > 1e-9:
                p += float(np.clip(0.5 * (v0 - v2) / den, -1, 1))
        ys.append(p)
        cur = p
    return idx, np.array(ys)


CURVE_SIGMA = 3.0      # 框线拟合曲线的平滑尺度（px）：越大越平滑、越不跟测线噪声


def make_curve(p, q, m, sigma=CURVE_SIGMA):
    """把跟踪到的线心点变成「曲线函数」 curve(x) → y。

    做法：域内按 1px 栅格重采样后做高斯平滑；域外按端点斜率线性外推。
    为什么不用多项式：deg≥8 虽能把域内残差压到 0.4px，但跟踪域之外（页面两端
    约 100px）的外推会爆炸（实测 x=0 处偏出 500px），把整页拉歪；deg=6 又太僵，
    域内还残留 4px 波弯。高斯平滑 + 线性外推两头都稳。
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = np.asarray(m, dtype=bool)
    if m.sum() < 12:
        return None
    ps, qs = p[m], q[m]
    o = np.argsort(ps)
    ps, qs = ps[o], qs[o]
    grid = np.arange(float(np.floor(ps[0])), float(np.ceil(ps[-1])) + 1.0, 1.0)
    if grid.size < 8:
        return None
    g = np.interp(grid, ps, qs)
    k = max(3, int(round(4.0 * sigma)) | 1)
    ker = np.exp(-0.5 * ((np.arange(k) - k // 2) / max(sigma, 0.5)) ** 2)
    ker /= ker.sum()
    gs = np.convolve(np.pad(g, (k // 2, k // 2), mode="edge"), ker, mode="valid")[:grid.size]
    d0 = float(gs[1] - gs[0]) if gs.size > 1 else 0.0
    d1 = float(gs[-1] - gs[-2]) if gs.size > 1 else 0.0

    def ev(x):
        x = np.asarray(x, dtype=np.float64)
        y = np.interp(x, grid, gs)
        if d0:
            y = np.where(x < grid[0], gs[0] + d0 * (x - grid[0]), y)
        if d1:
            y = np.where(x > grid[-1], gs[-1] + d1 * (x - grid[-1]), y)
        return y

    return ev


def robust_fit(p, q, deg=6, iters=5, sigma=CURVE_SIGMA):
    """含离群剔除的稳健拟合。返回 (curve_fn, mask)。

    ① 用高阶多项式残差剔掉离群跟踪点（多项式只用来「判离群」，不拿来当最终曲线）；
    ② 用 make_curve 在剩余点上构造平滑曲线（域内平滑、域外线性外推）。
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = ~np.isnan(q)
    if m.sum() < 12:
        return None, m
    deg = int(min(deg, max(1, m.sum() // 8)))
    c = np.polyfit(p[m], q[m], deg)
    for _ in range(iters):
        r = q - np.polyval(c, p)
        s = np.nanmedian(np.abs(r[m])) * 1.4826 + 0.05
        nm = (~np.isnan(q)) & (np.abs(r) < max(0.8, 3 * s))
        if nm.sum() < 12:
            break
        c = np.polyfit(p[nm], q[nm], deg)
        m = nm
    return make_curve(p, q, m, sigma), m


def dark_map(norm):
    """归一化图 → 「暗度」图（线越黑响应越大）"""
    return np.clip(RULE_REF * LEVEL - norm, 0, None).astype(np.float32)


def trace_line_dark(dark, axis, c0, i0, i_end, step=None, win=None, thr=0.6, smooth=0):
    """在「暗度图」上以固定窄窗（±win）跟踪线心（暗度加权质心）。

    与 trace_curve 的关键区别：不依赖长核形态学开运算的连续响应，因此框线因弯曲
    被开运算打断的段（页面两端）也能继续跟住，不会退化成靠多项式外推。
    thr 是加权用的相对阈值：0.6 跟得稳（适合拉直），0.8 只取线芯（适合量直线度，
    避免被线旁淡墨尾巴把质心带偏）。smooth>1 时对结果做中值平滑。
    """
    step = px(4, 2) if step is None else step
    win = px(14, 3) if win is None else win
    if i_end == i0:
        return np.array([]), np.array([])
    d = step if i_end > i0 else -step
    cur = float(c0)
    idx, ys = [], []
    for i in range(i0, i_end, d):
        a = max(0, int(round(cur)) - win)
        b = min(dark.shape[axis], int(round(cur)) + win + 1)
        if b - a < 3:
            continue
        prof = (dark[a:b, i] if axis == 0 else dark[i, a:b]).astype(np.float64)
        mx = float(prof.max())
        if mx <= 1.0:
            continue
        w = np.clip(prof - thr * mx, 0, None)
        s = float(w.sum())
        if s <= 1e-6:
            continue
        p = a + float((w * np.arange(prof.size)).sum() / s)
        idx.append(i)
        ys.append(p)
        cur = p
    idx, ys = np.array(idx), np.array(ys)
    if smooth > 1 and ys.size > smooth:
        ys = median_flat(ys, smooth)
    return idx, ys


def trace_line_both(dark, axis, c_of, lo, hi, step=None, win=None, thr=0.6, smooth=0):
    """从区间中点分别向两端跟踪，再拼起来 —— 避免单向跟踪在起点误差上累积"""
    step = px(4, 2) if step is None else step
    win = px(14, 3) if win is None else win
    mid = (lo + hi) // 2
    c = float(c_of(mid))
    i1, y1 = trace_line_dark(dark, axis, c, mid, hi, step, win, thr, smooth)
    i2, y2 = trace_line_dark(dark, axis, c, mid, lo, step, win, thr, smooth)
    if i1.size == 0 and i2.size == 0:
        return np.array([]), np.array([])
    return np.concatenate([i2, i1]), np.concatenate([y2, y1])


def straighten(img, mask, norm, verbose=True):
    """自动检出表格外框并把弯曲的框线拉直（整页一起平滑变形）。
    返回 (img2, mask2, applied)，未检出足够直线时不改动。"""
    H, W = norm.shape
    Hl, Vl, hor, ver = detect_rules(norm)
    DARK = dark_map(norm)
    mgh, mgw = RULE_EDGE_MARGIN * H, RULE_EDGE_MARGIN * W
    # 只保留「足够长、响应足够强、且不离页面边缘太近」的线：
    #   文字行 / 下划线等弱结构被强度门限挡掉；
    #   贴页面边缘的暗带是纸张边缘渐晕/背景残留，被边缘余量挡掉。
    hor = [r for r in hor if r["span"] >= RULE_MIN_SPAN
           and r["strength"] / max(W, 1) >= RULE_MIN_STRENGTH
           and mgh < r["c"] < H - mgh]
    ver = [r for r in ver if r["span"] >= RULE_MIN_SPAN
           and r["strength"] / max(H, 1) >= RULE_MIN_STRENGTH
           and mgw < r["c"] < W - mgw]
    if verbose:
        log("    候选框线：横线 %d 条（强度 %s）、竖线 %d 条（强度 %s）"
            % (len(hor), ["%.0f" % (r["strength"] / W) for r in hor],
               len(ver), ["%.0f" % (r["strength"] / H) for r in ver]))
    if len(hor) < 2 or len(ver) < 2:
        return img, mask, False
    T, B = hor[0], hor[-1]
    L, R = ver[0], ver[-1]
    if (B["c"] - T["c"]) < 0.35 * H or (R["c"] - L["c"]) < 0.35 * W:
        return img, mask, False

    def edge_line(resp, axis, grp, lo, hi, span):
        """两级跟踪：① 长核开运算响应上粗跟（抗文字干扰）；
        ② 在原始暗度图上沿全宽窄窗双向跟线心（能跟住被开运算打断的弯曲端段）。"""
        p, q = trace_curve(resp, axis, grp["c"], lo, hi)
        c, m = robust_fit(p, q)
        if c is None or m.sum() < 12:
            return None
        lo2, hi2 = int(0.012 * span), int(0.988 * span)
        p2, q2 = trace_line_both(DARK, axis, c, lo2, hi2, win=10, thr=0.7)
        c2, m2 = robust_fit(p2, q2)
        if c2 is not None and m2.sum() >= max(24, int(0.35 * m.sum())):
            return c2, m2, p2, q2
        return c, m, p, q

    eT = edge_line(Hl, 0, T, px(60), W - px(60), W)
    eB = edge_line(Hl, 0, B, px(60), W - px(60), W)
    eL = edge_line(Vl, 1, L, px(60), H - px(60), H)
    eR = edge_line(Vl, 1, R, px(60), H - px(60), H)
    if any(e is None for e in (eT, eB, eL, eR)):
        return img, mask, False

    Xs, Ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    yT = float(np.nanmedian(eT[3]))
    yB = float(np.nanmedian(eB[3]))
    xL = float(np.nanmedian(eL[3]))
    xR = float(np.nanmedian(eR[3]))

    def dev(e):        # 相对直线的弯曲量（峰谷）
        p, q, m = e[2], e[3], e[1]
        d = (q - np.polyval(np.polyfit(p[m], q[m], 1), p))[m]
        return float(np.nanmax(d) - np.nanmin(d))

    devs = [dev(e) for e in (eT, eB, eL, eR)]
    if verbose:
        log("    弯曲量（上/下/左/右，px）：%s" % ["%.2f" % d for d in devs])
    if max(devs) < 1.2 and not FORCE_STRAIGHTEN:
        if verbose:
            log("    弯曲 <1.2px，无需拉直")
        return img, mask, False

    v = (Ys - yT) / max(yB - yT, 1.0)
    src_y = eT[0](Xs) + v * (eB[0](Xs) - eT[0](Xs))
    u = (Xs - xL) / max(xR - xL, 1.0)
    src_x = eL[0](Ys) + u * (eR[0](Ys) - eL[0](Ys))
    img2 = cv2.remap(img, src_x.astype(np.float32), src_y.astype(np.float32),
                     cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    mask2 = cv2.remap(mask, src_x.astype(np.float32), src_y.astype(np.float32),
                      cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img2, mask2, True


# ================================================================ 5. 底纹检测
def detect_shading(gray, core, verbose=True):
    """检出大片淡灰打印底色（表单底纹）。

    背景基准 = 大窗中值（中位比极大稳健得多，不会被 JPEG 振铃/局部亮点抬高；
    窗宽远大于条带厚度时，条带处的中位仍落在周围纸面上）。
      rel  = 原图 / 该基准 / 纸面基准  →  纸面 ≈1.00，底纹 ≈0.94，墨迹 ≪1
    为避免文字抗锯齿边缘把「纸面像素」的范围拉宽，纸面像素取 g > 0.90 倍基准；
    再把 rel 在纸面像素上平滑（relf）再阈值化，掩膜才不会被文字打散。
    """
    H, W = gray.shape
    ks = int(np.clip(round(0.105 * max(H, W)), px(31), px(351))) | 1
    ref = median_big(gray.astype(np.uint8), ks).astype(np.float32)
    r = gray / np.maximum(ref, 1.0)
    sel = core & (gray > 0.90 * ref)
    if sel.sum() < 1000:
        if verbose:
            log("    未检出底纹（可用纸面像素太少）")
        return None, None
    base_r = float(np.percentile(r[sel], 90))
    rel = r / max(base_r, 1e-6)
    pm = sel.astype(np.float32)
    relf = cv2.GaussianBlur(rel * pm, (0, 0), 8.0) / \
        np.maximum(cv2.GaussianBlur(pm, (0, 0), 8.0), 1e-3)

    # 分块中位 → 自适应阈值（纸面块中位的 75 分位再压低 4%）
    bs = 12
    hy, wy = max(1, H // bs), max(1, W // bs)
    meds = []
    for by in range(bs):
        for bx in range(bs):
            v = relf[by * hy:(by + 1) * hy, bx * wy:(bx + 1) * wy]
            sm = core[by * hy:(by + 1) * hy, bx * wy:(bx + 1) * wy]
            v = v[sm]
            v = v[v > 0.9]
            if v.size > 200:
                meds.append(float(np.median(v)))
    if len(meds) < 12:
        if verbose:
            log("    未检出底纹（有效分块太少）")
        return None, None
    hi = float(np.clip(np.percentile(meds, 75) - 0.04, SHADE_LO + 0.02, SHADE_HI))
    if verbose:
        log("    纸面亮度基准 %.4f  分块中位 %.3f~%.3f → 底纹判定区间 [%.2f, %.3f]"
            % (base_r, min(meds), max(meds), SHADE_LO, hi))

    cand = ((relf > SHADE_LO) & (relf < hi) & core).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((px(9), px(9)), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE,
                            np.ones((px(5), px(151)), np.uint8))  # 跨过压在底纹上的文字
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, 8)
    if n <= 1:
        if verbose:
            log("    未检出底纹")
        return None, None
    min_area = SHADE_MIN_FRAC * float(core.sum())
    mx, my = 0.012 * W, 0.012 * H
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x0, y0, bw_, bh_ = st[i, 0], st[i, 1], st[i, 2], st[i, 3]
        if st[i, 4] < min_area:
            continue
        if x0 <= mx or y0 <= my or x0 + bw_ >= W - mx or y0 + bh_ >= H - my:
            continue          # 贴页面边缘的暗块 → 判为纸张边缘渐晕，不是印刷底纹
        keep[i] = True
    if not keep.any():
        if verbose:
            log("    未检出底纹（连通域均 < %.2f%% 页面积或不满足形状条件）" % (100 * SHADE_MIN_FRAC))
        return None, None
    shade = keep[lab]
    frac = shade.sum() / float(core.sum())
    if frac > SHADE_MAX_FRAC:            # 面积过大 → 大概率是光照残留，宁可不做
        if verbose:
            log("    检出「底纹」达 %.1f%% 页面积，超过 %.0f%% 上限 → 判为光照残留，忽略"
                % (100 * frac, 100 * SHADE_MAX_FRAC))
        return None, None
    shade = cv2.morphologyEx((shade * 255).astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((11, 11), np.uint8)) > 0
    shade &= core
    inner = cv2.erode(shade.astype(np.uint8), np.ones((px(9), px(9)), np.uint8)) > 0
    probe = inner if inner.sum() > 500 else shade
    g_shade = int(round(255.0 * float(np.median(rel[probe]))))
    if verbose:
        log("    检出底纹 %.2f%% 页面积，相对纸面亮度 %.4f → 输出淡灰 %d"
            % (100 * frac, float(np.median(rel[probe])), g_shade))
    return shade, g_shade


def rule_curvature(norm):
    """在给定归一化图上检出长直线并量其弯曲量（相对直线峰谷，px）。返回 {tag: px}"""
    H, W = norm.shape
    Hl, Vl, hor, ver = detect_rules(norm)
    mgh, mgw = RULE_EDGE_MARGIN * H, RULE_EDGE_MARGIN * W
    hor = [r for r in hor if r["span"] >= RULE_MIN_SPAN
           and r["strength"] / max(W, 1) >= RULE_MIN_STRENGTH
           and mgh < r["c"] < H - mgh]
    ver = [r for r in ver if r["span"] >= RULE_MIN_SPAN
           and r["strength"] / max(H, 1) >= RULE_MIN_STRENGTH
           and mgw < r["c"] < W - mgw]
    out = {}
    jobs = []
    if len(hor) >= 1:
        jobs.append(("上框", Hl, 0, hor[0], W))
        if len(hor) >= 2:
            jobs.append(("下框", Hl, 0, hor[-1], W))
    if len(ver) >= 1:
        jobs.append(("左框", Vl, 1, ver[0], H))
        if len(ver) >= 2:
            jobs.append(("右框", Vl, 1, ver[-1], H))
    DARK = dark_map(norm)
    for tag, resp, axis, grp, span in jobs:
        lo, hi = (px(60), W - px(60)) if axis == 0 else (px(60), H - px(60))
        p, q = trace_curve(resp, axis, grp["c"], lo, hi)
        c, m = robust_fit(p, q)
        if c is None:
            continue
        # 用与 straighten 相同的两级跟踪，量「全宽」的弯曲量，避免只量到中间平坦段；
        # 这里只取线芯（thr 高）+ 中值平滑，防止被线旁淡墨尾巴把质心带偏。
        p2, q2 = trace_line_both(DARK, axis, c,
                                 int(0.012 * span), int(0.988 * span),
                                 win=px(8, 3), thr=0.8, smooth=px(5, 1))
        c2, m2 = robust_fit(p2, q2)
        if c2 is not None and m2.sum() >= max(24, int(0.35 * m.sum())):
            p, q, m = p2, q2, m2
        d = (q - np.polyval(np.polyfit(p[m], q[m], 1), p))[m]
        out[tag] = float(np.nanmax(d) - np.nanmin(d))
    return out


# ================================================================ 6. 二值化
def binarize(norm, k=K_DEFAULT, win=WIN_DEFAULT, ss=SS_DEFAULT, gate=None):
    """Sauvola 局部阈值（在 ss 倍超采样上做，降采样后边缘自然抗锯齿）"""
    H, W = norm.shape
    big = cv2.resize(norm, (W * ss, H * ss), interpolation=cv2.INTER_CUBIC)
    big = cv2.medianBlur(big.astype(np.uint8), 3).astype(np.float32)
    w = max(3, int(round(win * ss * SCALE))) | 1
    mu = cv2.boxFilter(big, -1, (w, w), normalize=True, borderType=cv2.BORDER_REPLICATE)
    mu2 = cv2.boxFilter(big * big, -1, (w, w), normalize=True, borderType=cv2.BORDER_REPLICATE)
    sd = np.sqrt(np.maximum(mu2 - mu * mu, 0))
    ink = big < mu * (1.0 + k * (sd / 128.0 - 1.0))
    if gate is not None:                # 明显亮于「墨」的像素不可能是墨（防止底纹被误判）
        ink &= big < gate
    return ink


def clean_binary(binary, min_area, max_hole):
    """删小碎点 + 补笔画小孔（必须限制最大补孔面积，否则会吞掉表格框内部）"""
    b = binary.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, 4] >= min_area
    b = keep[lab].astype(np.uint8)
    h, wd = b.shape
    ff = b.copy()
    cv2.floodFill(ff, np.zeros((h + 2, wd + 2), np.uint8), (0, 0), 1)
    holes = ((ff == 0) & (b == 0)).astype(np.uint8)
    nh, lh, sh, _ = cv2.connectedComponentsWithStats(holes, 8)
    if nh > 1:
        small = np.zeros(nh, bool)
        small[1:] = sh[1:, 4] <= max_hole
        b[small[lh] & (holes > 0)] = 1
    return b.astype(bool)


def text_height(inkB, size, ss=SS_DEFAULT):
    """估算正文字号：去掉长直线后，取文字连通域「高度」的中位数（输出分辨率 px）。

    这是「要不要改用灰度渲染」的判据。手机拍摄的整页表格里，正文常只有十几像素高，
    此时硬二值化会把笔画削断成锯齿（字不成形）；而大字号文档二值化反而最干净。
    """
    W, H = size
    m = cv2.resize(inkB.astype(np.uint8), (W, H), interpolation=cv2.INTER_AREA) > 0
    if m.sum() < 50:
        return 0.0
    u8 = (m * 255).astype(np.uint8)
    lin = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((1, max(31, W // 10)), np.uint8))
    lin |= cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((max(31, H // 10), 1), np.uint8))
    lin = cv2.dilate(lin, np.ones((px(5), px(5)), np.uint8))
    txt = ((u8 > 0) & (lin == 0)).astype(np.uint8)
    n, _lab, st, _ = cv2.connectedComponentsWithStats(txt, 8)
    if n <= 1:
        return 0.0
    area, hh = st[1:, 4], st[1:, 3]
    sel = (area >= 12) & (area <= 4000) & (hh >= 3)
    if sel.sum() < 20:
        return 0.0
    return float(np.median(hh[sel]))


def tone_cover(norm, core, ink=INK_LEVEL, lo=TONE_LO, hi=TONE_HI):
    """「灰度渲染」的墨迹覆盖率（0=纸面纯白，1=实墨全黑）。

    与硬二值化的区别：保留笔画边缘的中间灰度，笔画不断、字腔不空，小字明显更清楚。
    LO/HI 软膝负责把纸面纹理压成纯白、把实墨拉成纯黑，中间线性过渡即抗锯齿。

    纸面基准必须取「局部」而不是全局 LEVEL：纸面自身有慢变化（页边渐晕、纸纹）与
    印刷淡灰底纹，用全局基准会把它们当成淡墨压成灰（实测底纹 234 → 219，页边一圈灰带）。
    局部基准由 `paper_pixels`（排除笔画）平滑得到，只跟随纸面的慢变化。
    """
    n = median_big(norm.astype(np.uint8), px(TONE_DENOISE, 3)).astype(np.float32)
    pm = paper_pixels(n, core).astype(np.float32)
    sig = pxf(TONE_REF_SIGMA)
    den = np.maximum(cv2.GaussianBlur(pm, (0, 0), sig), 1e-3)
    ref = cv2.GaussianBlur(n * pm, (0, 0), sig) / den
    c = np.clip((ref - n) / max(LEVEL - ink, 1e-6), 0.0, 1.0)
    c = np.clip((c - lo) / max(hi - lo, 1e-6), 0.0, 1.0) ** TONE_GAMMA
    return c


# ================================================================ 单张处理
def fallback_rect(img, expand=0.04):
    """纸张四角定位失败时的兜底裁剪框。

    取「亮区最大连通域」的外接矩形，四边各放 expand*长边 的余量（放宽是为了不切掉内容，
    亮区通常会漏掉纸的暗边/阴影边）。返回 (quad, 说明) 或 None。
    比「整图缩放进页面」干净得多：四周的桌面/阴影会被裁掉。
    """
    cands, g = _mask_candidates(img)
    H, W = g.shape
    gmean = float(g.mean())
    best = None
    for name, m in cands.items():
        mm = (m.astype(np.uint8) * 255)
        mm = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8), iterations=2)
        mm = cv2.morphologyEx(mm, cv2.MORPH_OPEN, np.ones((21, 21), np.uint8), iterations=2)
        n, lab, st, _ = cv2.connectedComponentsWithStats(mm, 8)
        if n <= 1:
            continue
        i = 1 + int(np.argmax(st[1:, 4]))
        if st[i, 4] / float(mm.size) < 0.10:
            continue
        x0, y0, bw_, bh_ = int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3])
        if g[y0:y0 + bh_, x0:x0 + bw_].mean() < gmean:
            continue
        area = bw_ * bh_
        if best is None or area > best[0]:
            best = (area, x0, y0, bw_, bh_, name)
    if best is None:
        return None
    _, x0, y0, bw_, bh_, name = best
    e = expand * max(H, W)
    lx, ty = max(0, int(x0 - e)), max(0, int(y0 - e))
    rx, by = min(W - 1, int(x0 + bw_ + e)), min(H - 1, int(y0 + bh_ + e))
    if (rx - lx) < 0.45 * W or (by - ty) < 0.45 * H:
        return None
    q = np.array([[lx, ty], [rx, ty], [rx, by], [lx, by]], np.float32)
    return q, name


def whiten_border(img, lo=0.40, hi=0.96, band=0.18, min_frac=0.0008, level=LEVEL):
    """把「与页面边缘连通的中灰区域」（桌面/阴影/渐晕残留）涂白，不动墨迹。

    只在页面四周 band 比例的外带内生效；墨迹核心远暗于 lo*LEVEL，因此不会被吃掉。
    纸张定位失败退化为「整图即文档」时，这一步能去掉画面四周残留的桌面/阴影灰框。
    返回 (新图, 涂白像素数)。灰度图与 BGR 图都可用。
    """
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    mid = ((g > lo * level) & (g < hi * level)).astype(np.uint8)
    if mid.sum() == 0:
        return img, 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(mid, 8)
    if n <= 1:
        return img, 0
    edge = set(lab[0, :].tolist()) | set(lab[-1, :].tolist()) | \
        set(lab[:, 0].tolist()) | set(lab[:, -1].tolist())
    edge.discard(0)
    if not edge:
        return img, 0
    by, bx = int(band * H), int(band * W)
    bm = np.zeros((H, W), bool)
    bm[:by, :] = True
    bm[-by:, :] = True
    bm[:, :bx] = True
    bm[:, -bx:] = True
    m = np.zeros((H, W), bool)
    for i in edge:
        if st[i, 4] >= min_frac * H * W:
            m |= (lab == i)
    m &= bm
    if not m.any():
        return img, 0
    out = img.copy()
    if out.ndim == 2:
        out[m] = 255
    else:
        out[m] = 255
    return out, int(m.sum())


def process_one(path, out_dir, name, args, verbose=True):
    global SCALE
    SCALE = max(float(args.dpi), 1.0) / 200.0
    log("\n=== 处理：%s ===" % os.path.basename(path))
    img = load_image(path)
    H0, W0 = img.shape[:2]
    log("  源图 %dx%d" % (W0, H0))

    # --- 1/2. 定位 + 透视
    manual = parse_corners(args.corners) if args.corners else None
    degraded = False
    if args.no_crop:
        quad = None
        degraded = True
    elif manual is not None:
        quad = manual.astype(np.float32)
        log("  使用手工角点")
    else:
        q0 = find_paper_quad(img, verbose)
        if q0 is None:
            fb = fallback_rect(img)
            if fb is not None:
                quad, nm = fb
                degraded = True
                log("  ⚠ 四角定位失败 → 用「%s」亮区外接矩形兜底裁剪 %s"
                    % (nm, np.round(quad, 0).astype(int).tolist()))
            else:
                if verbose:
                    log("  ⚠ 未能自动定位纸张 → 退化为「整图即文档」")
                quad = None
                degraded = True
        else:
            quad = refine_quad(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), q0)
            c = quad.mean(0)
            quad = quad + (c - quad) / np.linalg.norm(quad - c, axis=1, keepdims=True) * 4.0
            log("  纸张四角：%s" % np.round(quad, 1).tolist())

    if quad is None:
        size = pick_page_size(np.array([[0, 0], [W0 - 1, 0], [W0 - 1, H0 - 1], [0, H0 - 1]], np.float32),
                              args.paper, args.dpi)
    else:
        size = pick_page_size(quad, args.paper, args.dpi)
    W, H = size
    if quad is None:
        sc = min(W / W0, H / H0)
        r = cv2.resize(img, (max(1, int(W0 * sc)), max(1, int(H0 * sc))), interpolation=cv2.INTER_AREA)
        warp = np.full((H, W, 3), 255, np.uint8)
        y0 = (H - r.shape[0]) // 2
        x0 = (W - r.shape[1]) // 2
        warp[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
        mask = np.full((H, W), 255, np.uint8)
    else:
        warp, mask = warp_to_page(img, quad, size)
    log("  透视校正 → %dx%d (%.0f DPI)" % (W, H, args.dpi))

    # --- 3. 光照归一化（第一遍）
    if args.debug:
        cv2.imwrite(os.path.join(out_dir, "_debug", "%s_1_warp.png" % name), warp)
    g = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY).astype(np.float32)
    core = cv2.erode(mask, np.ones((px(9), px(9)), np.uint8), iterations=2) > 0
    pm = paper_pixels(g, core)
    norm0, _ = make_field(g, pm)

    # --- 4. 拉直
    if args.straighten != "off":
        warp2, mask2, applied = straighten(warp, mask, norm0, verbose)
        if applied:
            log("  已拉直表格外框")
        warp, mask = warp2, mask2
        g = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY).astype(np.float32)
        core = cv2.erode(mask, np.ones((px(9), px(9)), np.uint8), iterations=2) > 0
        pm = paper_pixels(g, core)
        norm0, _ = make_field(g, pm)
    if args.debug:
        cv2.imwrite(os.path.join(out_dir, "_debug", "%s_2_straight.png" % name), warp)

    # --- 5. 底纹
    shade, g_shade = (None, None)
    if args.shading != "off":
        shade, g_shade = detect_shading(g, core, verbose)
    if shade is None:
        shade = np.zeros((H, W), bool)

    # 最终光照场：纸面 + 底纹一起参与（底纹在归一化后仍是灰，稍后按 base 重绘）
    norm, _ = make_field(g, pm)

    # --- 6. 墨迹（两条路线）
    #  route bw   ：Sauvola 硬二值化 —— 大字号文档最干净
    #  route gray ：保色调的软覆盖 —— 小字号/低分辨率时硬二值会把笔画削断，灰度明显更清楚
    ink = binarize(norm, args.k, args.win, args.ss, gate=0.88 * LEVEL)
    inkB = clean_binary(ink, int(args.min_area * (args.ss / 2.0) ** 2 * SCALE ** 2),
                        int(args.max_hole * (args.ss / 2.0) ** 2 * SCALE ** 2))
    cover_bin = cv2.resize(inkB.astype(np.uint8) * 255, (W, H),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
    cover_bin[cover_bin < 18] = 0.0
    cover_bin /= 255.0

    th = text_height(inkB, (W, H), args.ss)
    th_mm = th / max(args.dpi, 1.0) * 25.4
    tone = args.tone
    if tone == "auto":
        tone = "gray" if (0 < th_mm < TONE_AUTO_TEXT_MM) else "bw"
    cover = (tone_cover(norm, core, args.tone_ink, args.tone_lo, args.tone_hi)
             if tone == "gray" else cover_bin)
    cover = np.where(core, cover, 0.0)

    # --- 合成
    base = np.full((H, W), 255.0, np.float32)
    if g_shade:
        base[shade] = float(g_shade)
    bw = np.clip(base * (1.0 - cover_bin), 0, 255).astype(np.uint8)
    gray = np.clip(base * (1.0 - cover), 0, 255).astype(np.uint8)

    # 彩色版：逐通道光照归一化 + 纸面/底纹为底，墨迹保留原色调
    f = warp.astype(np.float32)
    mm = pm.astype(np.float32)
    den = np.maximum(cv2.GaussianBlur(mm, (0, 0), 45.0), 1e-3)
    cbg = np.stack([cv2.GaussianBlur(
        cv2.GaussianBlur(f[:, :, ch] * mm, (0, 0), 45.0) / den, (0, 0), 25.0)
        for ch in range(3)], -1)
    cn = np.clip(f / np.maximum(cbg, 1.0) * 232.0, 0, 255)
    cb = np.full((H, W, 3), 255.0, np.float32)
    if g_shade:
        cb[shade] = float(g_shade)
    color = np.where((cover[:, :, None] > 0), cn, cb).astype(np.uint8)
    color = np.where(core[:, :, None], color, np.uint8(255))

    # 退化/兜底路径时，画面四周常残留桌面/阴影；把与边缘连通的中灰区涂白
    if degraded:
        bw, nw = whiten_border(bw)
        gray, _ = whiten_border(gray)
        color, _ = whiten_border(color)
        if verbose and nw:
            log("  已清理边缘残留中灰 %d 像素" % nw)

    main = gray if tone == "gray" else bw

    # --- 验收
    if verbose:
        log("  ── 验收 ──")
        log("    渲染：%s（正文高 %.1fmm，阈值 %.1fmm）"
            % ("灰度" if tone == "gray" else "黑白", th_mm, TONE_AUTO_TEXT_MM))
        log("    墨迹占比 %.2f%%   纸面纯白率 %.2f%%"
            % (100 * cover[core].mean(), 100 * (main[core] == 255).mean()))
        for tag, (y0, y1, x0, x1) in {"左上": (0.06, 0.21, 0.09, 0.42),
                                      "左中": (0.38, 0.64, 0.09, 0.42),
                                      "右中": (0.38, 0.64, 0.61, 0.94),
                                      "底部": (0.81, 0.96, 0.09, 0.91)}.items():
            r = main[int(H * y0):int(H * y1), int(W * x0):int(W * x1)]
            if (r > 160).any():
                log("    %-4s 纸面均值 %.1f" % (tag, r[r > 160].mean()))
        if g_shade:
            inside = main[shade]
            log("    底纹区 实测灰度中位 %.0f（目标 %d）" % (np.median(inside[inside > 120]), g_shade))
        ring = np.zeros_like(main, bool)
        ring[:12] = 1; ring[-12:] = 1; ring[:, :12] = 1; ring[:, -12:] = 1
        log("    页面四周12px环带非白像素 %d 个" % int((main[ring] < 250).sum()))
        cv = rule_curvature(norm)
        if cv:
            log("    外框直线度（px）：%s"
                % "  ".join("%s %.2f" % (k, v) for k, v in cv.items()))

    # --- 输出
    r = dict(name=name, main=main, bw=bw, gray=gray, tone=tone, text_h=th,
             color=color, size=(W, H), shade=g_shade)
    if verbose and args.debug:
        log("    调试中间图见 %s/_debug/" % out_dir)
    return r


# ================================================================ 主流程
FORCE_STRAIGHTEN = False


def main():
    global FORCE_STRAIGHTEN
    ap = argparse.ArgumentParser(description="拍摄文档照片 → 扫描件")
    ap.add_argument("-i", "--input", nargs="+", required=True, help="照片路径（可多个/可通配）")
    ap.add_argument("-o", "--output", default=".", help="输出目录")
    ap.add_argument("-n", "--name", default=None, help="输出文件名前缀（默认取第一张照片名）")
    ap.add_argument("--mode", choices=["both", "main", "color", "bw", "gray"], default="both",
                    help="输出哪些：both=主稿+彩色（默认）、main=只出主稿、color=只出彩色；"
                         "bw/gray 等价于 main（兼容旧写法）")
    ap.add_argument("--tone", choices=["auto", "bw", "gray"], default="auto",
                    help="主稿渲染：auto=按正文字号自动选（默认）、bw=硬二值黑白、gray=保色调灰度")
    ap.add_argument("--paper", choices=["auto", "a4", "letter", "a5"], default="a4",
                    help="纸张规格（默认 a4；auto=按四边形长宽比猜）")
    ap.add_argument("--dpi", type=float, default=200.0)
    ap.add_argument("--straighten", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--shading", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--corners", default=None, help='"x1,y1 x2,y2 x3,y3 x4,y4"（仅单张时可手工指定纸张四角）')
    ap.add_argument("--no-crop", action="store_true", help="不裁纸张，整图即文档")
    ap.add_argument("--k", type=float, default=K_DEFAULT, help="Sauvola k，越大笔画越细")
    ap.add_argument("--tone-ink", type=float, default=INK_LEVEL,
                    help="灰度渲染的「实墨」灰度（默认 %g）：调小 → 笔画更黑" % INK_LEVEL)
    ap.add_argument("--tone-lo", type=float, default=TONE_LO,
                    help="灰度渲染软膝下沿（默认 %g）：调小 → 淡笔画更容易显出来" % TONE_LO)
    ap.add_argument("--tone-hi", type=float, default=TONE_HI,
                    help="灰度渲染软膝上沿（默认 %g）" % TONE_HI)
    ap.add_argument("--win", type=int, default=WIN_DEFAULT)
    ap.add_argument("--ss", type=int, default=SS_DEFAULT)
    ap.add_argument("--min-area", type=float, default=MIN_AREA_DEFAULT)
    ap.add_argument("--max-hole", type=float, default=MAX_HOLE_DEFAULT)
    ap.add_argument("--debug", action="store_true", help="额外输出中间过程图")
    args = ap.parse_args()
    FORCE_STRAIGHTEN = (args.straighten == "on")

    files = []
    for pat in args.input:
        hit = sorted(_glob.glob(pat))
        files.extend(hit if hit else [pat])
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        sys.exit("[photo2scan] 找不到输入文件")

    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)
    if args.debug:
        os.makedirs(os.path.join(out_dir, "_debug"), exist_ok=True)
    name = args.name or os.path.splitext(os.path.basename(files[0]))[0]

    results = []
    for i, f in enumerate(files):
        nm = name if len(files) == 1 else "%s_%02d" % (name, i + 1)
        results.append(process_one(f, out_dir, nm, args))

    mains = [r["main"] for r in results]
    cols = [r["color"] for r in results]
    tone = results[0]["tone"] if results else "bw"
    suffix = "灰度" if tone == "gray" else "黑白"

    written = []
    if args.mode in ("both", "main", "bw", "gray"):
        for r in results:
            p = os.path.join(out_dir, r["name"] + "_扫描件_%s.png" % suffix)
            cv2.imwrite(p, r["main"]); written.append(p)
        p = os.path.join(out_dir, name + "_扫描件_%s.pdf" % suffix)
        save_pdf(mains, p, args.dpi, title=name)
        written.append(p)
    if args.mode in ("both", "color"):
        for r in results:
            p = os.path.join(out_dir, r["name"] + "_扫描件_彩色.png")
            cv2.imwrite(p, r["color"]); written.append(p)
        p = os.path.join(out_dir, name + "_扫描件_彩色.pdf")
        save_pdf(cols, p, args.dpi, title=name + "（彩色）")
        written.append(p)

    log("\n=== 输出 ===")
    for p in written:
        log("  %8d KB  %s" % (os.path.getsize(p) // 1024, p))


if __name__ == "__main__":
    main()
