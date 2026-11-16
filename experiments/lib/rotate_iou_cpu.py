"""CPU (numba.njit) port of OpenPCDet's kitti_object_eval_python/rotate_iou.py.

The original compiles numba.cuda kernels at import time, which fails with numba 0.68. This file
is the same algorithm, statement by statement, with cuda.local.array -> np.empty(float32) and the
CUDA grid replaced by two loops; argument order of the per-pair call (query box first) is kept.

Original header: Based on https://github.com/hongzhenwang/RRPN-revise, MIT License,
Author: yanyan, scrin@foxmail.com
"""
from __future__ import annotations

import math

import numba
import numpy as np


@numba.njit(cache=True)
def _trangle_area(a0, a1, b0, b1, c0, c1):
    return ((a0 - c0) * (b1 - c1) - (a1 - c1) * (b0 - c0)) / 2.0


@numba.njit(cache=True)
def _area(int_pts, num_of_inter):
    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(_trangle_area(int_pts[0], int_pts[1], int_pts[2 * i + 2], int_pts[2 * i + 3],
                                      int_pts[2 * i + 4], int_pts[2 * i + 5]))
    return area_val


@numba.njit(cache=True)
def _sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    if num_of_inter > 0:
        center = np.zeros(2, dtype=np.float32)
        for i in range(num_of_inter):
            center[0] += int_pts[2 * i]
            center[1] += int_pts[2 * i + 1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        v = np.empty(2, dtype=np.float32)
        vs = np.empty(16, dtype=np.float32)
        for i in range(num_of_inter):
            v[0] = int_pts[2 * i] - center[0]
            v[1] = int_pts[2 * i + 1] - center[1]
            d = math.sqrt(v[0] * v[0] + v[1] * v[1])
            v[0] = v[0] / d
            v[1] = v[1] / d
            if v[1] < 0:
                v[0] = -2 - v[0]
            vs[i] = v[0]
        for i in range(1, num_of_inter):
            if vs[i - 1] > vs[i]:
                temp = vs[i]
                tx = int_pts[2 * i]
                ty = int_pts[2 * i + 1]
                j = i
                while j > 0 and vs[j - 1] > temp:
                    vs[j] = vs[j - 1]
                    int_pts[j * 2] = int_pts[j * 2 - 2]
                    int_pts[j * 2 + 1] = int_pts[j * 2 - 1]
                    j -= 1
                vs[j] = temp
                int_pts[j * 2] = tx
                int_pts[j * 2 + 1] = ty


@numba.njit(cache=True)
def _line_segment_intersection(pts1, pts2, i, j, temp_pts):
    A = np.empty(2, dtype=np.float32)
    B = np.empty(2, dtype=np.float32)
    C = np.empty(2, dtype=np.float32)
    D = np.empty(2, dtype=np.float32)
    A[0] = pts1[2 * i]
    A[1] = pts1[2 * i + 1]
    B[0] = pts1[2 * ((i + 1) % 4)]
    B[1] = pts1[2 * ((i + 1) % 4) + 1]
    C[0] = pts2[2 * j]
    C[1] = pts2[2 * j + 1]
    D[0] = pts2[2 * ((j + 1) % 4)]
    D[1] = pts2[2 * ((j + 1) % 4) + 1]
    BA0 = B[0] - A[0]
    BA1 = B[1] - A[1]
    DA0 = D[0] - A[0]
    CA0 = C[0] - A[0]
    DA1 = D[1] - A[1]
    CA1 = C[1] - A[1]
    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        if abc != abd:
            DC0 = D[0] - C[0]
            DC1 = D[1] - C[1]
            ABBA = A[0] * B[1] - B[0] * A[1]
            CDDC = C[0] * D[1] - D[0] * C[1]
            DH = BA1 * DC0 - BA0 * DC1
            Dx = ABBA * DC0 - BA0 * CDDC
            Dy = ABBA * DC1 - BA1 * CDDC
            temp_pts[0] = Dx / DH
            temp_pts[1] = Dy / DH
            return True
    return False


@numba.njit(cache=True)
def _point_in_quadrilateral(pt_x, pt_y, corners):
    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]
    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]
    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]
    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1
    return abab >= abap >= 0 and adad >= adap >= 0


@numba.njit(cache=True)
def _quadrilateral_intersection(pts1, pts2, int_pts):
    num_of_inter = 0
    for i in range(4):
        if _point_in_quadrilateral(pts1[2 * i], pts1[2 * i + 1], pts2):
            int_pts[num_of_inter * 2] = pts1[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1]
            num_of_inter += 1
        if _point_in_quadrilateral(pts2[2 * i], pts2[2 * i + 1], pts1):
            int_pts[num_of_inter * 2] = pts2[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1]
            num_of_inter += 1
    temp_pts = np.empty(2, dtype=np.float32)
    for i in range(4):
        for j in range(4):
            if _line_segment_intersection(pts1, pts2, i, j, temp_pts):
                int_pts[num_of_inter * 2] = temp_pts[0]
                int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                num_of_inter += 1
    return num_of_inter


@numba.njit(cache=True)
def _rbbox_to_corners(corners, rbbox):
    # generate clockwise corners and rotate it clockwise
    angle = rbbox[4]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    x_d = rbbox[2]
    y_d = rbbox[3]
    corners_x = np.empty(4, dtype=np.float32)
    corners_y = np.empty(4, dtype=np.float32)
    corners_x[0] = -x_d / 2
    corners_x[1] = -x_d / 2
    corners_x[2] = x_d / 2
    corners_x[3] = x_d / 2
    corners_y[0] = -y_d / 2
    corners_y[1] = y_d / 2
    corners_y[2] = y_d / 2
    corners_y[3] = -y_d / 2
    for i in range(4):
        corners[2 * i] = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x
        corners[2 * i + 1] = -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y


@numba.njit(cache=True)
def _inter(rbbox1, rbbox2):
    corners1 = np.empty(8, dtype=np.float32)
    corners2 = np.empty(8, dtype=np.float32)
    intersection_corners = np.empty(16, dtype=np.float32)
    _rbbox_to_corners(corners1, rbbox1)
    _rbbox_to_corners(corners2, rbbox2)
    num_intersection = _quadrilateral_intersection(corners1, corners2, intersection_corners)
    _sort_vertex_in_convex_polygon(intersection_corners, num_intersection)
    return _area(intersection_corners, num_intersection)


@numba.njit(cache=True)
def _dev_rotate_iou_eval(rbox1, rbox2, criterion):
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    area_inter = _inter(rbox1, rbox2)
    if criterion == -1:
        return area_inter / (area1 + area2 - area_inter)
    elif criterion == 0:
        return area_inter / area1
    elif criterion == 1:
        return area_inter / area2
    else:
        return area_inter


@numba.njit(cache=True)
def _rotate_iou_all(boxes, query_boxes, iou, criterion):
    for n in range(boxes.shape[0]):
        for k in range(query_boxes.shape[0]):
            # the CUDA kernel calls devRotateIoUEval(query box, box)
            iou[n, k] = _dev_rotate_iou_eval(query_boxes[k], boxes[n], criterion)


def rotate_iou_cpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """Same contract as rotate_iou.rotate_iou_gpu_eval: (N, 5) x (K, 5) camera-BEV boxes
    [x, z, l, w, ry] -> (N, K) in the dtype of `boxes`. `device_id` is accepted and ignored."""
    box_dtype = boxes.dtype
    boxes = np.ascontiguousarray(boxes, dtype=np.float32)
    query_boxes = np.ascontiguousarray(query_boxes, dtype=np.float32)
    iou = np.zeros((boxes.shape[0], query_boxes.shape[0]), dtype=np.float32)
    if iou.size:
        _rotate_iou_all(boxes, query_boxes, iou, criterion)
    return iou.astype(box_dtype)
