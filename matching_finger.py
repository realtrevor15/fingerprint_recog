import json
import numpy as np
import cv2
from skimage.measure import ransac
from skimage.transform import AffineTransform

def load_minutiae_from_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def match_minutiae_ransac(query_minutiae, db_minutiae):
    """
    So khớp minutiae giữa ảnh đầu vào và một ảnh trong database bằng RANSAC.
    :param query_minutiae: Danh sách minutiae của ảnh đầu vào (list of dicts)
    :param db_minutiae: Danh sách minutiae của ảnh trong database (list of dicts)
    :return: Số lượng minutiae khớp nhau sau khi dùng RANSAC
    """
    if len(query_minutiae) < 3 or len(db_minutiae) < 3:
        return 0  # Không đủ điểm để so khớp

    # Chuyển minutiae thành dạng numpy array [x, y]
    src_pts = np.array([(m["x"], m["y"]) for m in query_minutiae])
    dst_pts = np.array([(m["x"], m["y"]) for m in db_minutiae])

    # Áp dụng thuật toán RANSAC để tìm sự tương đồng
    model, inliers = ransac((src_pts, dst_pts), AffineTransform, min_samples=3, residual_threshold=10, max_trials=1000)
    
    # Số lượng điểm khớp nhau
    return np.sum(inliers) if inliers is not None else 0
