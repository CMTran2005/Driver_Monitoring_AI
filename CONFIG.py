"""
CURRENT CONFIGURATION SUMMARY
Updated: 2026-04-12
"""

# ============================================================
# 1. EYE DETECTION (Mắt)
# ============================================================
brightness_threshold = 102  # if brightness < 102: CLOSED

eye_region_height = 0.35     # Vùng scan: từ 0 đến 35% chiều cao
eye_region_width = 0.1       # Từ 10% đến 90% chiều rộng

eye_threshold_frames = 8     # Cần 8 frames liên tục mới báo
eye_confidence_min = 0.5     # Độ tin cậy tối thiểu

# ============================================================
# 2. YAWN DETECTION (Ngáp)
# ============================================================
mouth_roi = {
    'y_start': 0.72,  # 72% chiều cao
    'y_end': 0.82,    # 82% chiều cao
    'x_start': 0.28,  # 28% chiều rộng
    'x_end': 0.72,    # 72% chiều rộng
}

yawn_threshold_frames = 20   # Cần 20 frames liên tục mới báo
yawn_confidence_min = 0.6    # Độ tin cậy tối thiểu

# ============================================================
# 3. MODEL ACCURACY (từ test set)
# ============================================================
# Eye Model:
#   Accuracy: 90.69%
#   Precision: 89.36%
#   Recall: 94.03%
#   F1-Score: 91.64%

# Yawn Model:
#   Accuracy: 95.14%
#   Precision: 96.12%
#   Recall: 94.66%
#   F1-Score: 95.38%

# ============================================================
# 4. HOW TO CALIBRATE
# ============================================================
# Run: python calibrate_eyes.py
# - Record 10 samples khi MỞ MẮTO ra
# - Record 10 samples khi NHẮM MẮT
# - Script sẽ suggest threshold mới

# ============================================================
# 5. HOW TO DEBUG
# ============================================================
# Run: python diagnostic.py
# - XANH LÒNG = Mắt detect (eye cascade found)
# - CYAN = Fallback brightness check
# - VÀNG = Vùng miệng (mouth ROI)

# ============================================================
# 6. ADJUST PARAMETERS
# ============================================================
# File: test_inference.py

# If eye detection too sensitive (false positives):
#   - Increase eye_threshold_frames: 8 → 12
#   - Increase eye_confidence_min: 0.5 → 0.7
#   - Adjust brightness_threshold: < 102 → < 95

# If yawn detection too sensitive:
#   - Increase yawn_threshold_frames: 20 → 30
#   - Increase yawn_confidence_min: 0.6 → 0.8
#   - Adjust mouth ROI coordinates

# If mouth region captures nose:
#   - Increase mouth_y_start: 0.72 → 0.75
#   - Decrease mouth_x_start: 0.28 → 0.30

print("✓ Configuration loaded successfully")
