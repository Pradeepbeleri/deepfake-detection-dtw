"""Landmark-based mouth crops; missing or ambiguous frames are never synthesized."""
import numpy as np
import torch
import torch.nn.functional as F

from .preprocessing import Prepared

MOUTH = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 13, 0, 37, 39, 40, 185, 267, 269, 270, 409)


def _iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def crop_mouths(video: Prepared, detector) -> tuple[Prepared | None, dict]:
    """Detector returns a list of landmark arrays, normalized x/y, per RGB frame.

    A single-face IoU check prevents silently switching identity. Missing frames are
    skipped; after >3 missing frames tracking is lost and later detections rejected.
    """
    crops, times, flags = [], [], []
    previous = None
    missing = 0
    lost = False
    for frame, timestamp in zip(video.values, video.timestamps):
        rgb = (frame.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
        faces = detector(rgb)
        reason = None
        if len(faces) != 1:
            reason = 'no_face' if not faces else 'multiple_faces'
        else:
            pts = np.asarray(faces[0], dtype=float)
            if pts.ndim != 2 or pts.shape[0] <= max(MOUTH) or pts.shape[1] < 2 or not np.isfinite(pts[:, :2]).all():
                reason = 'invalid_landmarks'
            else:
                xy = pts[:, :2]
                box = [float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())]
                mouth = xy[list(MOUTH)]
                x0, y0 = mouth.min(axis=0); x1, y1 = mouth.max(axis=0)
                pad = max(x1-x0, y1-y0) * .35
                x0, y0 = np.floor(np.maximum([x0-pad, y0-pad], 0) * 224).astype(int)
                x1, y1 = np.ceil(np.minimum([x1+pad, y1+pad], 1) * 224).astype(int)
                if lost or (previous is not None and _iou(previous, box) < .2):
                    reason = 'tracking_lost'
                    lost = True
                elif x1-x0 < 4 or y1-y0 < 4:
                    reason = 'invalid_mouth_crop'
                else:
                    crop = frame[:, y0:y1, x0:x1].unsqueeze(0)
                    crops.append(F.interpolate(crop, (224, 224), mode='bilinear', align_corners=False, antialias=True)[0])
                    times.append(timestamp)
                    previous = box
        if reason:
            missing += 1
            if previous is not None and missing > 3:
                lost = True
            flags.append({'timestamp_seconds': float(timestamp), 'status': reason})
        else:
            flags.append({'timestamp_seconds': float(timestamp), 'status': 'detected' if missing == 0 else 'detected_after_gap'})
            missing = 0
    report = {'frames_total': len(flags), 'mouths_detected': len(crops), 'detection_fraction': len(crops) / len(flags) if flags else 0, 'frames': flags, 'policy': 'independent detection; IoU continuity; gaps skipped, never fabricated; tracking loss requires new clip'}
    if not crops:
        return None, report
    return Prepared(torch.stack(crops), torch.stack(times), {**video.metadata, 'crop': 'MediaPipe lip-landmark bounds with 35% padding; resized 224x224', 'quality': report}), report


class FaceMeshDetector:
    def __init__(self):
        import mediapipe as mp
        self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2, refine_landmarks=False, min_detection_confidence=0.5)

    def __call__(self, rgb):
        result = self.mesh.process(rgb)
        return [np.array([(p.x, p.y) for p in face.landmark]) for face in (result.multi_face_landmarks or [])]

    def close(self):
        self.mesh.close()
