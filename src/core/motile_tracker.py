#!/usr/bin/env python3

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from numba import njit

# Numba-optimized helper functions (defined at module level to avoid re-compilation)
@njit
def _greedy_numba(trk, det, n_trk, n_det):
    """
    Numba-optimized greedy assignment algorithm.
    
    Args:
        trk: Array of track indices
        det: Array of detection indices  
        n_trk: Total number of tracks
        n_det: Total number of detections
        
    Returns:
        Tuple of (track_indices, detection_indices) for valid assignments
    """
    used_trk = np.zeros(n_trk, np.bool_)
    used_det = np.zeros(n_det, np.bool_)
    ti, dj = [], []
    for t, d in zip(trk, det):
        if not used_trk[t] and not used_det[d]:
            ti.append(t)
            dj.append(d)
            used_trk[t] = True
            used_det[d] = True
    return np.array(ti, dtype=np.intp), np.array(dj, dtype=np.intp)


@dataclass
class Detection:
    frame: int
    x: float
    y: float
    area: float
    curvature: Optional[np.ndarray] = None


@dataclass
class TrackPoint:
    frame: int
    x: float
    y: float
    score: float


@dataclass
class TrackerConfig:
    max_age: int = 5
    min_hits: int = 3

    max_distance: float = 30.0
    theta_hard: float = np.deg2rad(120)
    sigma_theta: float = np.deg2rad(45)

    gamma_tau: float = 1.2

    w_dir: float = 0.35
    w_dist: float = 0.30
    w_speed: float = 0.20
    w_morph: float = 0.15

    use_tau: bool = True
    use_kf: bool = False

    assignment_mode: str = "greedy"
    history_len: int = 2

    min_edge_frames: int = 3
    edge_spawn_threshold: float = 0.1
    img_width: float = 1200.0
    img_height: float = 1200.0

    track_history_file: Optional[str] = None


@dataclass
class TrajectoryPostConfig:
    min_track_length: int = 10

    min_hits: int = 3

    min_confidence: float = 0.3


@dataclass
class PipelineConfig:
    tracker: TrackerConfig
    post: TrajectoryPostConfig
    img_shape: Tuple[int, int]


class TrackState:
    ACTIVE = "active"
    INACTIVE = "inactive"  # finished (lost, exited, occluded)


class Track:
    def __init__(self, track_id: int, det: Detection):
        self.id = track_id
        self.history: List[TrackPoint] = []
        self.missing: int = 0
        self.state: str = TrackState.ACTIVE

        self.dir_angles: List[float] = []
        self.assoc_scores: List[float] = []

        self.lifecycle_events: List[Dict] = []
        self.spawn_frame: int = det.frame
        self.death_frame: Optional[int] = None
        self.termination_reason: Optional[str] = None
        self.assignment_history: List[Dict] = []

        self.was_updated_this_frame: bool = False

        self.append(det, score=1.0, frame=det.frame)

        if hasattr(self, "lifecycle_events"):
            event = {
                "event_type": "spawn",
                "frame": int(det.frame),
                "track_id": int(self.id),
                "position": (float(det.x), float(det.y)),
                "reason": "new_detection",
                "timestamp": 0,
            }
            self.lifecycle_events.append(event)

    def last_position(self) -> np.ndarray:
        p = self.history[-1]
        return np.array([p.x, p.y])

    def prev_direction(self, cfg: TrackerConfig) -> Optional[np.ndarray]:
        if len(self.history) < 2:
            return None

        k = min(cfg.history_len, len(self.history) - 1)
        if k < 1:
            return None

        p_prev = self.history[-(k + 1)]
        p_curr = self.history[-1]

        v = np.array([p_curr.x - p_prev.x, p_curr.y - p_prev.y])
        if np.linalg.norm(v) < 1e-6:
            return None
        return v

    def prev_speed(self, cfg: TrackerConfig) -> Optional[float]:
        v = self.prev_direction(cfg)
        if v is None:
            return None
        k = min(cfg.history_len, len(self.history) - 1)
        if k < 1:
            return None
        return np.linalg.norm(v) / k

    def tau_predict(self, cfg: TrackerConfig) -> np.ndarray:
        p = self.last_position()
        v = self.prev_direction(cfg)

        if v is None:
            return p

        u = v / np.linalg.norm(v)

        s = self.prev_speed(cfg) or np.linalg.norm(v)

        l_tau = min(np.linalg.norm(v), cfg.gamma_tau * s, cfg.max_distance)

        result = p + l_tau * u

        return result

    def append(self, det: Detection, score: float, frame: Optional[int] = None):
        self.history.append(TrackPoint(det.frame, det.x, det.y, score))
        self.assoc_scores.append(score)
        self.missing = 0
        self.was_updated_this_frame = True

        if hasattr(self, "lifecycle_events"):
            event = {
                "event_type": "update",
                "frame": int(frame) if frame is not None else int(det.frame),
                "position": (float(det.x), float(det.y)),
                "score": float(score),
                "track_length": int(len(self.history)),
                "timestamp": int(len(self.lifecycle_events)),
            }
            self.lifecycle_events.append(event)

    def mark_missing(self, frame: Optional[int] = None):
        self.missing += 1

        if hasattr(self, "lifecycle_events"):
            event = {
                "event_type": "missing",
                "frame": int(frame) if frame is not None else None,
                "missing_count": int(self.missing),
                "timestamp": int(len(self.lifecycle_events)),
            }
            self.lifecycle_events.append(event)

    def confidence(self) -> float:
        if not self.assoc_scores:
            return 0.0

        mean_score = float(np.mean(self.assoc_scores))

        if self.dir_angles:
            mean_theta = float(np.mean(self.dir_angles))
            dir_stability = np.exp(-(mean_theta / np.deg2rad(45)))
        else:
            dir_stability = 1.0

        confidence = mean_score * dir_stability

        return confidence


class DirectionFirstTracker:
    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.tracks: List[Track] = []
        self.next_id: int = 0

    def update(self, detections: List[Detection], frame: int):
        for trk in self.tracks:
            if trk.state == TrackState.ACTIVE:
                trk.was_updated_this_frame = False

        if not self.tracks:
            for d in detections:
                self._spawn(d, frame, "no_existing_tracks")

            return

        C = self._build_cost_matrix(detections)

        if C.size and np.any(np.isfinite(C)):
            if self.cfg.assignment_mode == "greedy":
                trk_idx, det_idx = self._greedy_assignment(C)
            else:
                try:
                    trk_idx, det_idx = linear_sum_assignment(C)
                except ValueError as _e:
                    pass

                    trk_idx, det_idx = np.array([], dtype=int), np.array([], dtype=int)
        else:
            trk_idx, det_idx = np.array([], dtype=int), np.array([], dtype=int)

        assigned_trk = set()
        assigned_det = set()

        for i, j in zip(trk_idx, det_idx):
            if not np.isfinite(C[i, j]):
                continue

            trk = self.tracks[i]
            det = detections[j]

            if not self._post_validate(trk, det):
                continue

            score = 1.0 - C[i, j]
            trk.append(det, score, frame)

            assigned_trk.add(i)
            assigned_det.add(j)

        missing_count = 0
        for i, trk in enumerate(self.tracks):
            if not trk.was_updated_this_frame:
                trk.mark_missing(frame)
                missing_count += 1

        spawned_count = 0
        rejected_count = 0
        for j, det in enumerate(detections):
            if j not in assigned_det:
                if frame >= self.cfg.min_edge_frames:
                    x_ratio = det.x / self.cfg.img_width
                    y_ratio = det.y / self.cfg.img_height
                    edge_threshold = self.cfg.edge_spawn_threshold

                    near_edge = (
                        x_ratio <= edge_threshold
                        or x_ratio >= 1 - edge_threshold
                        or y_ratio <= edge_threshold
                        or y_ratio >= 1 - edge_threshold
                    )

                    if near_edge:
                        self._spawn(det, frame, "edge_detection")
                        spawned_count += 1
                    else:
                        has_nearby_track = False
                        for trk in self.tracks:
                            if (
                                np.linalg.norm(
                                    trk.last_position() - np.array([det.x, det.y])
                                )
                                < self.cfg.max_distance
                            ):
                                has_nearby_track = True
                                break
                        if not has_nearby_track:
                            self._spawn(det, frame, "internal_unmatched")
                            spawned_count += 1
                        else:
                            rejected_count += 1
                else:
                    self._spawn(det, frame, "early_frame_unfiltered")
                    spawned_count += 1

        dead_tracks = []

        for t in self.tracks:
            max_allowed_age = (
                self.cfg.max_age
                if len(t.history) < self.cfg.min_hits
                else self.cfg.max_age
            )
            if t.missing > max_allowed_age:
                t.death_frame = frame
                t.termination_reason = "max_age_exceeded"
                t.state = TrackState.INACTIVE
                dead_tracks.append(t)

    def _greedy_assignment(self, C: np.ndarray):
        """
        Greedy assignment using module-level numba-optimized helper function.
        This avoids re-compilation on every method call.
        """
        n_trk, n_det = C.shape
        valid_mask = np.isfinite(C)
        trk, det = np.where(valid_mask)
        cost = C[valid_mask]
        order = np.argsort(cost)
        return _greedy_numba(trk[order], det[order], n_trk, n_det)

    def _build_cost_matrix(self, detections: List[Detection]) -> np.ndarray:
        n_trk = len(self.tracks)
        n_det = len(detections)

        C = np.full((n_trk, n_det), np.inf)

        if detections:
            detection_positions = np.array([[det.x, det.y] for det in detections])
            detection_tree = cKDTree(detection_positions)
        else:
            return C

        for i, trk in enumerate(self.tracks):
            if trk.state != TrackState.ACTIVE:
                continue
            p = trk.last_position()
            v_prev = trk.prev_direction(self.cfg)

            r = trk.tau_predict(self.cfg) if self.cfg.use_tau else p

            nearby_indices = detection_tree.query_ball_point(p, self.cfg.max_distance)

            for j_idx in nearby_indices:
                j = j_idx
                det = detections[j]
                d = detection_positions[j]

                dist = np.linalg.norm(r - d)

                if v_prev is not None:
                    v_ij = d - p
                    cos_theta_ij = np.dot(v_prev, v_ij) / (
                        np.linalg.norm(v_prev) * np.linalg.norm(v_ij) + 1e-9
                    )
                    cos_theta_hard = np.cos(self.cfg.theta_hard)

                    if cos_theta_ij < cos_theta_hard:
                        continue
                    else:
                        pass

                    theta = np.arccos(np.clip(cos_theta_ij, -1, 1))
                else:
                    theta = 0.0

                W_dir = np.exp(-((theta / self.cfg.sigma_theta) ** 2))

                W_dist = np.exp(-((dist / (self.cfg.max_distance / 2)) ** 2))

                s_prev = trk.prev_speed(self.cfg)
                if s_prev is not None:
                    W_speed = np.exp(-(((dist - s_prev) / self.cfg.max_distance) ** 2))
                else:
                    W_speed = 1.0
                    pass

                W_morph = 1.0
                if det.curvature is not None:
                    W_morph = 1.0
                else:
                    W_morph = 1.0

                S = (
                    self.cfg.w_dir * W_dir
                    + self.cfg.w_dist * W_dist
                    + self.cfg.w_speed * W_speed
                    + self.cfg.w_morph * W_morph
                )

                C[i, j] = 1.0 - S

        return C

    def _post_validate(self, trk: Track, det: Detection) -> bool:
        p = trk.last_position()
        d = np.array([det.x, det.y])

        if np.linalg.norm(d - p) > self.cfg.max_distance:
            return False

        dist = np.linalg.norm(d - p)

        if dist > self.cfg.max_distance:
            return False

        v_prev = trk.prev_direction(self.cfg)
        if v_prev is not None:
            v_ij = d - p
            cosang = np.dot(v_prev, v_ij) / (
                np.linalg.norm(v_prev) * np.linalg.norm(v_ij) + 1e-9
            )
            theta = np.arccos(np.clip(cosang, -1, 1))
            if theta > self.cfg.theta_hard:
                return False
            trk.dir_angles.append(theta)
        else:
            pass

        return True

    def _spawn(self, det: Detection, frame: int, reason: str = "new_detection"):
        new_track = Track(self.next_id, det)
        self.tracks.append(new_track)

        pass

        self.next_id += 1


def run_tracking_with_configs(
    detections_df: pd.DataFrame,
    total_frames: int,
    img_shape: Tuple[int, int],
    tracker_cfg: TrackerConfig,
    post_cfg: TrajectoryPostConfig,
) -> pd.DataFrame:
    tracker_cfg.img_width = float(img_shape[1])
    tracker_cfg.img_height = float(img_shape[0])

    pipeline_cfg = PipelineConfig(
        tracker=tracker_cfg, post=post_cfg, img_shape=img_shape
    )

    tracker = DirectionFirstTracker(pipeline_cfg.tracker)

    if not detections_df.empty:
        frames = sorted(detections_df["frame"].unique())
    else:
        frames = []

    for i, f in enumerate(frames):
        dets = [
            Detection(
                frame=row["frame"],
                x=row["x"],
                y=row["y"],
                area=row["area"],
                curvature=getattr(row, "curvature", None),
            )
            for _, row in detections_df[detections_df["frame"] == f].iterrows()
        ]
        tracker.update(dets, f)

    rows = []
    for trk in tracker.tracks:
        for i, p in enumerate(trk.history):
            rows.append(
                {
                    "track_id": trk.id,
                    "frame": p.frame,
                    "x": p.x,
                    "y": p.y,
                    "accumulated_length": i + 1,
                    "track_length": len(trk.history),
                    "confidence": trk.confidence(),
                }
            )

    return pd.DataFrame(rows)
