"""Derived metrics from pose keypoints."""
import math
import numpy as np


def _angle(a, b, c):
    """Angle at point b (in degrees) given three [x,y] points."""
    if a is None or b is None or c is None:
        return None
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return None
    cos_angle = max(-1, min(1, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def _pt(keypoints, idx):
    """Extract valid point or None."""
    kp = keypoints[idx]
    if kp[2] < 0.3:
        return None
    return [kp[0], kp[1]]


def compute_analytics(pose_data):
    """Compute time-series analytics from pose estimation results.

    Args:
        pose_data: dict with 'frames' list, each having 'time' and 'keypoints'

    Returns:
        dict with time-series arrays for each metric
    """
    frames = pose_data.get('frames', [])
    if not frames:
        return {}

    times = []
    left_knee_angle = []
    right_knee_angle = []
    left_hip_angle = []
    right_hip_angle = []
    left_elbow_angle = []
    right_elbow_angle = []
    hip_center_y = []
    left_ankle_y = []
    right_ankle_y = []

    for fr in frames:
        kps = fr['keypoints']
        t = fr['time']
        times.append(t)

        # Joint angles
        # Knee: hip-knee-ankle
        left_knee_angle.append(_angle(_pt(kps, 11), _pt(kps, 13), _pt(kps, 15)))
        right_knee_angle.append(_angle(_pt(kps, 12), _pt(kps, 14), _pt(kps, 16)))
        # Hip: shoulder-hip-knee
        left_hip_angle.append(_angle(_pt(kps, 5), _pt(kps, 11), _pt(kps, 13)))
        right_hip_angle.append(_angle(_pt(kps, 6), _pt(kps, 12), _pt(kps, 14)))
        # Elbow: shoulder-elbow-wrist
        left_elbow_angle.append(_angle(_pt(kps, 5), _pt(kps, 7), _pt(kps, 9)))
        right_elbow_angle.append(_angle(_pt(kps, 6), _pt(kps, 8), _pt(kps, 10)))

        # Vertical positions
        lh = _pt(kps, 11)
        rh = _pt(kps, 12)
        if lh and rh:
            hip_center_y.append((lh[1] + rh[1]) / 2)
        elif lh:
            hip_center_y.append(lh[1])
        elif rh:
            hip_center_y.append(rh[1])
        else:
            hip_center_y.append(None)

        la = _pt(kps, 15)
        ra = _pt(kps, 16)
        left_ankle_y.append(la[1] if la else None)
        right_ankle_y.append(ra[1] if ra else None)

    # Compute velocities (pixels/second between consecutive frames)
    def velocity(positions, times):
        vel = [None]
        for i in range(1, len(positions)):
            if positions[i] is not None and positions[i-1] is not None and times[i] != times[i-1]:
                dt = times[i] - times[i-1]
                vel.append(abs(positions[i] - positions[i-1]) / dt)
            else:
                vel.append(None)
        return vel

    # Ankle velocity as proxy for movement speed
    def joint_speed(kps_list, idx, times):
        """Speed of a joint (euclidean distance / dt)."""
        vel = [None]
        for i in range(1, len(kps_list)):
            p0 = _pt(kps_list[i-1], idx)
            p1 = _pt(kps_list[i], idx)
            dt = times[i] - times[i-1] if times[i] != times[i-1] else 1
            if p0 and p1:
                dist = math.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)
                vel.append(dist / dt)
            else:
                vel.append(None)
        return vel

    all_kps = [fr['keypoints'] for fr in frames]

    return {
        'times': times,
        'left_knee_angle': left_knee_angle,
        'right_knee_angle': right_knee_angle,
        'left_hip_angle': left_hip_angle,
        'right_hip_angle': right_hip_angle,
        'left_elbow_angle': left_elbow_angle,
        'right_elbow_angle': right_elbow_angle,
        'hip_center_y': hip_center_y,
        'hip_vertical_velocity': velocity(hip_center_y, times),
        'left_ankle_speed': joint_speed(all_kps, 15, times),
        'right_ankle_speed': joint_speed(all_kps, 16, times),
        'left_wrist_speed': joint_speed(all_kps, 9, times),
        'right_wrist_speed': joint_speed(all_kps, 10, times),
    }
