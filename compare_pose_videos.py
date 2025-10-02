# compare_pose_videos.py
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

mp_pose = mp.solutions.pose

# MediaPipe Pose 33 landmark names in index order
POSE_LANDMARKS = [
    "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right","left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb",
    "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle","left_heel","right_heel",
    "left_foot_index","right_foot_index"
]

def extract_keypoints(video_path, model_complexity=1, smooth=True):
    """
    Returns DataFrame with columns:
    frame, joint, x, y, z, visibility, img_w, img_h
    x,y,z are normalized to [0,1] as per MediaPipe; we keep img size for optional denorm.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=smooth,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            for j, lm in enumerate(res.pose_landmarks.landmark):
                rows.append({
                    "frame": frame_idx,
                    "joint": POSE_LANDMARKS[j] if j < len(POSE_LANDMARKS) else f"joint_{j}",
                    "x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility,
                    "img_w": width, "img_h": height
                })
        else:
            # Fill NaNs for missing poses to keep frame alignment
            for j in range(len(POSE_LANDMARKS)):
                rows.append({
                    "frame": frame_idx,
                    "joint": POSE_LANDMARKS[j],
                    "x": np.nan, "y": np.nan, "z": np.nan, "visibility": np.nan,
                    "img_w": width, "img_h": height
                })
        frame_idx += 1

    cap.release()
    pose.close()
    df = pd.DataFrame(rows)
    print(df.value_counts(dropna=False))
    return df

def normalize(df, method="hips"):
    """
    Coordinate normalization for comparability between videos:
    - method='image': use image width/height (already normalized, but clamps and keeps NaNs)
    - method='hips': recenter to pelvis midpoint and scale by hip distance
    """
    df = df.copy()
    # Clamp x,y into [0,1] in case of small drift; keep NaN as NaN
    df["x"] = df["x"].clip(0, 1)
    df["y"] = df["y"].clip(0, 1)

    if method == "image":
        df["xn"], df["yn"] = df["x"], df["y"]
        return df

    # Build per-frame arrays for pelvis-centered normalization
    pivot_left = "left_hip"
    pivot_right = "right_hip"

    def norm_frame(group):
        # group is rows of a single frame
        try:
            lx = float(group.loc[group["joint"]==pivot_left, "x"].values[0])
            ly = float(group.loc[group["joint"]==pivot_left, "y"].values[0])
            rx = float(group.loc[group["joint"]==pivot_right, "x"].values[0])
            ry = float(group.loc[group["joint"]==pivot_right, "y"].values[0])
        except Exception:
            group["xn"], group["yn"] = np.nan, np.nan
            return group

        if np.isnan([lx,ly,rx,ry]).any():
            group["xn"], group["yn"] = np.nan, np.nan
            return group

        cx, cy = (lx + rx) / 2.0, (ly + ry) / 2.0
        hip_dist = np.hypot(rx - lx, ry - ly)
        if hip_dist == 0:
            hip_dist = 1e-6

        group["xn"] = (group["x"] - cx) / hip_dist
        group["yn"] = (group["y"] - cy) / hip_dist
        return group

    df = df.groupby("frame", group_keys=False).apply(norm_frame)
    return df

def save_csv(df, out_csv):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

def align_frames(dfA, dfB):
    # Keep only frames present in both (inner join)
    common = sorted(set(dfA["frame"]).intersection(set(dfB["frame"])))
    return (
        dfA[dfA["frame"].isin(common)].reset_index(drop=True),
        dfB[dfB["frame"].isin(common)].reset_index(drop=True)
    )

def plot_joint_timeseries(dfA, dfB, joint="right_wrist", labelA="video_A", labelB="video_B"):
    A = dfA[dfA["joint"]==joint].sort_values("frame")
    B = dfB[dfB["joint"]==joint].sort_values("frame")

    frames = A["frame"].values
    plt.figure(figsize=(10,4))
    plt.plot(frames, A["xn"], linewidth=1.2, label=f"{labelA} x")
    plt.plot(frames, A["yn"], linewidth=1.2, label=f"{labelA} y")
    plt.plot(frames, B["xn"], linewidth=1.2, label=f"{labelB} x", linestyle="--")
    plt.plot(frames, B["yn"], linewidth=1.2, label=f"{labelB} y", linestyle="--")
    plt.title(f"{joint} normalized x/y over frames")
    plt.xlabel("frame")
    plt.ylabel("normalized coord")
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig("joint_timeseries")

def compute_displacement(dfA, dfB):
    """Return DataFrame with per-frame per-joint displacement between A and B in normalized space."""
    key = ["frame","joint"]
    merged = pd.merge(
        dfA[key+["xn","yn"]],
        dfB[key+["xn","yn"]],
        on=key, how="inner", suffixes=("_A","_B")
    )
    merged["disp"] = np.hypot(merged["xn_A"]-merged["xn_B"], merged["yn_A"]-merged["yn_B"])
    return merged

def plot_displacement_timeseries(disp_df, joint="right_wrist"):
    d = disp_df[disp_df["joint"]==joint].sort_values("frame")
    plt.figure(figsize=(10,4))
    plt.plot(d["frame"], d["disp"], linewidth=1.2)
    plt.title(f"Per-frame displacement: {joint}")
    plt.xlabel("frame")
    plt.ylabel("Δ (normalized units)")
    plt.tight_layout()
    #plt.show()
    plt.savefig("displacement_timeseries")

def plot_avg_displacement_bar(disp_df, top_k=15):
    avg = disp_df.groupby("joint", as_index=False)["disp"].mean().sort_values("disp", ascending=False)
    if top_k:
        avg = avg.head(top_k)
    plt.figure(figsize=(10,5))
    plt.bar(avg["joint"], avg["disp"])
    plt.title("Average displacement per joint (A vs B)")
    plt.ylabel("mean Δ (normalized)")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    #plt.show()
    plt.savefig("plot_avg_dispalcement_bar")
    return avg

def main(video_a, video_b, out_dir="pose_out", norm_method="hips",
         labelA="weight_1", labelB="weight_2", example_joint="right_wrist"):
    out_dir = Path(out_dir)

    # 1) Extract
    dfA = extract_keypoints(video_a)
    dfB = extract_keypoints(video_b)

    # 2) Normalize
    dfA = normalize(dfA, method=norm_method)
    dfB = normalize(dfB, method=norm_method)

    # 3) Align frames
    dfA, dfB = align_frames(dfA, dfB)

    # 4) Save CSVs
    save_csv(dfA, out_dir / f"{labelA}_pose.csv")
    save_csv(dfB, out_dir / f"{labelB}_pose.csv")

    # 5) Visualizations
    plot_joint_timeseries(dfA, dfB, joint=example_joint, labelA=labelA, labelB=labelB)

    disp_df = compute_displacement(dfA, dfB)
    plot_displacement_timeseries(disp_df, joint=example_joint)
    avg_table = plot_avg_displacement_bar(disp_df, top_k=20)

    # Also save the summary table
    avg_table.to_csv(out_dir / "avg_displacement_per_joint.csv", index=False)
    print("Top joints by average Δ:")
    print(avg_table.head(10))

if __name__ == "__main__":
    # Example:
    # python compare_pose_videos.py
    # Then edit the paths below or pass via argparse if you prefer.
    #video_a_path = "components/dataset/videos_generated_real2_0.5/test_video_1_test_video_2_euclidean_distances_wA0.5_wB0.5_front_video(3).mp4"
    video_a_path ="components/dataset/comparison_input/test_video_1.mp4"
    video_b_path ="components/dataset/comparison_input/test_video_2.mp4"
    #video_b_path = "components/dataset/videos_generated_real2_0.5/test_video_1_test_video_2_euclidean_distances_wA0.5_wB0.5_front_video(4).mp4"
    main(video_a_path, video_b_path,
         out_dir="pose_out",
         norm_method="hips",
         labelA="w0_2", labelB="w0_9",
         example_joint="right_wrist")
    
#try real videos
# possibly because generated video quality?
#
