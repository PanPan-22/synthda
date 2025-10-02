import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
from .optimisation_utils import map_h36m_to_smpl, upsample_pose_data, center_and_rotate_smpl, compute_P_opt

# real_pose_path refers to the Generated Tracked-Motion Data
# synthetic_pose_path refers to the Generated Synthetic-Motion Data

# used when one is real and one is synth
def main_synth_real(real_path_npz, synthetic_path, folder_path):

    # print(folder_path_variations)
    folder_path_variations = Path(folder_path) / "all_variations"
    folder_path_variations.mkdir(parents=True, exist_ok=True)

    # path to save the real npy file after it has been processed
    real_path_npy = folder_path + '/output_keypoints_3d.npy'

    # change to .npz when real videos are used
    synthetic_path_flipped = synthetic_path.replace(".npy", "_flip.npy")

    # mapping the npz to npy file with 22 joints
    real_data = map_h36m_to_smpl(real_path_npz)
    np.save(real_path_npy, real_data)

    # creating a rotated version of the synthetic
    center_and_rotate_smpl(synthetic_path, synthetic_path_flipped)

    # checking which frames are shorter, to match the frames to be the same
    # Load the real and synthetic pose data from .npy files
    P_r = np.load(real_path_npy)  # Shape: (J, 3)
    P_s = np.load(synthetic_path_flipped)  # Shape: (J, 3)

    # Determine the smaller array
    if P_r.shape[0] > P_s.shape[0]:
        smaller_array = synthetic_path_flipped
        bigger_array = real_path_npy
        max_frames = P_r.shape[0]
    else:
        smaller_array = real_path_npy
        bigger_array = synthetic_path_flipped
        max_frames = P_s.shape[0]

    # upsampling the smaller array to the number of frames
    pose_data_upsampled = upsample_pose_data(smaller_array, target_frames=max_frames)

    extended_new_path = smaller_array.replace(".npy", "_extended.npy")

    # saving the new array
    np.save(extended_new_path, pose_data_upsampled)
    print(pose_data_upsampled.shape)

    # generating the all variations of optimisation files
    weight_pairs = [
        (0.1, 0.9),
        (0.2, 0.8),
        (0.3, 0.7),
        (0.4, 0.6),
        (0.5, 0.5),
        (0.6, 0.4),
        (0.7, 0.3),
        (0.8, 0.2),
        (0.9, 0.1)
    ]

    alpha_weight=0.3
            
    if bigger_array == real_path_npy:
        for w_A, w_B in weight_pairs:
            P_opt = compute_P_opt(real_path_npy, extended_new_path, alpha=alpha_weight, w_A=w_A, w_B=w_B)
            np.save(f"{folder_path_variations}/_euclidean_distances_wA{w_A}_wB{w_B}.npy", P_opt)

    # in a situation where the real_path is shorter than the synthetic path
    if smaller_array == real_path_npy:
        for w_A, w_B in weight_pairs:
            P_opt = compute_P_opt(extended_new_path, synthetic_path_flipped, alpha=alpha_weight, w_A=w_A, w_B=w_B) 
            np.save(f"{folder_path_variations}/_euclidean_distances_wA{w_A}_wB{w_B}.npy", P_opt)
