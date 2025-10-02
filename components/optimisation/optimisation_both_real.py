import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
from .optimisation_utils import map_h36m_to_smpl, upsample_pose_data, compute_P_opt

# real_path_npz_1 refers to the First Generated Tracked-Motion Data
# real_path_npz_2 refers to the Second Generated Tracked-Motion Data

# used when one is real and another is real too
def main_real_real(real_path_npz_1, real_path_npz_2, folder_path):

    # folder_path_variations = folder_path + "all_variations/"
    # print(folder_path_variations)
    folder_path_variations = Path(folder_path) / "all_variations"
    folder_path_variations.mkdir(parents=True, exist_ok=True)

    # converting npz to npy
    real_data_1_path = folder_path + '/output_keypoints_3d_real1.npy'
    real_data_2_path = folder_path + '/output_keypoints_3d_real2.npy'

    print(real_data_1_path)
    print(real_data_2_path)
    
    # mapping the npz to npy file with 22 joints
    real_data_1 = map_h36m_to_smpl(real_path_npz_1)
    np.save(real_data_1_path, real_data_1)

    # mapping the npz to npy file with 22 joints
    real_data_2 = map_h36m_to_smpl(real_path_npz_2)
    np.save(real_data_2_path, real_data_2)

    # checking which frames are shorter, to match the frames to be the same
    # Load the real and synthetic pose data from .npy files
    P_r = np.load(real_data_1_path)  # Shape: (J, 3)
    P_s = np.load(real_data_2_path)  # Shape: (J, 3)

    # Determine the smaller array
    if P_r.shape[0] > P_s.shape[0]:
        smaller_array = real_data_2_path
        bigger_array = real_data_1_path
        max_frames = P_r.shape[0]
    else:
        smaller_array = real_data_1_path
        bigger_array = real_data_2_path
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

    if bigger_array == real_data_1_path:
        for w_A, w_B in weight_pairs:
            P_opt = compute_P_opt(real_data_1_path, extended_new_path, alpha=alpha_weight, w_A=w_A, w_B=w_B)
            np.save(f"{folder_path_variations}/_euclidean_distances_wA{w_A}_wB{w_B}.npy", P_opt)

    # in a situation where the real_path is shorter than the synthetic path
    if smaller_array == real_data_1_path:
        for w_A, w_B in weight_pairs:
            P_opt = compute_P_opt(real_data_2_path, extended_new_path, alpha=alpha_weight, w_A=w_A, w_B=w_B) 
            np.save(f"{folder_path_variations}/_euclidean_distances_wA{w_A}_wB{w_B}.npy", P_opt)
