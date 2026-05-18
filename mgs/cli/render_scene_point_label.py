import os
import importlib
from copy import deepcopy

import hydra
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from mgs.env.selector import get_env_from_dict
from mgs.gripper.base import MjScannableGripper
from mgs.gripper.selector import get_gripper
from mgs.sampler.helper import farthest_point_sampling
from mgs.util.img_proc import detect_outlier, rgbd_to_pcd, voxel_downsample_pcd
import mujoco
import open3d as o3d

def visualize_pointcloud_and_wait(points: np.ndarray, colors: np.ndarray):
    """Show a point cloud in Open3D and block until the user presses Space/Enter/Q."""
   

    if points.size == 0:
        print("Point cloud is empty. Skipping Open3D visualization.")
        return

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Scene Point Cloud")

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd_o3d.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    vis.add_geometry(pcd_o3d)

    def _close_callback(v):
        v.close()
        return False

    # Continue when user presses Space, Enter, or Q.
    vis.register_key_callback(ord(" "), _close_callback)
    vis.register_key_callback(257, _close_callback)  # GLFW_KEY_ENTER
    vis.register_key_callback(ord("Q"), _close_callback)

    print("Open3D viewer opened. Press Space/Enter/Q (or close window) to continue...")
    vis.run()
    vis.destroy_window()


def scan(cfg: DictConfig, scene_def):
    gripper = get_gripper(cfg.gripper)
    assert isinstance(gripper, MjScannableGripper)
    env = get_env_from_dict(cfg.env, (deepcopy(scene_def)), headless=True)
    images, extrinsics, image_masks, segmentation_mask, segmentation_labels  = env.scan(num_images=cfg.num_images)
    intrinsics = env.get_camera_intrinsics()
    return images, extrinsics, intrinsics, image_masks, segmentation_mask, segmentation_labels

@hydra.main(config_path="config", config_name="render_scene")
def main(cfg: DictConfig):
    #output_dir_all = os.getenv("MGS_OUTPUT_DIR")
    #input_dir_all = os.getenv("MGS_INPUT_DIR")

    #assert output_dir_all is not None
    #assert input_dir_all is not None
    
    input_dir_all = "/home/ws/data/outputs/context_clutter_v3" 
    output_dir_all = input_dir_all
    assert output_dir_all is not None, "No output_dir defined!"
    assert input_dir_all is not None, "No input_dir defined!"

    # Ensure helpers that call os.getenv("MGS_INPUT_DIR") / MGS_OUTPUT_DIR
    # receive valid paths (prevents passing None into os.path.join).
    #os.environ.setdefault("MGS_INPUT_DIR", input_dir)
    # set base output dir (before adding gripper/hash suffix)
    #os.environ.setdefault("MGS_OUTPUT_DIR", output_dir)

    input_dir_all = os.path.join(input_dir_all, cfg.gripper.name)
     
    scene_dir_list = [
        d for d in os.listdir(input_dir_all) if os.path.isdir(os.path.join(input_dir_all, d))
    ]
    
    # filter all scene where file name is starting with cfg.input_id
    print(cfg.input_id)
    scene_dir_list = [d for d in scene_dir_list if d.startswith(str(cfg.input_id))]
    
    
    # filter all scene dirs where file scene_pcd.npz already exists
    scene_dir_list = [
        d for d in scene_dir_list if not os.path.exists(os.path.join(input_dir_all, d, "scene_pcd.npz"))
    ]
    

    num  = len(scene_dir_list)
    count = 1
    for scene_dir in scene_dir_list:

        input_dir = os.path.join(input_dir_all, scene_dir)
        print("Scene dir: ", input_dir)

        scene_path = os.path.join(input_dir, "scene.npz")
        scene = np.load(scene_path, allow_pickle=True)
        scene_dict = scene["scene_definition"].item()
        images, extrinsics, intrinsics, image_masks, segmentation_label, segmentation_label_names = scan(
            deepcopy(cfg), deepcopy(scene_dict)
        )
        pcd, feature = rgbd_to_pcd(images, intrinsics, extrinsics)
        pcd = pcd[image_masks]
        feature = feature[image_masks]
        segmentation_label = segmentation_label[image_masks]

        region_mask = np.all(
            (pcd < np.array([[0.225, 0.225, 1.0]]))
            & (pcd > np.array([[-0.225, -0.225, -0.01]])),
            axis=-1,
        )
        pcd = pcd[region_mask]
        feature = feature[region_mask]
        segmentation_label = segmentation_label[region_mask]
        
        pcd, feature, segmentation_label = voxel_downsample_pcd(pcd, feature, voxel_size=0.002, segmentation_label=segmentation_label)
        mask = detect_outlier(pcd, radius=0.008, min_neighbors=2)
        pcd, feature, segmentation_label = pcd[mask], feature[mask], segmentation_label[mask]
        idx = farthest_point_sampling(
            jnp.asarray(pcd, dtype=jnp.float32), num_samples=15000
        )
        pcd = pcd[idx]
        feature = feature[idx]
        segmentation_label = segmentation_label[idx]

        #visualize_pointcloud_and_wait(pcd, feature)

        output_dir = os.path.join(output_dir_all, cfg.gripper.name, scene_dir)
        os.makedirs(output_dir, exist_ok=True)
        np.savez(
            os.path.join(output_dir, "scene_pcd"),
            **{
                "points": np.asarray(pcd, dtype=np.float32),
                "colors": np.asarray(feature, dtype=np.float32),
                "labels": np.asarray(segmentation_label, dtype=np.int32),
                "label_names": segmentation_label_names
            },
        )
        print(f"Finished with scene {count} of {num}!")
        count += 1


if __name__ == "__main__":
    main()
