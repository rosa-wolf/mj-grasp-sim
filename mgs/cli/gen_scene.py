import os
import json
from copy import deepcopy

import hydra
import numpy as np
from omegaconf import DictConfig

from mgs.env.selector import get_env, get_env_from_dict
from mgs.gripper.selector import get_gripper
from mgs.obj.selector import get_objects
from mgs.util.file import generate_unique_hash
from mgs.util.geo.transforms import SE3Pose

# In-bound XY limits (replicated from cleaning logic)
_XY_MIN = -0.20
_XY_MAX = 0.20

# Optional JAX-based fingertip center computation for in-bound filtering
import jax.numpy as jnp  # type: ignore
from flax import nnx  # type: ignore

from mgs.sampler.kin.allegro import AllegroKinematicsModel  # type: ignore
from mgs.sampler.kin.dexee import DexeeKinematicsModel  # type: ignore
from mgs.sampler.kin.op import forward_kinematic_point_transform  # type: ignore
from mgs.sampler.kin.shadow import ShadowKinematicsModel  # type: ignore

_GRIPPERS_WITH_INBOUND = {"AllegroGripper", "DexeeGripper", "ShadowHand"}


def _get_kinematics(gripper_name: str):
    if gripper_name == "AllegroGripper":
        return AllegroKinematicsModel()
    if gripper_name == "DexeeGripper":
        return DexeeKinematicsModel()
    if gripper_name == "ShadowHand":
        return ShadowKinematicsModel()
    return None  # Others skip in-bound filtering


def _compute_in_bound_mask(
    poses: np.ndarray, joints: np.ndarray, gripper_name: str
) -> np.ndarray:
    """Return boolean mask of in-bound grasps for select grippers.

    Mirrors logic from `clean_grasp_centers.py` (padding to 1500 grasps) to
    avoid shape-dependent issues inside vmap. Unsupported grippers -> all True.
    """
    if gripper_name not in _GRIPPERS_WITH_INBOUND:
        return np.ones((poses.shape[0],), dtype=bool)
    if poses.size == 0:
        return np.zeros((0,), dtype=bool)

    kin = _get_kinematics(gripper_name)
    if kin is None:
        return np.ones((poses.shape[0],), dtype=bool)

    g, s = nnx.split(kin)
    fingertip_idx = kin.fingertip_idx.value  # (K,)
    num_fingertips = fingertip_idx.shape[0]
    local_points = jnp.zeros((num_fingertips, 3), dtype=jnp.float32)

    transformed_local = nnx.vmap(  # over grasps
        nnx.vmap(  # over fingertip indices
            forward_kinematic_point_transform,
            in_axes=(None, 0, 0, None, None),
        ),
        in_axes=(0, None, None, None, None),
    )(jnp.asarray(joints, dtype=jnp.float32), local_points, fingertip_idx, g, s)

    world_pts = (
        jnp.einsum("bij,bkj->bki", jnp.asarray(poses)[:, :3, :3], transformed_local)
        + jnp.asarray(poses)[:, :3, 3][:, None, :]
    )  # (B,K,3)
    centers = jnp.mean(world_pts, axis=1)  # (B,3)
    in_bound = (
        (centers[:, 0] < _XY_MAX)
        & (centers[:, 0] > _XY_MIN)
        & (centers[:, 1] < _XY_MAX)
        & (centers[:, 1] > _XY_MIN)
    )
    return in_bound


def fps_rank_grasps(
    poses_mat: np.ndarray,
    k=None,
    rot_weight: float = 0.05,
    seed=None,
) -> np.ndarray:
    """
    Farthest-point sampling (greedy) over SE(3) grasps.

    Args
    ----
    poses_mat : (N,4,4) homogeneous transforms for grasps.
    k         : how many to keep (defaults to N = full ranking).
    rot_weight: length scale (meters per radian) for blending orientation.
                Larger -> orientation matters more.
    seed      : optional RNG seed for deterministic start when data is flat.

    Returns
    -------
    order : (k,) int indices giving a *ranking* (most diverse first).
    """
    assert poses_mat.ndim == 3 and poses_mat.shape[1:] == (4, 4)
    N = poses_mat.shape[0]
    if N == 0:
        return np.empty((0,), dtype=np.int64)
    if k is None or k > N:
        k = N

    # Extract positions + unit quaternions (wxyz) via SE3Pose for robustness.
    se3 = SE3Pose.from_mat(poses_mat, type="wxyz")
    X = se3.pos.astype(np.float32)  # (N,3)
    Q = se3.quat.astype(np.float32)  # (N,4), unit

    # Helper: angular distance (radians) between quats, using double-cover fix.
    def ang_dist_to(q_sel: np.ndarray, Q_all: np.ndarray) -> np.ndarray:
        dots = np.abs(np.sum(q_sel[None, :] * Q_all, axis=1))  # |dot|
        dots = np.clip(dots, -1.0, 1.0)
        return 2.0 * np.arccos(dots).astype(np.float32)  # in [0, pi]

    rng = np.random.default_rng(seed)
    # Start: pick the pose farthest from the centroid (good heuristic), fallback random if degenerate.
    centroid = X.mean(axis=0, keepdims=True)
    d0 = np.linalg.norm(X - centroid, axis=1)
    start = int(np.argmax(d0)) if np.any(d0 > 0) else int(rng.integers(N))

    selected = np.empty(k, dtype=np.int64)
    selected[0] = start

    # Current distance to the selected set (min over selected so far)
    # Init with distance to the first selected
    d_pos = np.linalg.norm(X - X[start], axis=1)  # (N,)
    d_rot = ang_dist_to(Q[start], Q)  # (N,)
    d_min = np.sqrt(d_pos**2 + (rot_weight * d_rot) ** 2)  # (N,)

    # Greedy updates
    for t in range(1, k):
        # pick farthest from current selected set
        idx = int(np.argmax(d_min))
        selected[t] = idx

        # update distances using new center
        d_pos = np.minimum(d_min, np.linalg.norm(X - X[idx], axis=1))
        d_rot = ang_dist_to(Q[idx], Q)
        d_comb = np.sqrt(
            np.linalg.norm(X - X[idx], axis=1) ** 2 + (rot_weight * d_rot) ** 2
        )
        # maintain min distance to any selected
        d_min = np.minimum(d_min, d_comb)

    return selected


def get_grasps(gripper_name, obj_id):
    grasp_dir = os.path.join(  # type: ignore
        os.getenv("MGS_INPUT_DIR"),  # type: ignore
        
        gripper_name,
        obj_id,
    )
    poses, joints = [], []
    for file in os.listdir(grasp_dir):
        path = os.path.join(grasp_dir, file)
        grasp_dict = np.load(path)
        poses.append(grasp_dict["poses"])
        joints.append(grasp_dict["joints"])

    if len(poses) == 0 or len(joints) == 0:
        return None, None

    poses = np.concatenate(poses, axis=0)
    joints = np.concatenate(joints, axis=0)

    idx = np.random.permutation(len(poses))
    poses = poses[idx][:50000]
    joints = joints[idx][:50000]
    order = fps_rank_grasps(poses, rot_weight=0.1, k=10000)
    poses, joints = poses[order], joints[order]

    return poses, joints


def gen_stable_scene(cfg: DictConfig, max_attempts: int = 5):
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        obj_list = get_objects(cfg.object)
        
        # If num_objects is specified and less than available objects, randomly select
        #if hasattr(cfg, 'num_objects') and cfg.num_objects is not None and cfg.num_objects < len(obj_list):
        #    raise ValueError(f"more than {cfg.num_objects} objecs selected.")
            
        gripper = get_gripper(
            cfg.gripper,
            default_pose=SE3Pose(
                np.array([5.0, 5.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0]), type="wxyz"
            ),
        )
        env = get_env(cfg.env, gripper=deepcopy(gripper), obj_list=deepcopy(obj_list), headless=True)
        obj_poses = env.gen_clutter()
        scene_dict = env.to_dict()
        
        # Convert SE3Pose objects to matrices for serialization
        obj_poses_matrices = {}
        obj_ids = {} 
        for i, (obj_name, pose) in enumerate(obj_poses.items()):
            obj_poses_matrices[obj_name] = pose.to_mat()
            obj_ids[obj_name] = env.object_ids[i]   
        scene_dict["obj_poses"] = obj_poses_matrices
        scene_dict["obj_ids"] = obj_ids 

        exclude_scene = False
        for obj_name in getattr(env, "object_names", []):
            jnt = env.model.jnt(f"{obj_name}:joint")
            jnt_adr_start = jnt.qposadr[0].item()
            obj_position = np.copy(env.data.qpos[jnt_adr_start : jnt_adr_start + 3])
            x, y = float(obj_position[0]), float(obj_position[1])
            if (0.20 < abs(x)) or (0.20 < abs(y) # too far at edge of later rendered field
            ):
                exclude_scene = True
                break

        if env.is_stable() and not exclude_scene:
            return scene_dict
        last_error = ValueError("Scene unstable or excluded")
    raise (
        last_error if last_error is not None else ValueError("Scene generation failed")
    )


def filter_grasps(cfg: DictConfig, scene_def):
    env = get_env_from_dict(cfg.env, (deepcopy(scene_def)), headless=True)

    all_grasps = []
    for obj_name, obj_id in zip(env.object_names, env.object_ids):
        poses, joints = get_grasps(
            gripper_name=cfg.gripper.name,
            obj_id=obj_id,
        )
        if poses is None or joints is None:
            continue  # skip objects with missing grasp data
        o2w = env.get_obj_pose(obj_name)
        se3_pose = SE3Pose.from_mat(deepcopy(poses))
        grasp_pose = o2w @ se3_pose
        all_grasps.append(
            (
                grasp_pose.to_mat(),
                joints,
                obj_name,
                obj_id,
            )
        )

    all_poses = []
    all_joints = []
    obj_indices = []
    obj_map = []

    for idx, (
        collision_free_poses,
        collision_free_joints,
        obj_name,
        obj_id,
    ) in enumerate(all_grasps):
        pose_count = len(collision_free_poses)
        if pose_count > 0:
            all_poses.append(collision_free_poses)
            all_joints.append(collision_free_joints)
            obj_indices.append(np.full(pose_count, idx, dtype=np.int32))
            obj_map.append((obj_name, obj_id))

    if len(all_poses) == 0:
        raise ValueError("No grasps loaded")
    all_poses = np.concatenate(all_poses, axis=0)
    all_joints = np.concatenate(all_joints, axis=0)
    obj_indices = np.concatenate(obj_indices, axis=0)

    # In-bound filtering (before collision check) for select grippers
    in_bound_mask = _compute_in_bound_mask(all_poses, all_joints, cfg.gripper.name)
    if in_bound_mask.sum() == 0:
        raise ValueError("No in-bound grasps")
    all_poses = all_poses[in_bound_mask]
    all_joints = all_joints[in_bound_mask]
    obj_indices = obj_indices[in_bound_mask]

    collision_free_mask = env.grasp_collision_mask(
        SE3Pose.from_mat(deepcopy(all_poses), type="wxyz"),
        deepcopy(all_joints),
        with_padding=0.002,
    )

    if sum(collision_free_mask) <= 100:
        raise ValueError(
            f"Not enough collision free grasps! Only: {sum(collision_free_mask)}"
        )

    collision_free_poses = all_poses[collision_free_mask]
    collision_free_joints = all_joints[collision_free_mask]
    collision_free_obj_indices = obj_indices[collision_free_mask]

    collision_poses = all_poses[~collision_free_mask]
    collision_joints = all_joints[~collision_free_mask]
    collision_obj_indices = obj_indices[~collision_free_mask]

    if not cfg.only_collision_free:
        order = fps_rank_grasps(
            collision_free_poses,
            k=None,  # keep ordering for all; set to an int to subsample if desired
            rot_weight=getattr(cfg, "fps_rot_weight", 0.1),
            seed=getattr(cfg, "fps_seed", None),
        )
        collision_free_poses = collision_free_poses[order]
        collision_free_joints = collision_free_joints[order]
        collision_free_obj_indices = collision_free_obj_indices[order]

        print(f"Total collision-free grasps: {len(collision_free_poses)}")
        if sum(collision_free_mask) < cfg.min_stable:
            raise ValueError(
                f"Not enough collision free grasps! Only: {sum(collision_free_mask)}"
            )
        
        #stable_grasp_mask, success_labels = env.gen_success_labels(
        #    SE3Pose.from_mat(deepcopy(collision_free_poses), type="wxyz"),
        #    deepcopy(collision_free_joints),
        #    collision_free_obj_indices,
        #    deepcopy(scene_def["env_state"]["state"]),
        #    enough_stable=cfg.enough_stable,
        #)

        # get two-length label ([correct object, successful grasp] )
        stable_grasp_mask = env.grasp_stable_mask(
            SE3Pose.from_mat(deepcopy(collision_free_poses), type="wxyz"),
            deepcopy(collision_free_joints),
            collision_free_obj_indices,
            deepcopy(scene_def["env_state"]["state"]),
            enough_stable=cfg.enough_stable,
            enough_failed=cfg.enough_failed,
            with_wrong_label=True,
            apply_external_force=True,
            check_wrong_object=False,
        )

        # get idx of all grasps where label is [True, True]
        stable_idx = np.where(stable_grasp_mask[0].all(axis=1))[0]
        unstable_idx = np.where((stable_grasp_mask[0][:, 0] == True) & (stable_grasp_mask[0][:, 1] == False))[0] 
        if stable_idx.shape[0] < cfg.min_stable:
            raise ValueError(
                f"Not enough stable grasps! Only: {stable_idx.shape[0]}"
            )

        result_poses = collision_free_poses[stable_idx]
        result_joints = collision_free_joints[stable_idx]
        result_obj_indices = collision_free_obj_indices[stable_idx]

        #if stable_grasp_mask.shape[0] >= cfg.enough_stable:
        #    stable_grasp_mask[cfg.enough_stable:] = True

        if cfg.with_failed_grasps:
            # take unstalbe grasps as initial set of failed grasps
            failed_poses = collision_free_poses[unstable_idx]
            failed_joints = collision_free_joints[unstable_idx]
            failed_obj_indices = collision_free_obj_indices[unstable_idx]
            
            
            #failed_poses = np.array([])
            #failed_joints = np.array([])
            #failed_obj_indices = np.array([])

            if failed_poses.shape[0] < cfg.enough_failed:
                failed_poses_extra, failed_joints_extra, failed_obj_indices_extra = env.gen_failed_grasps(
                    SE3Pose.from_mat(deepcopy(collision_free_poses[np.concat((stable_idx, unstable_idx))]), type="wxyz"),
                    deepcopy(collision_free_joints[np.concat((stable_idx, unstable_idx))]),
                    collision_free_obj_indices[np.concat((stable_idx, unstable_idx))],
                    deepcopy(scene_def["env_state"]["state"]),
                    enough_failed=cfg.enough_failed
            )
                failed_poses = np.concat((failed_poses, failed_poses_extra), axis=0)
                failed_joints = np.concat((failed_joints, failed_joints_extra), axis=0)
                failed_obj_indices = np.concat((failed_obj_indices, failed_obj_indices_extra), axis=0)

            if failed_poses.shape[0] < cfg.min_failed:
                raise ValueError(
                    f"Not enough failed grasps! Only: {failed_poses.shape[0]}"
                )

    else:
        result_poses = collision_free_poses
        result_joints = collision_free_joints
        result_obj_indices = collision_free_obj_indices

    result = []
    neg_result = []
    failed_result = []
    grasp_info = {} 
    for obj_idx in np.unique(result_obj_indices):
        mask = result_obj_indices == obj_idx
        if sum(mask) == 0:
            continue
        obj_name, obj_id = obj_map[obj_idx]
        result.append(
            {
                "object_id": obj_id,
                "object_name": obj_name,
                "pose": result_poses[mask],
                "joints": result_joints[mask],
            }
        )
        grasp_info[obj_id] = result_poses[mask].shape[0]    
        
        if cfg.save_collision_grasps:
            collision_mask = collision_obj_indices == obj_idx
            if sum(collision_mask) > 0:
                neg_result.append(
                    {
                        "object_id": obj_id,
                        "object_name": obj_name,
                        "pose": collision_poses[collision_mask],
                        "joints": collision_joints[collision_mask],
                    }
                )
        if cfg.with_failed_grasps:
            failed_mask = failed_obj_indices == obj_idx
            if failed_poses.shape[0] > 0:
                failed_result.append(
                    {
                        "object_id": obj_id,
                        "object_name": obj_name,
                        "pose": failed_poses[failed_mask],
                        "joints": failed_joints[failed_mask],
                    }
                )

    return result, neg_result, failed_result, grasp_info


@hydra.main(version_base=None, config_path="config", config_name="gen_scene")
def main(cfg: DictConfig):    #output_dir = os.getenv("MGS_OUTPUT_DIR")
    #input_dir = os.getenv("MGS_INPUT_DIR")
    
    output_dir = "/home/ws/data/outputs/context_clutter_v3_set2" 
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    input_dir = os.path.join(repo_root, "outputs_obj_grasps")
    assert output_dir is not None, "No output_dir defined!"
    assert input_dir is not None, "No input_dir defined!"

    # Ensure helpers that call os.getenv("MGS_INPUT_DIR") / MGS_OUTPUT_DIR
    # receive valid paths (prevents passing None into os.path.join).
    os.environ["MGS_INPUT_DIR"] = input_dir
    # set base output dir (before adding gripper/hash suffix)
    os.environ["MGS_OUTPUT_DIR"] = output_dir
    
    #try:
    scene_dict = gen_stable_scene(cfg)
    valid_grasps, invalid_grasps, failed_grasps, grasp_info = filter_grasps(cfg, scene_dict)
    
    output_dir = os.path.join(
        output_dir,
        cfg.gripper.name,
        generate_unique_hash(16) + "_" + valid_grasps[0]["object_id"],
    )

    scene_path = os.path.join(output_dir, "scene")
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        scene_path,
        **{
            "scene_definition": scene_dict,
        },
    )
    # save grasp info as json
    grasp_info_path = os.path.join(output_dir, "grasp_info.json")
    with open(grasp_info_path, 'w') as f:
        json.dump(grasp_info, f, indent=2)
    
    for grasps in valid_grasps:
        obj_id, obj_name = grasps["object_id"], grasps["object_name"]
        object_path = os.path.join(output_dir, obj_id + "_" + obj_name)
        np.savez(
            object_path,
            **{
                "pose": grasps["pose"],
                "joints": grasps["joints"],
            },
        )
    for grasps in invalid_grasps:
        obj_id, obj_name = grasps["object_id"], grasps["object_name"]
        object_path = os.path.join(
            output_dir, obj_id + "_" + obj_name + "_" + "collision"
        )
        np.savez(
            object_path,
            **{
                "pose": grasps["pose"],
                "joints": grasps["joints"],
            },
        )
    for grasps in failed_grasps:
        obj_id, obj_name = grasps["object_id"], grasps["object_name"]
        object_path = os.path.join(
            output_dir, obj_id + "_" + obj_name + "_" + "failed"
        )
        np.savez(
            object_path,
            **{
                "pose": grasps["pose"],
                "joints": grasps["joints"],
            },
        )
            
    print(f"Scene and grasps saved to: {output_dir}")

    #except Exception as e:
    #    print(e)


if __name__ == "__main__":
    main()
