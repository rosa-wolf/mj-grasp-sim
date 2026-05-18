# Copyright (c) 2025 Robert Bosch GmbH
# Author: Roman Freiberg
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import random
from copy import deepcopy
from typing import Dict, List, TypedDict

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from mgs import obj
from mgs.env.base import Loadable, MjScanEnv
from mgs.gripper.base import MjScannable, MjShakableOpenCloseGripper
from mgs.obj.base import CollisionMeshObject
from mgs.util.camera import fibonacci_sphere
from mgs.util.geo.convert import quat_xyzw_to_wxyz
from mgs.util.geo.transforms import SE3Pose



class ClutterTableState(TypedDict):
    geom_conaffinity: np.ndarray
    geom_contype: np.ndarray
    geom_rgba: np.ndarray
    body_gravcomp: np.ndarray
    state: np.ndarray


XML = r"""
<mujoco>
    <compiler angle="radian" autolimits="true" />
    <size memory="32M" />
    <option integrator="implicitfast" timestep="0.001"/>
    <compiler discardvisual="false"/>
    <option noslip_iterations="3"> </option>
    <option><flag multiccd="enable"/> </option>
    <option cone="elliptic" gravity="0 0 -9.81" impratio="3" timestep="0.001" noslip_iterations="3" noslip_tolerance="1e-10" tolerance="1e-10"/>
    {gripper}
    <worldbody>
        {lights}
        <body name="body:table" pos="0.0 0 -0.02">
           <geom name="geom:table" pos="0 0 0" rgba="{table_color}" size="10 10 0.02" type="box" density="500" friction="1.0 0.1 0.1"/>
        </body>
        <body name="body:camera" pos="0.0 0.0 -1.0" quat="1.0 0.0 0 0" gravcomp="1">
          <freejoint name="camera:joint"/>
          <geom name="geom:camera" size="0.01"/>
          <camera name="camera" mode="targetbody" target="base_origin"/>
        </body>
        <body name="base_origin" pos="0.0 0.0 -0.025" quat="1.0 0.0 0 0">
          <geom name="geom:base_origin" size="0.01" rgba="1.0 0 0 1"/>
        </body>
        <body name="body:wall_top" pos="0.0 1.0 0.1">
           <geom name="geom:wall_top" pos="0 0 0" rgba="1. 0. 0. 0." size="1.0 0.02 0.2" type="box" density="500"/>
        </body>
        <body name="body:wall_right" pos="1.0 0.0 0.1">
           <geom name="geom:wall_right" pos="0 0 0" rgba="1. 0. 0. 0." size="0.02 1.0 0.2" type="box" density="500"/>
        </body>
        <body name="body:wall_bottom" pos="0.0 -1.0 0.1">
           <geom name="geom:wall_bottom" pos="0 0 0" rgba="1. 0. 0. 0." size="1.0 0.02 0.2" type="box" density="500"/>
        </body>
        <body name="body:wall_left" pos="-1.0 0.0 0.1">
           <geom name="geom:wall_left" pos="0 0 0" rgba="1. 0. 0. 0." size="0.02 1.0 0.2" type="box" density="500"/>
        </body>
    </worldbody>
    {objects}
</mujoco>
"""


class ClutterTableEnv(MjScanEnv, Loadable):
    def __init__(
        self,
        gripper: MjShakableOpenCloseGripper,
        objects: List[CollisionMeshObject],
        headless=True,
        scene_randomization=True,
    ):
        self.gripper = gripper
        self.objects = objects
        self.gripper_xml, self.gripper_assets = gripper.to_xml()
        self.object_xml_assets = [obj.to_xml() for obj in objects]
        self.object_names = [obj.name for obj in objects]
        self.object_ids = [obj.object_id for obj in objects]

        self.objs_xml_concat = ""
        self.objs_assets = {}
        for obj_xml, obj_assets in self.object_xml_assets:
            self.objs_xml_concat += obj_xml
            self.objs_assets = {**self.objs_assets, **obj_assets}

        if scene_randomization:
            random_color = (
                " ".join([str(np.random.uniform(0, 1)) for _ in range(3)]) + " 1.0"
            )
        else:
            random_color = "0.5 0.5 0.5 1.0"

        rand_x, rand_y = np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)
        light_one = f"""<light name="light:one" pos="{rand_x} {rand_y} 2.0" attenuation="1.0 0.2 0.2" mode="targetbody" target="base_origin"/>"""
        rand_x, rand_y = np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)
        light_two = f"""<light name="light:two" pos="{rand_x} {rand_y} 2.0" attenuation="1.0 0.2 0.2" mode="targetbody" target="base_origin"/>"""
        rand_x, rand_y = np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)
        light_three = f"""<light name="light:three" pos="{rand_x} {rand_y} 2.0" attenuation="1.0 0.2 0.2" mode="targetbody" target="base_origin"/>"""
        num_lights = random.randint(1, 3)
        light_xml = "".join([light_one, light_two, light_three][:num_lights])

        self.model_xml = XML.format(
            **{
                "gripper": self.gripper_xml,
                "objects": self.objs_xml_concat,
                "table_color": random_color,
                "lights": light_xml,
            }
        )

        self.env_defintion = {
            "model_xml": self.model_xml,
            "assets": {**self.gripper_assets, **self.objs_assets},
        }
        self.model = mujoco.MjModel.from_xml_string(  # type: ignore
            self.model_xml, {**self.gripper_assets, **self.objs_assets}
        )
        self.data = mujoco.MjData(self.model)  # type: ignore
        mujoco.mj_forward(self.model, self.data)  # type: ignore

        self.viewer = None
        if not headless:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            print(f"[INFO]: Viewer launched")

        self.next_x, self.next_y, self.counter = -5.5, -5.0, 0
        super().__init__("camera", 480, 480)

    def get_object(self, object_name: str):
        for obj in self.objects:
            if obj.name == object_name:
                return obj
        return None

    def remove_obj(self, obj):
        body_id = self.model.body(obj.name).id

        for i in range(self.model.ngeom):
            if self.model.geom_bodyid[i] == body_id:
                self.model.geom_conaffinity[i] = 0
                self.model.geom_contype[i] = 0
                self.model.geom_rgba[i] = [0, 0, 0, 0]
        self.model.body_gravcomp[body_id] = 1.0
        mujoco.mj_forward(self.model, self.data)  # type: ignore

    def settle(self):
        mujoco.mj_step(self.model, self.data, 10000) # type: ignore
        if self.viewer:
            if self.viewer.is_running():
                self.viewer.sync() 

    def is_stable(self):
        stats = {}
        for obj in self.objects:
            stats[obj.name] = 0
        start_poses = []
        for _ in range(10):
            start_poses = []
            for obj in self.objects:
                obj_id = mujoco.mj_name2id(  # type:ignore
                    m=self.model,
                    name=(obj.name + ":joint"),
                    type=mujoco.mjtObj.mjOBJ_JOINT,  # type: ignore
                )
                obj_qpos_addr = self.model.jnt_qposadr[obj_id]
                start_pos = deepcopy(self.data.qpos[obj_qpos_addr : obj_qpos_addr + 3])
                start_poses.append(start_pos)

            mujoco.mj_step(self.model, self.data, 100)  # type: ignore
            if self.viewer:
                if self.viewer.is_running():
                    self.viewer.sync()

            for obj, start_pos in zip(self.objects, start_poses):
                obj_id = mujoco.mj_name2id(  # type: ignore
                    m=self.model,
                    name=(obj.name + ":joint"),
                    type=mujoco.mjtObj.mjOBJ_JOINT,  # type: ignore
                )
                obj_qpos_addr = self.model.jnt_qposadr[obj_id]
                end_pos = deepcopy(self.data.qpos[obj_qpos_addr : obj_qpos_addr + 3])
                delta = np.sum(np.abs(end_pos - start_pos))
                stats[obj.name] += delta

        max_delta = 0.0
        for delta in stats.values():
            if delta > max_delta:
                max_delta = delta

        return max_delta < 5e-3

    def gen_clutter(self) -> Dict[str, SE3Pose]:
        def random_pose():
            scipy_random_quat = Rotation.random().as_quat()  # type: ignore
            mujoco_random_quat = quat_xyzw_to_wxyz(scipy_random_quat)
            if np.random.rand() < 0.05:
                # sample rotation angle around z-axis
                scipy_random_quat = Rotation.from_euler("z", np.random.uniform(0, 2 * np.pi)).as_quat() 
                mujoco_random_quat = quat_xyzw_to_wxyz(scipy_random_quat)
            return SE3Pose(
                pos=np.array([np.random.uniform(-0.08, 0.08), np.random.uniform(-0.08, 0.08), 0.4]), quat=mujoco_random_quat, type="wxyz"
                #pos=np.array(0.4, np.random.uniform(0., 0.4), 0.4]), quat=mujoco_random_quat, type="wxyz"
            )

        
        for obj_name in self.object_names:
            drop_pose = random_pose()
            jnt_adr_start = (
                self.model.jnt("{}:joint".format(obj_name)).qposadr[0].item()
            )
            self.data.qpos[jnt_adr_start : jnt_adr_start + 7] = deepcopy(
                drop_pose.to_vec(layout="pq", type="wxyz")
            )
            self.data.qacc[:] = 0.0
            self.data.qvel[:] = 0.0
            for _ in range(900):
                np.clip(self.data.qacc, -50.0, 50.0, out=self.data.qacc)
                np.clip(self.data.qvel, -50.0, 50.0, out=self.data.qvel)
                mujoco.mj_step(self.model, self.data)  # type: ignore
                if self.viewer:
                    if self.viewer.is_running():
                        self.viewer.sync()
        for _ in range(5000):
            np.clip(self.data.qacc, -1.0, 1.0, out=self.data.qacc)
            np.clip(self.data.qvel, -50.0, 50.0, out=self.data.qvel)
            mujoco.mj_step(self.model, self.data)  # type: ignore
            if self.viewer:
                if self.viewer.is_running():
                    self.viewer.sync()
                    
        # Capture final poses after settling
        final_poses = {}
        for obj_name in self.object_names:
            final_poses[obj_name] = self.get_obj_pose(obj_name)
            
        return final_poses

    def update_camera_settings(self, num_images, i):
        rnd_pos = fibonacci_sphere(total_num=num_images, i=i) * 0.75
        rnd_pos[2] = np.abs(rnd_pos[2]) + 0.01  # upper hemisphere
        jnt_adr_start = self.model.jnt("camera:joint").qposadr[0].item()
        self.data.qpos[jnt_adr_start : jnt_adr_start + 3] = rnd_pos
        mujoco.mj_forward(self.model, self.data)  # type: ignore
        options = (
            self.gripper.get_render_options()
            if isinstance(self.gripper, MjScannable)
            else None
        )
        self.renderer.update_scene(self.data, camera="camera", scene_option=options)

    def check_gripper_collision(self):
        """
        As the geoms are ordered accordingly to the XML. We can simply
        check for contacts between obj geoms and gripper geoms by ids
        relative to the table (which is inbetween obj and gripper by construction)
        """
        table_id = self.model.geom("geom:table").id
        for contact_pairs in self.data.contact.geom:
            if (
                (contact_pairs[0] < table_id and contact_pairs[1] > table_id)
                or (contact_pairs[0] > table_id and contact_pairs[1] < table_id)
                or (contact_pairs[0] == table_id and contact_pairs[1] < table_id)
                or (contact_pairs[0] < table_id and contact_pairs[1] == table_id)
            ):
                return 1.0
        return 0.0

    def check_gripper_contact(self):
        table_id = self.model.geom("geom:table").id
        for contact_pairs in self.data.contact.geom:
            if (
                (contact_pairs[0] < table_id and contact_pairs[1] > table_id)
                or (contact_pairs[0] > table_id and contact_pairs[1] < table_id)
                and (
                    not (
                        (contact_pairs[0] == table_id and contact_pairs[1] < table_id)
                        or (
                            contact_pairs[0] < table_id and contact_pairs[1] == table_id
                        )
                    )
                )
            ):
                return 1.0
        return 0.0
    

    def find_named_parent_for_geom(self, geom_idx):
        # try geom name first
        geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_idx)
        if geom_name:
            return ("geom", geom_name)

        # fallback: geom -> body -> walk up body parent chain
        body_id = int(self.model.geom_bodyid[geom_idx])
        while body_id != -1 and body_id is not None:
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name:
                return body_name
            # walk to parent body
            body_id = int(self.model.body_parentid[body_id])

        return None
    

    def collision_obj_id(self):
        """
        As the geoms are ordered accordingly to the XML. We can simply
        check for contacts between obj geoms and gripper geoms by ids
        relative to the table (which is inbetween obj and gripper by construction)
        """
        # geom ids to objects
        id_names = []
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name is None:
                name = self.find_named_parent_for_geom(i)
            if "hand" in name or "finger" in name or "col" in name:
                name = "gripper"
            id_names.append(name)
        
        collision_obj = {'id': [], 'name': []}
        for contact_pairs in self.data.contact.geom:
            if (id_names[contact_pairs[0]] == "gripper" and id_names[contact_pairs[1]] in self.object_names):
                    if id_names[contact_pairs[1]] not in collision_obj['id']:
                        collision_obj['id'].append(id_names[contact_pairs[1]])
                        idx = [i for i in range(len(self.object_names)) if self.object_names[i] == id_names[contact_pairs[1]]][0]
                        collision_obj['name'].append(self.object_ids[idx])
            elif (id_names[contact_pairs[1]] == "gripper" and id_names[contact_pairs[0]] in self.object_names):
                    if id_names[contact_pairs[0]] not in collision_obj['id']:
                        collision_obj['id'].append(id_names[contact_pairs[0]])
                        idx = [i for i in range(len(self.object_names)) if self.object_names[i] == id_names[contact_pairs[1]]][0]
                        collision_obj['name'].append(self.object_ids[idx])
        return collision_obj

    
    
    def grasp_stable_mask_bench(
        self,
        poses: SE3Pose,
        joints: np.ndarray,
        env_state,
        nstep_lift: int = 3000,  # Steps for lifting simulation
        lift_dist: float = 0.3,  # Distance to lift
        enough_stable=None,
        show_progress: bool = True,
        progress_desc: str | None = None,
    ):
        """
        Evaluate grasp stability with a live tqdm progress bar.
        Shows running success rate, success/fail counts, evaluated count, and skips (if enough_stable triggers).

        If tqdm is not installed/available, progress is silently disabled.
        """
        # lazy import tqdm; fall back gracefully
        pbar = None
        if show_progress:
            try:
                from tqdm.auto import tqdm  # type: ignore

                pbar = tqdm(
                    total=len(poses),
                    desc=progress_desc or "Evaluating grasps",
                    dynamic_ncols=True,
                    leave=False,
                )
            except Exception:
                pbar = None  # disable progress if tqdm missing or no TTY

        results: List[bool] = []
        num_grasps = len(poses)
        gripper_joint_idxs = self.get_joint_idxs(
            self.gripper.get_actuator_joint_names()
        )

        count_stable = 0  # number of successful grasps
        eval_count = 0  # number of grasps actually simulated (excludes 'skipped' due to enough_stable)
        skipped_count = 0  # how many we skipped after hitting enough_stable
        contact_loss_failures = 0  # failures during lift due to contact loss

        for i in range(num_grasps):
            lift_passed = True

            # early stopping: we still append False to keep mask length, but don't simulate
            if enough_stable is not None and count_stable >= enough_stable:
                results.append(False)
                skipped_count += 1
                # progress update
                if pbar is not None:
                    sr = (count_stable / eval_count) if eval_count > 0 else 0.0
                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"succ={count_stable} fail={eval_count - count_stable} "
                        f"skipped={skipped_count} SR={sr*100:.1f}%"
                    )
                continue

            # restore environment state for each evaluation
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            mujoco.mj_setState(self.model, self.data, env_state, spec)

            b2c = self.gripper.base_to_contact_transform()
            pose_processed = poses[i] @ b2c
            self.set_qpos(joints[i], gripper_joint_idxs)
            self.gripper.set_pose(self, pose_processed)

            # Update geom positions, then close
            mujoco.mj_forward(self.model, self.data)
            self.gripper.close_gripper_at(self, pose_processed)

            # --- Lift test ---
            eval_count += 1
            start_pos_lift = np.copy(self.data.mocap_pos[0, :])
            lift_target_z = start_pos_lift[2] + lift_dist
            for t in range(nstep_lift):
                alpha = t / nstep_lift
                self.data.mocap_pos[0, 2] = (
                    start_pos_lift[2] + (lift_target_z - start_pos_lift[2]) * alpha
                )
                mujoco.mj_step(self.model, self.data)

                # every 100 steps, verify contact
                if (t + 1) % 100 == 0 and not self.check_gripper_contact():
                    lift_passed = False
                    contact_loss_failures += 1
                    break

            results.append(lift_passed)
            count_stable += int(lift_passed)

            # progress update
            if pbar is not None:
                sr = (count_stable / eval_count) if eval_count > 0 else 0.0
                pbar.update(1)
                pbar.set_postfix_str(
                    f"succ={count_stable} fail={eval_count - count_stable} "
                    f"skipped={skipped_count} SR={sr*100:.1f}%"
                )

        if pbar is not None:
            # final line with totals; leave the bar collapsed
            sr = (count_stable / eval_count) if eval_count > 0 else 0.0
            pbar.clear()
            pbar.close()
            # optional: final one-liner print
            print(
                f"[grasp_stable_mask] evaluated={eval_count}, succ={count_stable}, "
                f"fail={eval_count - count_stable}, skipped={skipped_count}, "
                f"contact_fail={contact_loss_failures}, SR={sr*100:.1f}%"
            )

        stable_grasp_masks = np.array(results, dtype=bool)
        return stable_grasp_masks
    
    
    def grasp_stable_mask(
        self,
        poses: SE3Pose,
        joints: np.ndarray,
        ids: np.ndarray,
        env_state,
        nstep_lift: int = 2000,  # Steps for lifting simulation
        lift_dist: float = 0.3,  # Distance to lift
        enough_stable=None,
        enough_failed=None,
        show_progress: bool = False,
        progress_desc: str | None = None,
        inference=False,
        with_wrong_label=False,
        apply_external_force=False,
        check_wrong_object=False,
    ):
        """
        Evaluate grasp stability with a live tqdm progress bar.
        Shows running success rate, success/fail counts, evaluated count, and skips (if enough_stable triggers).

        If tqdm is not installed/available, progress is silently disabled.
        """
        # lazy import tqdm; fall back gracefully
        pbar = None
        if show_progress:
            try:
                from tqdm.auto import tqdm  # type: ignore

                pbar = tqdm(
                    total=len(poses),
                    desc=progress_desc or "Evaluating grasps",
                    dynamic_ncols=True,
                    leave=False,
                )
            except Exception:
                pbar = None  # disable progress if tqdm missing or no TTY

        results: List[bool] = []
        num_grasps = len(poses)
        gripper_joint_idxs = self.get_joint_idxs(
            self.gripper.get_actuator_joint_names()
        )

        count_stable = 0  # number of successful grasps
        eval_count = 0  # number of grasps actually simulated (excludes 'skipped' due to enough_stable)
        skipped_count = 0  # how many we skipped after hitting enough_stable
        count_fail = 0  # failures during lift due to contact loss
        count_wrong = 0 # failures due to wrong object being grasped

        for i in range(num_grasps):
            lift_passed = True
            correct_object = True

            # early stopping: we still append False to keep mask length, but don't simulate
            if enough_stable is not None and count_stable >= enough_stable:
                    results.append(np.array([None, None])) if with_wrong_label else results.append(None)
                    skipped_count += 1
                    # progress update
                    if pbar is not None:
                        sr = (count_stable / eval_count) if eval_count > 0 else 0.0
                        pbar.update(1)
                        pbar.set_postfix_str(
                            f"succ={count_stable} fail={eval_count - count_stable} "
                            f"skipped={skipped_count} SR={sr*100:.1f}%"
                        )
                    continue

            # restore environment state for each evaluation
            
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            mujoco.mj_setState(self.model, self.data, env_state, spec)

            if inference:
                pose_processed = poses[i]
                # open gripper
                self.gripper.open_gripper(self)
                #self.set_qpos(joints[i], gripper_joint_idxs)
                self.gripper.set_pose(self, pose_processed)  
            else:
                b2c = self.gripper.base_to_contact_transform()
                pose_processed = poses[i] @ b2c
                self.set_qpos(joints[i], gripper_joint_idxs)
                self.gripper.set_pose(self, pose_processed)

            # Update geom positions, then close
            mujoco.mj_forward(self.model, self.data)
            if self.viewer:
                if self.viewer.is_running():
                    self.viewer.sync()
            self.gripper.close_gripper_at(self, pose_processed)
            
            if self.viewer:
                if self.viewer.is_running():
                    self.viewer.sync()

            # --- Lift test ---
            eval_count += 1
            start_pos_lift = np.copy(self.data.mocap_pos[0, :])
            lift_target_z = start_pos_lift[2] + lift_dist
            for t in range(nstep_lift):
                alpha = t / nstep_lift
                self.data.mocap_pos[0, 2] = (
                    start_pos_lift[2] + (lift_target_z - start_pos_lift[2]) * alpha
                )
                mujoco.mj_step(self.model, self.data)
                if self.viewer:
                    if self.viewer.is_running():
                        self.viewer.sync()

                # every 100 steps, verify contact
                if (t + 1) % 100 == 0 and not self.check_gripper_contact():
                    lift_passed = False
                    count_fail += 1
                    break
            
            # check if grasp on correct object
            obj_id = self.collision_obj_id()
            if check_wrong_object:
                if ids is not None:
                    if len(obj_id['id']) == 0 and len(obj_id["name"]) == 0:
                        # grasped no object
                        correct_object = False
                    elif len(obj_id['id']) != 1 or len(obj_id["name"]) != 1:
                        # grasped more than one object
                        correct_object = False
                    elif isinstance(ids[i], int):
                        if self.object_names[ids[i]] not in obj_id['id']:
                            # grasped wrong object
                            correct_object = False
                    elif isinstance(ids[i], str):
                        if ids[i] not in obj_id["name"][0]:
                            # grasped wrong object
                            correct_object = False
                
            if lift_passed and correct_object:
                if apply_external_force:
                    IMPULSE_FORCE_N = float(
                        200.0
                    )  # one-step force magnitude (N) -> impulse J = F*dt
                    object_bid = self.model.body(obj_id["id"][0]).id   # apply at object COM
                    # snapshot the post-close state for deterministic, repeatable kicks
                    closed_state = self.get_state()
                    # world rotation of the grasp frame
                    Rg = pose_processed.to_mat()[:3, :3].astype(float)

                    # local unit axes in grasp frame
                    local_dirs = np.eye(3, dtype=float)
                    dirs_world = np.concatenate(
                        [
                            Rg @ local_dirs[:, [0, 1, 2]],  # +x,+y,+z
                            -(Rg @ local_dirs[:, [0, 1, 2]]),
                        ],
                        axis=1,
                    ).T  # -x,-y,-z
                    # dirs_world: shape (6, 3)

                    for d in dirs_world:
                        # restore saved state
                        self.set_state(closed_state)

                        F = IMPULSE_FORCE_N * d
                        for i in range(5):
                            self.data.xfrc_applied[object_bid, :3] += F
                            mujoco.mj_step(
                                self.model, self.data, nstep=1
                            )  # integrates one step
                            self.data.xfrc_applied[object_bid, :] = (
                                0.0  # clear so it doesn't persist
                            )
                            mujoco.mj_step(
                                self.model, self.data, nstep=10
                            )  # integrates one step

                        mujoco.mj_step(self.model, self.data, nstep=500)
                        if self.viewer:
                            if self.viewer.is_running():
                                self.viewer.sync()

                        # check if object is still in contact with gripper
                        if not self.check_gripper_contact():
                            lift_passed = False
                            break
                

            if with_wrong_label:
                results.append(np.array([correct_object, lift_passed]))
            else:
                results.append(lift_passed and correct_object)
            
            #print(f"[INFO]: id ={ids[i]}, lift_passed = {lift_passed}, correct_object = {correct_object}")   
            
            count_stable += int(lift_passed and correct_object)
            count_wrong += int(not correct_object)

            # progress update
            if pbar is not None:
                sr = (count_stable / eval_count) if eval_count > 0 else 0.0
                pbar.update(1)
                pbar.set_postfix_str(
                    f"succ={count_stable} fail={eval_count - count_stable}, wrong_object{count_wrong}"
                    f"skipped={skipped_count} SR={sr*100:.1f}%"
                )

        if pbar is not None:
            # final line with totals; leave the bar collapsed
            sr = (count_stable / eval_count) if eval_count > 0 else 0.0
            pbar.clear()
            pbar.close()
            # optional: final one-liner print
            print(
                f"[grasp_stable_mask] evaluated={eval_count}, succ={count_stable}, "
                f"fail={eval_count - count_stable}, skipped={skipped_count}, "
                f"contact_fail={count_fail}, wrong object grasped fail={count_wrong} SR={sr*100:.1f}%"
            )

        stable_grasp_masks = np.array(results, dtype=object)
        if inference:
            return stable_grasp_masks, {
                "count_stable": count_stable,
                "count_failed": eval_count - count_stable,
                "count_wrong": count_wrong}

        return stable_grasp_masks,#{ 
                #"count_stable": count_stable,
                #"count_failed": eval_count - count_stable,
                #"count_wrong": count_wrong}
    

    def grasp_wrench_stable_mask(
        self,
        poses: SE3Pose,
        joints: np.ndarray,
        ids: np.ndarray,
        env_state,
        nstep_lift: int = 3000,  # Steps for lifting simulation
        lift_dist: float = 0.3,  # Distance to lift
        apply_external_force: bool = False,
        ext_force_max = 3.,
        ext_torque_max = 0.5,
        enough_stable=None,
        show_progress: bool = True,
        progress_desc: str | None = None,
    ):
        """
        Evaluate grasp stability with a live tqdm progress bar.
        Shows running success rate, success/fail counts, evaluated count, and skips (if enough_stable triggers).

        If tqdm is not installed/available, progress is silently disabled.
        """
        # lazy import tqdm; fall back gracefully
        pbar = None
        if show_progress:
            try:
                from tqdm.auto import tqdm  # type: ignore

                pbar = tqdm(
                    total=len(poses),
                    desc=progress_desc or "Evaluating grasps",
                    dynamic_ncols=True,
                    leave=False,
                )
            except Exception:
                pbar = None  # disable progress if tqdm missing or no TTY

        results: List[bool] = []
        num_grasps = len(poses)
        gripper_joint_idxs = self.get_joint_idxs(
            self.gripper.get_actuator_joint_names()
        )

        count_stable = 0  # number of successful grasps
        eval_count = 0  # number of grasps actually simulated (excludes 'skipped' due to enough_stable)
        skipped_count = 0  # how many we skipped after hitting enough_stable
        contact_loss_failures = 0  # failures during lift due to contact loss

        for i in range(num_grasps):
            lift_passed = True

            # early stopping: we still append False to keep mask length, but don't simulate
            if enough_stable is not None and count_stable >= enough_stable:
                results.append(False)
                skipped_count += 1
                # progress update
                if pbar is not None:
                    sr = (count_stable / eval_count) if eval_count > 0 else 0.0
                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"succ={count_stable} fail={eval_count - count_stable} "
                        f"skipped={skipped_count} SR={sr*100:.1f}%"
                    )
                continue

            # restore environment state for each evaluation
            self.set_state(env_state)

            b2c = self.gripper.base_to_contact_transform()
            pose_processed = poses[i] @ b2c
            self.set_qpos(joints[i], gripper_joint_idxs)
            self.gripper.set_pose(self, pose_processed)

            # Update geom positions, then close
            mujoco.mj_forward(self.model, self.data)
            self.gripper.close_gripper_at(self, pose_processed)

            # --- Lift test ---
            eval_count += 1
            start_pos_lift = np.copy(self.data.mocap_pos[0, :])
            lift_target_z = start_pos_lift[2] + lift_dist
            for t in range(nstep_lift):
                alpha = t / nstep_lift
                self.data.mocap_pos[0, 2] = (
                    start_pos_lift[2] + (lift_target_z - start_pos_lift[2]) * alpha
                )
                mujoco.mj_step(self.model, self.data)
                if self.viewer:
                    if self.viewer.is_running():
                        self.viewer.sync()

                # every 100 steps, verify contact
                if (t + 1) % 100 == 0 and not self.check_gripper_contact():
                    lift_passed = False
                    contact_loss_failures += 1
                    break
            
            if lift_passed:
                if apply_external_force:
                    obj_id = self.collision_obj_id()
                    if len(obj_id) != 1 or self.object_names[ids[i]] not in obj_id:
                        lift_passed = False

                    # apply external wrench and check stability
                    # get approach vector from pose
                    pose_mat = poses[i].to_mat()
                    approach_vector = pose_mat[:3, :3] @ np.array([0, 0, -1])  # z-axis in gripper frame   
                    approach_vector /= np.linalg.norm(approach_vector)

                    # get wrench on object in approach direction
                    ext_force = np.arange(0.1, ext_force_max + 0.1, 0.1)

                    # apply force on object in approach direction
                    for f in ext_force:
                        force_vector = f * approach_vector
                        mujoco.mj_applyFT(  # type: ignore
                            self.model,
                            self.data,
                            force_vector,
                            np.array([0.0, 0.0, 0.0]),
                            mujoco.mj_name2id(  # type: ignore
                                self.model,
                                self.object_names[ids[i]] + ":body",
                                mujoco.mjtObj.mjOBJ_BODY,  # type: ignore
                            ),
                        )

                        # simulate for a short duration to see if grasp holds
                        perturb_step = 0
                        while perturb_step < 100:
                            mujoco.mj_step(self.model, self.data)
                            if self.viewer:
                                if self.viewer.is_running():
                                    self.viewer.sync()
                            perturb_step += 1

                            # check if object is still in contact with gripper
                            if not self.check_gripper_contact():
                                lift_passed = False
                                break

                
                    perturb_step = 0


            results.append(lift_passed)
            count_stable += int(lift_passed)

            # progress update
            if pbar is not None:
                sr = (count_stable / eval_count) if eval_count > 0 else 0.0
                pbar.update(1)
                pbar.set_postfix_str(
                    f"succ={count_stable} fail={eval_count - count_stable} "
                    f"skipped={skipped_count} SR={sr*100:.1f}%"
                )

        if pbar is not None:
            # final line with totals; leave the bar collapsed
            sr = (count_stable / eval_count) if eval_count > 0 else 0.0
            pbar.clear()
            pbar.close()
            # optional: final one-liner print
            print(
                f"[grasp_stable_mask] evaluated={eval_count}, succ={count_stable}, "
                f"fail={eval_count - count_stable}, skipped={skipped_count}, "
                f"contact_fail={contact_loss_failures}, SR={sr*100:.1f}%"
            )

        stable_grasp_masks = np.array(results, dtype=bool)
        return stable_grasp_masks
    
    def gen_success_labels(
        self,
        poses: SE3Pose,
        joints: np.ndarray,
        ids: np.ndarray,
        env_state,
        nstep_lift: int = 3000,  # Steps for lifting simulation
        lift_dist: float = 0.3,  # Distance to lift
        force_min = 2.1,
        force_max = 5.0,
        enough_stable=None,
        enough_unstable=None):

        if force_min > force_max:
            raise ValueError("force_min must be less or equal to force_max!")

        force_list = np.arange(force_min, force_max + 0.2, 0.2)
        success_labels = np.zeros(len(poses), dtype=float)
        stable_grasp_mask = np.zeros(len(poses), dtype=bool)

        num_stable = 0
        num_unstable = 0

        for i in range(len(poses)):

            # binary search to find lowest grasp force that results in stable grasp
            lo, hi = 0, len(force_list) - 1
            ans = -1

            while lo <= hi:
                mid = (lo + hi) // 2

                # set panda gripper force to force_list[mid]
                self.gripper.set_finger_max_force(sim=self, max_force=force_list[mid])
                
                stable_mask = self.grasp_stable_mask(
                    poses[i][None],
                    joints[i][None],
                    ids[i][None],
                    env_state
                )
        
                if stable_mask[0]:
                    ans = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
            
            # calculate success label from ans
            success_label = (force_max - force_list[ans]) / (force_max - force_min)
            success_labels[i] = success_label

            if success_label >= 0.5:
                num_stable += 1
                stable_grasp_mask[i] = True
            else:
                num_unstable += 1
            if enough_stable is not None and num_stable >= enough_stable:
                #if enough_unstable is not None and num_unstable >= enough_unstable:
                break
        
        return stable_grasp_mask, success_labels
    
    def gen_failed_grasps(
        self,
        poses: SE3Pose,
        joints: np.ndarray,
        ids: np.ndarray,
        env_state,
        nstep_lift: int = 2000,  # Steps for lifting simulation
        lift_dist: float = 0.3,  # Distance to lift
        enough_failed=None,
        show_progress: bool = True,
        progress_desc: str | None = None,
    ):
        """
        Evaluate grasp stability with a live tqdm progress bar.
        Shows running success rate, success/fail counts, evaluated count, and skips (if enough_stable triggers).

        If tqdm is not installed/available, progress is silently disabled.
        """

        failed_poses: List[SE3Pose] = []
        failed_joints = []
        failed_ids = []

        num_grasps = len(poses)
        gripper_joint_idxs = self.get_joint_idxs(
            self.gripper.get_actuator_joint_names()
        )
        count_failed = 0  # number of successful grasps

        max_iterations = 30

        sigma_rot = 0.09
        sigma_trans = 0.005
        
        # for each grasp store probability of failure when displacing it, then sample from these grasps
        prob = np.ones(num_grasps) / num_grasps
        indices = np.arange(num_grasps)
        for _ in range(max_iterations):
            if enough_failed is not None and count_failed >= enough_failed:
                break
            
            # only apply rotation with certain probability
            if np.random.rand() < 0.25:
                delta_poses: SE3Pose = SE3Pose.randn_se3_perturb(num_grasps, sigma_rot=0.12, sigma_trans=0.)
            else:
                delta_poses: SE3Pose = SE3Pose.randn_se3_perturb(num_grasps, sigma_rot=sigma_rot, sigma_trans=sigma_trans)
            candidate_poses = poses[indices] .__matmul__(delta_poses)

            # add perturbation on gripper width
            # max width on panda is 0.08
            #new_joints = joints[indices] 
            #if np.random.rand() < 1.0:
            #    current_widths = new_joints[:, 0] + 0.04 + new_joints[:, 1]  
            #    new_joints = self.gripper.width_to_joints(current_widths + np.random.normal(0., 0.01, size=current_widths.shape[0]))

            candidate_joints = joints[indices]
            candidate_ids = ids[indices]
            stable_mask = self.grasp_stable_mask(
                candidate_poses,
                candidate_joints,
                candidate_ids,
                env_state,
                with_wrong_label=True,
                apply_external_force=True,
                check_wrong_object=False,
            )

            # TODO: only append grasps with success label [1, 0], i.e., correct object failed grasp 
            failed_idx = np.where( (stable_mask[0] ==[True, False]).all(axis=1))[0] 
            success_idx = np.where( (stable_mask[0] ==[True, True]).all(axis=1))[0]
            failed_poses.append(candidate_poses[failed_idx])
            failed_joints.append(candidate_joints[failed_idx])
            failed_ids.append(candidate_ids[failed_idx])
            count_failed += failed_idx.shape[0]  
            
            # update probability
            prob[indices[success_idx]] /= 0.05
            prob = prob / np.sum(prob)
            indices = np.random.choice(num_grasps, size=num_grasps, replace=True, p=prob)

        failed_poses = [pose.to_mat() for pose in failed_poses]
        failed_poses = np.concatenate(failed_poses, axis=0)
        failed_joints = np.concatenate(failed_joints, axis=0)
        failed_ids = np.concatenate(failed_ids, axis=0)
        
        return failed_poses, failed_joints, failed_ids

    def get_obj_pose(self, object_name: str):
        mujoco.mj_forward(self.model, self.data)  # type: ignore
        jnt_adr_start = self.model.jnt("{}:joint".format(object_name)).qposadr[0].item()
        obj_position = np.copy(self.data.qpos[jnt_adr_start : jnt_adr_start + 3])
        obj_quat = np.copy(self.data.qpos[jnt_adr_start + 3 : jnt_adr_start + 7])
        return SE3Pose(obj_position, obj_quat, "wxyz")

    def grasp_collision_mask(
        self,
        poses: SE3Pose,
        joints: np.ndarray,
        with_padding: float | None = None,
    ) -> np.ndarray:
        """
        Checks collisions for each grasp pose (and optionally its 6 axis-aligned
        local translations by `with_padding`) against the current scene.

        - Bounds are checked on the *unperturbed* pose (scene-generation constraints).
        - Padding offsets are applied in the grasp pose's local frame, *before*
          base-to-contact transform.
        - If any of the 7 tests (orig + ±x/±y/±z) collides, the grasp is invalid.
        """
        if len(poses) != len(joints):
            raise ValueError(
                f"Number of poses ({len(poses)}) must match number of joint configurations ({len(joints)})."
            )
        if joints.shape[1] != len(self.gripper.get_actuator_joint_names()):
            raise ValueError(
                f"Joints array has incorrect dimension ({joints.shape[1]}), "
                f"expected {len(self.gripper.get_actuator_joint_names())}."
            )

        # build local-frame perturbations (origin + ±x/±y/±z)
        if with_padding is not None and with_padding > 0:
            p = float(with_padding)
            zero = np.zeros(3, dtype=np.float32)
            qwxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # identity (wxyz)
            deltas = [
                SE3Pose(zero, qwxyz, "wxyz"),
                SE3Pose(np.array([+p, 0.0, 0.0], np.float32), qwxyz, "wxyz"),
                SE3Pose(np.array([-p, 0.0, 0.0], np.float32), qwxyz, "wxyz"),
                SE3Pose(np.array([0.0, +p, 0.0], np.float32), qwxyz, "wxyz"),
                SE3Pose(np.array([0.0, -p, 0.0], np.float32), qwxyz, "wxyz"),
                SE3Pose(np.array([0.0, 0.0, +p], np.float32), qwxyz, "wxyz"),
                SE3Pose(np.array([0.0, 0.0, -p], np.float32), qwxyz, "wxyz"),
            ]
        else:
            deltas = [None]  # only test the original pose

        collision_free_grasps: List[bool] = []
        initial_state = self.get_state()
        gripper_joint_idxs = self.get_joint_idxs(
            self.gripper.get_actuator_joint_names()
        )
        b2c = self.gripper.base_to_contact_transform()

        for i in range(len(joints)):
            pose = poses[i]
            joint = joints[i]

            # scene-generation bounds on the *unperturbed* pose
            in_bound = (
                (pose.pos[..., 0] < 0.20)
                & (pose.pos[..., 0] > -0.20)
                & (pose.pos[..., 1] < 0.20)
                & (pose.pos[..., 1] > -0.20)
                & (pose.pos[..., 2] < 1.0)
                & (pose.pos[..., 2] > 0.0)
            ).item()
            if not in_bound:
                collision_free_grasps.append(False)
                continue

            all_clear = True
            for delta in deltas:
                # restore scene before each check
                self.set_state(initial_state)

                # local offset in grasp frame, then base->contact
                grasp_pose = pose if delta is None else (pose @ delta)
                pose_processed = grasp_pose @ b2c

                # place gripper & evaluate contacts
                self.set_qpos(joint, gripper_joint_idxs)
                self.gripper.set_pose(self, pose_processed)
                mujoco.mj_forward(self.model, self.data)

                if self.check_gripper_collision():
                    all_clear = False
                    break

            collision_free_grasps.append(all_clear)

        self.set_state(initial_state)
        return np.array(collision_free_grasps, dtype=bool)

    def to_dict(self):
        state_dict: ClutterTableState = {
            "geom_conaffinity": deepcopy(self.model.geom_conaffinity),
            "geom_contype": deepcopy(self.model.geom_contype),
            "geom_rgba": deepcopy(self.model.geom_rgba),
            "body_gravcomp": deepcopy(self.model.body_gravcomp),
            "state": self.get_state(),
        }
        dict = {
            "gripper": deepcopy(self.gripper),
            "objects": deepcopy(self.objects),
            "env_state": deepcopy(state_dict),
        }
        return dict

    @classmethod
    def from_dict(cls, state_dict, headless=True):
        gripper, obj_list, state = (
            state_dict["gripper"],
            state_dict["objects"],
            state_dict["env_state"],
        )
        env = ClutterTableEnv(gripper, obj_list, scene_randomization=False, headless=headless)
        env.set_state(state["state"])

        env.model.geom_conaffinity[:] = state["geom_conaffinity"]
        env.model.geom_contype[:] = state["geom_contype"]
        env.model.geom_rgba[:] = state["geom_rgba"]
        env.model.body_gravcomp[:] = state["body_gravcomp"]
        
        if env.viewer:
            if env.viewer.is_running():
                env.viewer.sync() 
        return env
