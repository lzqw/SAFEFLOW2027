"""Official-URDF provisional Piper Reach/Payload, simulation only.

No Panda model/weights are loaded. User identified standard Piper/gripper;
TCP/dynamics are provisional, not physically calibrated. A trained candidate
must carry this simulation identity and later pass model/calibration matching.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as bullet


@dataclass
class PiperSimConfig:
    task: str = 'Reach'
    model_commit: str = 'ac41fcbcdda598f01b51cf6175ed9a24d0dacadc'
    horizon: int = 100
    physics_dt_s: float = .002
    substeps: int = 20
    displacement_scale_m: float = .01
    safety_distance_m: float = .015
    success_distance_m: float = .03
    obstacle_radius_m: float = .045
    payload_half_extents_m: tuple = (.025, .018, .025)
    payload_mass_kg: float = .05
    tcp_from_flange_m: tuple = (0., 0., .1358)
    start_tcp_base_m: tuple = (.32, -.12, .28)
    goal_tcp_base_m: tuple = (.32, .12, .28)
    workspace_min_m: tuple = (.05, -.45, .05)
    workspace_max_m: tuple = (.65, .45, .65)
    hardware_matched: bool = False

    def __post_init__(self):
        if self.task not in ('Reach', 'Payload'):
            raise ValueError('Task must be Reach or Payload')
        if self.horizon <= 0 or self.substeps <= 0 or self.physics_dt_s <= 0:
            raise ValueError('Invalid simulated horizon/control timing')
        if self.hardware_matched:
            raise ValueError('This provisional fixture cannot assert hardware matching')


class PiperSimulation(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, asset_receipt: dict, config: PiperSimConfig | None = None):
        super().__init__()
        self.config = config or PiperSimConfig()
        if asset_receipt['model_commit'] != self.config.model_commit:
            raise ValueError('Pinned model mismatch')
        urdf = Path(asset_receipt['resolved_urdf'])
        if hashlib.sha256(urdf.read_bytes()).hexdigest() != asset_receipt['resolved_sha256']:
            raise ValueError('Resolved official model digest mismatch')
        self.asset_receipt = asset_receipt
        self.client = bullet.connect(bullet.DIRECT)
        self.b = lambda name, *a, **k: getattr(bullet, name)(*a, physicsClientId=self.client, **k)
        self.b('setGravity', 0., 0., -9.81)
        self.b('setTimeStep', self.config.physics_dt_s)
        self.robot = self.b('loadURDF', str(urdf), useFixedBase=True,
                            flags=bullet.URDF_USE_INERTIA_FROM_FILE)
        names = {self.b('getJointInfo', self.robot, j)[1].decode(): j
                 for j in range(self.b('getNumJoints', self.robot))}
        self.arm_joints = [names['joint'+str(j)] for j in range(1, 7)]
        self.flange = names['joint6']
        self.finger_joints = [names['joint7'], names['joint8']]
        self.protected_links = tuple(range(self.b('getNumJoints', self.robot)))
        shape = self.b('createCollisionShape', bullet.GEOM_SPHERE,
                       radius=self.config.obstacle_radius_m)
        self.obstacle = self.b('createMultiBody', baseMass=0., baseCollisionShapeIndex=shape,
                              basePosition=[2., 2., 2.])
        self.payload = None
        if self.config.task == 'Payload':
            shape = self.b('createCollisionShape', bullet.GEOM_BOX,
                           halfExtents=self.config.payload_half_extents_m)
            self.payload = self.b('createMultiBody', baseMass=self.config.payload_mass_kg,
                                 baseCollisionShapeIndex=shape, basePosition=[2., 2., 2.])
            # Exactly the allowed held-object contact set; obstacle tests are
            # separate and never disabled. Other payload-arm contacts remain.
            for link in [self.flange, names['joint6_to_gripper_base'], *self.finger_joints]:
                self.b('setCollisionFilterPair', self.robot, self.payload, link, -1, 0)
            self.b('createConstraint', self.robot, self.flange, self.payload, -1,
                   bullet.JOINT_FIXED, [0., 0., 0.], self.config.tcp_from_flange_m, [0., 0., 0.])
        self.action_space = spaces.Box(-1., 1., shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(34,), dtype=np.float32)
        self.total_control_intervals = 0
        self.total_physics_substeps = 0
        self._done = True

    @property
    def control_dt(self):
        return self.config.physics_dt_s * self.config.substeps

    def tcp_pose(self):
        state = self.b('getLinkState', self.robot, self.flange, computeForwardKinematics=1)
        # Link-frame pose, not the inertial centre of mass.
        position, orientation = bullet.multiplyTransforms(state[4], state[5],
            self.config.tcp_from_flange_m, [0., 0., 0., 1.])
        return np.asarray(position), np.asarray(orientation)

    def _joint_target(self, tcp, orientation):
        rotation = np.asarray(bullet.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        flange = np.asarray(tcp)-rotation@np.asarray(self.config.tcp_from_flange_m)
        solution = self.b('calculateInverseKinematics', self.robot, self.flange,
                         flange.tolist(), list(orientation), maxNumIterations=200,
                         residualThreshold=1e-6)
        return np.asarray(solution[:6])

    def _motor(self, target):
        self.b('setJointMotorControlArray', self.robot, self.arm_joints, bullet.POSITION_CONTROL,
               targetPositions=list(target), forces=[80.]*6)
        self.b('setJointMotorControlArray', self.robot, self.finger_joints, bullet.POSITION_CONTROL,
               targetPositions=[.018, -.018], forces=[10.]*2)

    def distances(self):
        self.b('performCollisionDetection')
        points = self.b('getClosestPoints', self.robot, self.obstacle, distance=2.)
        arm = min((float(p[8]) for p in points if p[3] in self.protected_links), default=2.)
        payload = 2.
        if self.payload is not None:
            payload = min((float(p[8]) for p in self.b('getClosestPoints', self.payload,
                          self.obstacle, distance=2.)), default=2.)
        return float(arm), float(payload)

    def _obs(self):
        states = [self.b('getJointState', self.robot, j) for j in self.arm_joints]
        tcp, _ = self.tcp_pose()
        if self.payload is None:
            pp, pq, dims, mass = np.zeros(3), np.zeros(4), np.zeros(3), 0.
        else:
            pp, pq = self.b('getBasePositionAndOrientation', self.payload)
            dims, mass = self.config.payload_half_extents_m, self.config.payload_mass_kg
        return np.concatenate([[s[0]/np.pi for s in states], [s[1]/4. for s in states],
            tcp, self.goal, self.obstacle_position, [self.config.obstacle_radius_m],
            [max(0., 1.-self.elapsed/self.config.horizon)], pp, pq, dims, [mass]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.elapsed, self._done = 0, False
        self.goal = np.asarray(self.config.goal_tcp_base_m, dtype=float)
        start = np.asarray(self.config.start_tcp_base_m, dtype=float)
        self.obstacle_position = (start+self.goal)/2
        if not (options or {}).get('fixed_layout', False):
            offset = self.np_random.uniform([-.015, -.01, -.015], [.015, .01, .015])
            start, self.goal = start+offset, self.goal+offset
            self.obstacle_position += self.np_random.uniform(-.01, .01, size=3)
        for joint, angle in zip(self.arm_joints, [0., 1., -1., 0., .5, 0.]):
            self.b('resetJointState', self.robot, joint, angle)
        for joint, angle in zip(self.finger_joints, [.018, -.018]):
            self.b('resetJointState', self.robot, joint, angle)
        _, self.orientation = self.tcp_pose()
        target = self._joint_target(start, self.orientation)
        for joint, angle in zip(self.arm_joints, target):
            self.b('resetJointState', self.robot, joint, float(angle))
        self._motor(target)
        self.b('resetBasePositionAndOrientation', self.obstacle, self.obstacle_position, [0.,0.,0.,1.])
        if self.payload is not None:
            pp, pq = self.tcp_pose()
            self.b('resetBasePositionAndOrientation', self.payload, pp, pq)
            self.b('resetBaseVelocity', self.payload, [0.,0.,0.], [0.,0.,0.])
        arm, payload = self.distances()
        tcp, _ = self.tcp_pose()
        reset_error = float(np.linalg.norm(tcp-start))
        if min(arm, payload) <= self.config.safety_distance_m or reset_error > .025:
            self._done = True
            raise RuntimeError(f'Unsafe/unreachable provisional Piper reset: arm={arm}, payload={payload}, error={reset_error}')
        self.previous_goal_distance = float(np.linalg.norm(tcp-self.goal))
        return self._obs(), dict(minimum_distance=min(arm,payload), arm_distance=arm,
            payload_distance=payload, reset_ik_error_m=reset_error, hardware_matched=False)

    def step(self, action):
        if self._done:
            raise RuntimeError('Reset after terminal action')
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError('Expected three finite normalized actions')
        tcp, _ = self.tcp_pose()
        desired = tcp+np.clip(action,-1.,1.)*self.config.displacement_scale_m
        clipped = np.clip(desired,self.config.workspace_min_m,self.config.workspace_max_m)
        self._motor(self._joint_target(clipped,self.orientation))
        arm_min, payload_min = self.distances()
        for _ in range(self.config.substeps):
            self.b('stepSimulation')
            self.total_physics_substeps += 1
            arm,payload=self.distances()
            arm_min,payload_min=min(arm_min,arm),min(payload_min,payload)
        self.total_control_intervals += 1
        self.elapsed += 1
        minimum=min(arm_min,payload_min)
        violation=minimum < self.config.safety_distance_m
        tcp,_=self.tcp_pose()
        distance=float(np.linalg.norm(tcp-self.goal))
        success=distance < self.config.success_distance_m and not violation
        terminal=bool(success or violation or self.elapsed>=self.config.horizon)
        reward=5.*(self.previous_goal_distance-distance)-.01+10.*success-10.*violation
        self.previous_goal_distance=distance
        self._done=terminal
        return self._obs(),float(reward),terminal,False,dict(
            cost=float(violation),safe_success=bool(success),violation=bool(violation),
            collision=bool(minimum<=0.),arm_minimum_distance=arm_min,
            payload_minimum_distance=payload_min,minimum_distance=minimum,
            local_margin=self.config.safety_distance_m-minimum,
            true_horizon_terminal=self.elapsed>=self.config.horizon,
            workspace_clipped=bool(np.any(np.abs(desired-clipped)>1e-9)),
            control_dt=self.control_dt,physical_connection=False)

    def definition(self):
        return dict(robot_family='Piper',config=asdict(self.config),
            model_source=self.asset_receipt['model_repository'],
            source_urdf_sha256=self.asset_receipt['source_sha256'],
            observation_units='joint rad/pi; joint velocity rad/s /4; positions/extent m; quaternion xyzw; mass kg',
            action='normalized deltaXYZ in robot base, fixed reset orientation',
            task_geometry='provisional reach/payload around spherical obstacle, not calibrated real robot',
            physical_connection=False,hardware_matched=False)

    def close(self):
        if bullet.isConnected(self.client):
            bullet.disconnect(self.client)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-receipt',type=Path,required=True)
    parser.add_argument('--task',choices=['Reach','Payload'],default='Reach')
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Preserve earlier simulation receipts')
    env=PiperSimulation(json.loads(args.assets_receipt.read_text()),PiperSimConfig(task=args.task))
    rows=[]
    try:
        _,reset=env.reset(seed=62027,options={'fixed_layout':True})
        for i in range(args.steps):
            _,reward,done,_,info=env.step([0.,0.,0.])
            rows.append(dict(step=i+1,reward=reward,**info))
            if done:
                break
        receipt=dict(status='completed_official_model_simulation_smoke',definition=env.definition(),
            reset=reset,records=rows,new_training_transitions=0,
            new_simulation_control_intervals=env.total_control_intervals,
            new_physics_substeps=env.total_physics_substeps,
            real_interactions=0,candidate_policies_generated=0)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({k:v for k,v in receipt.items() if k not in ['records','definition','reset']}))
    finally:
        env.close()


if __name__=='__main__':
    main()
