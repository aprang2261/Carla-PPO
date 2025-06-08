import os
import sys
import random
import time
import numpy as np
import gymnasium as gym
import torch
import torch_geometric.nn as gnn
import torch.nn as nn
import carla
import traceback
import sumolib

from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# --- 설정 상수 ---
# CARLA 서버 설정
CARLA_HOST = "localhost"
CARLA_PORT = 2000
MAP_NAME = "Town05"  # 사용하실 맵 이름

# 시뮬레이션 설정
FIXED_DELTA_SECONDS = 0.1
MAX_STEPS_PER_EPISODE = 3600  # 3600 * 0.1초 = 6분
TICKS_PER_AGENT_STEP = 10  # 에이전트 행동 1번 당 시뮬레이션 10틱 (1초)
MIN_PHASE_DURATION_STEPS = 5
YELLOW_LIGHT_DURATION_STEPS = 3
NUM_VEHICLES = 150

# 모델 및 통계 파일 경로 (SUMO 학습 결과 경로에 맞게 수정 필요)
MODEL_PATH = "../Learning_SUMO/info/model_checkpoints/ppo_sumo_4phase_1000160_steps.zip"
STATS_PATH = "../Learning_SUMO/info/model_checkpoints/vecnormalize_1000160_steps.pkl"

# GNN/관측 공간 관련 상수 (SUMO 학습 때와 동일하게 유지)
NODE_FEATURE_DIM = 10
GNN_ADJACENCY_THRESHOLD = 200.0
QUEUE_COUNT_OBS_MAX = 50.0
MAX_WAIT_TIME_OBS_MAX = 300.0  # 300틱 = 30초
ELAPSED_TIME_OBS_MAX = 200.0  # 200 에이전트 스텝

# SUMO 학습 시 사용한 실제 TLS ID 목록 (이것이 관측 공간 크기를 결정합니다!)
REAL_TLS_IDS_SUMO_LEARNING = ['1050', '1070', '1148', '1175', '1260', '139', '421', '509', '53', '599', '751', '829',
                              '905', '943', '965']


# --- GAT 특징 추출기 (SUMO 학습 때와 완전히 동일) ---
class GATFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, adjacency_matrix: np.ndarray, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]
        self.node_feature_dim = NODE_FEATURE_DIM
        assert obs_dim % self.node_feature_dim == 0
        self.num_tls = obs_dim // self.node_feature_dim
        self.conv1 = gnn.GATConv(self.node_feature_dim, 32, heads=4, concat=True)
        self.conv2 = gnn.GATConv(32 * 4, features_dim, heads=1, concat=False)
        self.relu = nn.ReLU()
        sparse_matrix = coo_matrix(adjacency_matrix)
        edge_index, _ = from_scipy_sparse_matrix(sparse_matrix)
        self.register_buffer("edge_index", edge_index)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        device = observations.device
        x = observations.view(batch_size * self.num_tls, self.node_feature_dim)
        edge_index_batch = self.edge_index.to(device)
        edge_indices = [edge_index_batch + i * self.num_tls for i in range(batch_size)]
        edge_index_batched = torch.cat(edge_indices, dim=1)
        x = self.relu(self.conv1(x, edge_index_batched))
        x = self.relu(self.conv2(x, edge_index_batched))
        batch_vector = torch.arange(batch_size, device=device).repeat_interleave(self.num_tls)
        return gnn.global_mean_pool(x, batch_vector)


# --- CARLA 환경 어댑터 ---
class CarlaValidationEnv(gym.Env):
    # CARLA에서 제어할 논리적 페이즈 상태
    PHASE_NS_GREEN, PHASE_NS_YELLOW, PHASE_EW_GREEN, PHASE_EW_YELLOW = 0, 1, 2, 3

    # SUMO 학습 모델의 액션 정의 (0, 1, 2, 3)
    ACTION_EW_S, ACTION_EW_L, ACTION_NS_S, ACTION_NS_L = 0, 1, 2, 3

    def __init__(self):
        super().__init__()
        # CARLA 클라이언트 연결
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        if self.world.get_map().name.split('/')[-1] != MAP_NAME:
            self.world = self.client.load_world(MAP_NAME)

        # 동기 모드 설정
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(settings)

        self.tm = self.client.get_trafficmanager()
        self.tm.set_synchronous_mode(True)

        self._cleanup_actors()
        self.vehicles = []
        self.vehicle_wait_times = {}
        self.vehicle_queue_times = {}

        self.node_feature_dim = NODE_FEATURE_DIM

        # num_junctions는 _setup_junctions에서 최종 결정되므로 임시 초기화
        self.num_junctions = len(REAL_TLS_IDS_SUMO_LEARNING)

        # Phase tracking variables, 크기는 num_junctions에 따라 결정됨
        self.tl_phases = np.zeros(self.num_junctions, dtype=int)
        self.last_phase_change_step = np.zeros(self.num_junctions, dtype=int)
        self.current_phase_action = np.zeros(self.num_junctions, dtype=int)

        # 신호등 그룹화 및 공간 정의
        self._setup_junctions()  # 이 시점에서 self.num_junctions가 확정됨
        self._define_spaces()  # 확정된 self.num_junctions를 기반으로 공간 정의

        self.current_episode_step = 0

    def _cleanup_actors(self):
        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter('vehicle.*')
        for vehicle in vehicle_list:
            if vehicle.is_alive:
                try:
                    vehicle.destroy()
                except RuntimeError:
                    pass
        self.vehicles = []

    def _get_actor_orientation(self, actor):
        """액터의 회전값(yaw)을 바탕으로 남북(NS) 방향인지 동서(EW) 방향인지 판단합니다."""
        yaw = actor.get_transform().rotation.yaw
        # CARLA의 yaw는 x축(동쪽)을 기준으로 반시계 방향
        # 0도: 동쪽, 90도: 북쪽, 180도: 서쪽, -90도(270도): 남쪽
        # NS 방향: 북향(90도 근처) 또는 남향(-90도/270도 근처)
        if (yaw >= 45 and yaw < 135) or (yaw < -45 and yaw > -135):
            return "NS"
        else:  # 동쪽 또는 서쪽 (0도, 180도, -180도 근처)
            return "EW"

    def _setup_junctions(self):
        sumo_tls_ids = REAL_TLS_IDS_SUMO_LEARNING

        sumo_net_path = "../Learning_SUMO/sumo_files/Town05.with_tls.net.xml"
        if not os.path.exists(sumo_net_path):
            raise FileNotFoundError(f"SUMO 네트워크 파일 '{sumo_net_path}'를 찾을 수 없습니다. 학습 스크립트를 실행하여 생성해주세요.")

        sumo_net = sumolib.net.readNet(sumo_net_path)
        offsetX, offsetY = sumo_net.getLocationOffset()
        print(f"SUMO netOffset: x={offsetX}, y={offsetY}")

        sumo_junction_locations = {jid: sumo_net.getNode(jid).getCoord() for jid in sumo_tls_ids}

        all_tls_actors = list(self.world.get_actors().filter("traffic.traffic_light"))

        actor_to_junction_map = {}
        for carla_tl in all_tls_actors:
            carla_loc = carla_tl.get_location()

            min_dist = float('inf')
            closest_junction_id = None
            for jid, jloc in sumo_junction_locations.items():
                carla_equiv_x = jloc[0] - offsetX
                carla_equiv_y = -(jloc[1] - offsetY)
                dist = carla_loc.distance(carla.Location(x=carla_equiv_x, y=carla_equiv_y, z=carla_loc.z))
                if dist < min_dist:
                    min_dist = dist
                    closest_junction_id = jid

            if closest_junction_id and min_dist < GNN_ADJACENCY_THRESHOLD:
                actor_to_junction_map[carla_tl.id] = closest_junction_id

        junction_groups = {jid: [] for jid in sumo_tls_ids}
        for actor_id, junction_id in actor_to_junction_map.items():
            actor = self.world.get_actor(actor_id)
            if actor:
                junction_groups[junction_id].append(actor)

        self.junctions = []
        self.junction_sumo_ids = []
        for jid in sumo_tls_ids:
            group = junction_groups.get(jid, [])
            self.junctions.append(sorted(group, key=lambda x: x.id))
            self.junction_sumo_ids.append(jid)

        self.num_junctions = len(self.junctions)  # 실제 매핑된 교차로 수로 업데이트

        print(f"총 {len(all_tls_actors)}개의 신호등 액터 중 SUMO 기준 {self.num_junctions}개의 교차로로 그룹화 시도.")

        empty_groups = [self.junction_sumo_ids[i] for i, group in enumerate(self.junctions) if not group]
        if empty_groups:
            print(f"경고: 다음 {len(empty_groups)}개 교차로 그룹이 CARLA 신호등 액터와 매칭되지 않아 비어있습니다: {empty_groups}")
            print("이 교차로들은 관측/행동 계산 시 0으로 처리됩니다.")

        self.actor_orientations = {}
        # 각 교차로 그룹 내의 신호등 액터들을 NS/EW 직진/좌회전 방향으로 분류 (추정치)
        # 이 부분은 CARLA의 TrafficLight Actor의 실제 역할을 추정하는 고도로 복잡한 로직이 필요합니다.
        # 여기서는 단순히 NS/EW 방향으로 크게 나누고, 직진/좌회전은 "모든" 해당 방향 신호등에 동일하게 적용하는 방식으로 단순화합니다.
        # 즉, 학습 모델이 NS_S 액션을 내든 NS_L 액션을 내든, CARLA에서는 해당 교차로의 모든 NS 방향 신호등이 Green이 됩니다.
        self.junction_lane_groups = {jid: {'ns_s': [], 'ns_l': [], 'ew_s': [], 'ew_l': []} for jid in
                                     self.junction_sumo_ids}

        for j_idx, junction_group in enumerate(self.junctions):
            current_sumo_tl_id = self.junction_sumo_ids[j_idx]
            for tl_actor in junction_group:
                orientation = self._get_actor_orientation(tl_actor)
                self.actor_orientations[tl_actor.id] = orientation  # 오리엔테이션 저장

                # CARLA에서는 직진/좌회전 신호등을 정확히 분리하기 어렵습니다.
                # 여기서는 모든 NS 방향 신호등을 'ns_s' (직진) 그룹에, EW 방향 신호등을 'ew_s' (직진) 그룹에 넣습니다.
                # 모델이 좌회전 액션을 내더라도, 실제 신호등은 직진과 동일하게 작동하게 됩니다.
                # 더 정확한 구현을 위해서는 CARLA 웨이포인트 및 차선 정보 분석이 필요합니다.
                if orientation == "NS":
                    self.junction_lane_groups[current_sumo_tl_id]['ns_s'].append(tl_actor)
                    # self.junction_lane_groups[current_sumo_tl_id]['ns_l'].append(tl_actor) # 필요시 분리
                else:  # EW
                    self.junction_lane_groups[current_sumo_tl_id]['ew_s'].append(tl_actor)
                    # self.junction_lane_groups[current_sumo_tl_id]['ew_l'].append(tl_actor) # 필요시 분리

        # num_junctions가 확정된 후, 상태 추적 변수들의 크기를 재할당 (중요!)
        self.tl_phases = np.zeros(self.num_junctions, dtype=int)
        self.last_phase_change_step = np.zeros(self.num_junctions, dtype=int)
        self.current_phase_action = np.zeros(self.num_junctions, dtype=int)

        # 초기 페이즈 설정 (NS_S 액션에 해당하는 NS_GREEN 페이즈)
        self.tl_phases.fill(self.PHASE_NS_GREEN)
        self.last_phase_change_step.fill(0)
        self.current_phase_action.fill(self.ACTION_NS_S)

    def _define_spaces(self):
        """SUMO 환경과 동일한 관측/행동 공간을 정의합니다."""
        num_features = NODE_FEATURE_DIM
        obs_space_size = self.num_junctions * num_features
        obs_low = np.zeros(obs_space_size, dtype=np.float32)
        obs_high = np.full(obs_space_size, 1.0, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([4] * self.num_junctions)  # 4가지 액션

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._cleanup_actors()
        self.vehicle_wait_times.clear()
        self.vehicle_queue_times.clear()

        # 차량 스폰
        blueprint_library = self.world.get_blueprint_library().filter('vehicle.*')
        spawn_points = self.world.get_map().get_spawn_points()

        vehicles_spawned = 0
        max_spawn_attempts = NUM_VEHICLES * 2
        attempts = 0

        while vehicles_spawned < NUM_VEHICLES and attempts < max_spawn_attempts:
            try:
                blueprint = random.choice(blueprint_library)
                # Traffic Manager를 통해 자율 주행 설정 시 충돌 방지 노력
                spawn_point = random.choice(spawn_points)
                vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
                if vehicle:
                    vehicle.set_autopilot(True, self.tm.get_port())
                    self.vehicles.append(vehicle)
                    vehicles_spawned += 1
            except Exception:
                pass
            attempts += 1

        print(f"총 {vehicles_spawned}대의 차량 스폰 완료.")

        # 초기 상태 설정
        self.tl_phases.fill(self.PHASE_NS_GREEN)
        self.current_phase_action.fill(self.ACTION_NS_S)

        self.last_phase_change_step.fill(0)
        self.current_episode_step = 0
        self._apply_phase_to_lights(self.current_phase_action, transition=False)  # 초기에는 전환 없이 즉시 적용

        # 시뮬레이션 안정화
        for _ in range(50):  # 50틱 진행 (5초)
            self.world.tick()

        return self._get_observation(), {}

    def step(self, actions):
        """에이전트의 행동(num_junctions 개)을 받아 CARLA 환경에 적용합니다."""

        new_logical_phases = np.copy(self.tl_phases)
        for i in range(self.num_junctions):
            if not self.junctions[i]:  # 비어있는 그룹은 처리하지 않음
                continue

            prev_agent_action = self.current_phase_action[i]
            new_agent_action = actions[i]  # 현재 스텝에서 에이전트가 내린 액션

            # 이전 메인 페이즈 (NS_S/L -> NS_GREEN, EW_S/L -> EW_GREEN)
            prev_main_phase_type = self.PHASE_EW_GREEN if prev_agent_action in [self.ACTION_EW_S,
                                                                                self.ACTION_EW_L] else self.PHASE_NS_GREEN
            # 새 메인 페이즈
            new_main_phase_type = self.PHASE_EW_GREEN if new_agent_action in [self.ACTION_EW_S,
                                                                              self.ACTION_EW_L] else self.PHASE_NS_GREEN

            time_in_current_phase = (self.current_episode_step - self.last_phase_change_step[i])

            # 1. 현재 Yellow 신호 중이고, Yellow 지속 시간이 지났으면 다음 Green으로 전환
            if self.tl_phases[i] in [self.PHASE_NS_YELLOW, self.PHASE_EW_YELLOW] and \
                    time_in_current_phase >= (YELLOW_LIGHT_DURATION_STEPS * TICKS_PER_AGENT_STEP):

                # Yellow 다음은 반대 방향 Green (CARLA에는 All-Red가 없으므로 Yellow -> Red/Opposite Green)
                if self.tl_phases[i] == self.PHASE_NS_YELLOW:
                    new_logical_phases[i] = self.PHASE_EW_GREEN
                else:  # EW_YELLOW
                    new_logical_phases[i] = self.PHASE_NS_GREEN

                self.last_phase_change_step[i] = self.current_episode_step
                self.current_phase_action[i] = new_agent_action  # 최종 Green 상태로 전환됐으므로 액션 업데이트

            # 2. 현재 Green 신호 중이고, 에이전트가 다른 방향으로 전환을 원하며, 최소 Green 지속 시간을 넘겼으면 Yellow로 전환 시작
            # 여기서 '다른 방향으로 전환을 원함'은 메인 페이즈 타입(NS vs EW)이 변경되는 것을 의미합니다.
            elif self.tl_phases[i] in [self.PHASE_NS_GREEN, self.PHASE_EW_GREEN] and \
                    new_main_phase_type != prev_main_phase_type and \
                    time_in_current_phase >= (MIN_PHASE_DURATION_STEPS * TICKS_PER_AGENT_STEP):

                if self.tl_phases[i] == self.PHASE_NS_GREEN:
                    new_logical_phases[i] = self.PHASE_NS_YELLOW
                else:  # EW_GREEN
                    new_logical_phases[i] = self.PHASE_EW_YELLOW

                self.last_phase_change_step[i] = self.current_episode_step
                # Yellow 상태에서는 current_phase_action은 이전 Green 상태를 유지 (모델의 액션은 반영하지 않음)

            # 3. 그 외의 경우 (현재 Yellow인데 지속 시간 안 됐거나, Green인데 전환 요청 없거나 최소 지속시간 안 됐거나)
            # --> 페이즈는 그대로 유지.
            # 이 경우 모델이 같은 메인 페이즈 내에서 액션(예: NS_S -> NS_L)을 변경하더라도
            # 실제 신호등 상태 (NS_GREEN)는 변하지 않습니다.
            # 그러나 current_phase_action은 새로운 액션으로 업데이트하여 관측에 반영합니다.
            if new_logical_phases[i] == self.tl_phases[i]:  # 페이즈가 바뀌지 않았을 경우
                self.current_phase_action[i] = new_agent_action

        self.tl_phases = new_logical_phases  # 업데이트된 페이즈 적용

        # 실제 시뮬레이션 틱 진행
        for _ in range(TICKS_PER_AGENT_STEP):
            self._apply_phase_to_lights(self.current_phase_action)  # <-- 현재 액션에 따라 신호등 적용
            self.world.tick()
            self.current_episode_step += 1

        obs = self._get_observation()
        done = self.current_episode_step >= MAX_STEPS_PER_EPISODE

        info = {
            "current_episode_step": self.current_episode_step,
            "current_logical_phases": self.tl_phases.copy(),
            "actions_applied": actions.copy()
        }
        return obs, 0.0, done, False, info

    def _apply_phase_to_lights(self, actions_for_current_step: np.ndarray, transition=False):
        """
        에이전트가 선택한 4가지 액션에 따라 신호등 페이즈를 적용합니다.
        transition=True이면 황색/적색 전환을 수행합니다 (step 함수에서 제어).
        """
        for i, current_sumo_tl_id in enumerate(self.junction_sumo_ids):
            # 만약 해당 교차로에 매핑된 신호등 액터가 없으면 건너뜀
            if not self.junctions[i]:
                continue

            logical_phase = self.tl_phases[i]  # 현재 이 교차로가 표시해야 할 논리적 페이즈 (Green/Yellow)

            # 모든 신호등을 Red로 초기화
            all_lights = (self.junction_lane_groups[current_sumo_tl_id]['ns_s'] +
                          self.junction_lane_groups[current_sumo_tl_id]['ns_l'] +  # 현재는 비어있거나 ns_s와 동일
                          self.junction_lane_groups[current_sumo_tl_id]['ew_s'] +
                          self.junction_lane_groups[current_sumo_tl_id]['ew_l'])  # 현재는 비어있거나 ew_s와 동일
            for tl_actor in all_lights:
                tl_actor.set_state(carla.TrafficLightState.Red)

            # 논리적 페이즈에 따라 Green 또는 Yellow 신호 설정
            if logical_phase == self.PHASE_NS_GREEN:
                # NS 직진/좌회전 그룹 모두 Green (CARLA 한계로 직진/좌회전 동시 Green)
                for tl_actor in self.junction_lane_groups[current_sumo_tl_id]['ns_s']:
                    tl_actor.set_state(carla.TrafficLightState.Green)
                for tl_actor in self.junction_lane_groups[current_sumo_tl_id]['ns_l']:  # 현재는 ns_s와 동일
                    tl_actor.set_state(carla.TrafficLightState.Green)
            elif logical_phase == self.PHASE_EW_GREEN:
                # EW 직진/좌회전 그룹 모두 Green
                for tl_actor in self.junction_lane_groups[current_sumo_tl_id]['ew_s']:
                    tl_actor.set_state(carla.TrafficLightState.Green)
                for tl_actor in self.junction_lane_groups[current_sumo_tl_id]['ew_l']:  # 현재는 ew_s와 동일
                    tl_actor.set_state(carla.TrafficLightState.Green)
            elif logical_phase == self.PHASE_NS_YELLOW:
                # NS 방향 모두 Yellow
                for tl_actor in (self.junction_lane_groups[current_sumo_tl_id]['ns_s'] +
                                 self.junction_lane_groups[current_sumo_tl_id]['ns_l']):
                    tl_actor.set_state(carla.TrafficLightState.Yellow)
            elif logical_phase == self.PHASE_EW_YELLOW:
                # EW 방향 모두 Yellow
                for tl_actor in (self.junction_lane_groups[current_sumo_tl_id]['ew_s'] +
                                 self.junction_lane_groups[current_sumo_tl_id]['ew_l']):
                    tl_actor.set_state(carla.TrafficLightState.Yellow)

    def _get_observation(self):
        """CARLA 세계에서 num_junctions 개 교차로에 대한 관측을 생성합니다."""
        obs_list = []
        all_vehicles = self.world.get_actors().filter('vehicle.*')

        for i, junction_group in enumerate(self.junctions):
            if not junction_group:  # 비어있는 그룹은 0으로 처리
                obs_list.extend([0.0] * self.node_feature_dim)
                continue

            center_loc = self._get_junction_center(junction_group)

            # SUMO 환경의 큐/대기 시간 정의 (4개 방향)
            ns_s_q, ns_l_q, ew_s_q, ew_l_q = 0, 0, 0, 0
            ns_s_wait, ns_l_wait, ew_s_wait, ew_l_wait = 0, 0, 0, 0

            for v in all_vehicles:
                # 차량이 교차로 반경 50m 이내에 있는지 확인 (SUMO 환경과 유사하게)
                if v.get_location().distance(center_loc) > 50.0:
                    continue

                # 차량의 속도 기준으로 대기 시간 계산
                if v.get_velocity().length() < 0.1:  # 정지 상태
                    self.vehicle_wait_times[v.id] = self.vehicle_wait_times.get(v.id, 0) + 1  # 틱 단위
                elif v.id in self.vehicle_wait_times:
                    del self.vehicle_wait_times[v.id]

                wait_time_ticks = self.vehicle_wait_times.get(v.id, 0)

                v_transform = v.get_transform()
                v_loc = v_transform.location
                v_yaw = v_transform.rotation.yaw

                relative_loc_x = v_loc.x - center_loc.x
                relative_loc_y = v_loc.y - center_loc.y

                # 차량의 현재 yaw를 기반으로 NS/EW 방향 판단
                # 그리고 교차로 중심으로부터의 상대 위치로 직진/좌회전 대략적 추정

                # NS 방향 (북향 또는 남향)
                if (v_yaw > 45 and v_yaw < 135) or (v_yaw < -45 and v_yaw > -135):
                    # 직진 (교차로 중앙 X축 범위 내에 있다면)
                    # 여기서는 15m를 임계값으로 사용했지만, 맵과 교차로 구조에 따라 조정 필요
                    if abs(relative_loc_x) < 15:
                        ns_s_q += 1
                        ns_s_wait = max(ns_s_wait, wait_time_ticks)
                    else:  # 좌회전 (교차로 중앙 X축 범위에서 벗어났다면)
                        ns_l_q += 1
                        ns_l_wait = max(ns_l_wait, wait_time_ticks)
                # EW 방향 (동향 또는 서향)
                elif (v_yaw >= -45 and v_yaw <= 45) or (v_yaw >= 135 and v_yaw <= 180) or (
                        v_yaw <= -135 and v_yaw >= -180):
                    # 직진
                    if abs(relative_loc_y) < 15:
                        ew_s_q += 1
                        ew_s_wait = max(ew_s_wait, wait_time_ticks)
                    else:  # 좌회전
                        ew_l_q += 1
                        ew_l_wait = max(ew_l_wait, wait_time_ticks)

            # SUMO 학습 코드의 10개 특징에 맞춰 관측값 정규화
            # 큐 카운트 (0~3)
            obs_list.append(min(ns_s_q / QUEUE_COUNT_OBS_MAX, 1.0))
            obs_list.append(min(ns_l_q / QUEUE_COUNT_OBS_MAX, 1.0))
            obs_list.append(min(ew_s_q / QUEUE_COUNT_OBS_MAX, 1.0))
            obs_list.append(min(ew_l_q / QUEUE_COUNT_OBS_MAX, 1.0))

            # 대기 시간 (4~7)
            obs_list.append(min(ns_s_wait / MAX_WAIT_TIME_OBS_MAX, 1.0))
            obs_list.append(min(ns_l_wait / MAX_WAIT_TIME_OBS_MAX, 1.0))
            obs_list.append(min(ew_s_wait / MAX_WAIT_TIME_OBS_MAX, 1.0))
            obs_list.append(min(ew_l_wait / MAX_WAIT_TIME_OBS_MAX, 1.0))

            # 현재 페이즈 (8)
            # PPO는 4가지 액션 (0,1,2,3)으로 학습되었으므로, 현재 PPO가 기대하는 액션 값을 페이즈로 사용합니다.
            obs_list.append(self.current_phase_action[i] / 3.0)  # 0,1,2,3 -> 0, 1/3, 2/3, 1로 정규화

            # 경과 시간 (9)
            elapsed_time_steps = (self.current_episode_step - self.last_phase_change_step[i]) / TICKS_PER_AGENT_STEP
            obs_list.append(min(elapsed_time_steps / ELAPSED_TIME_OBS_MAX, 1.0))

        return np.array(obs_list, dtype=np.float32)

    def _get_junction_center(self, junction_group):
        """교차로 그룹의 중심 위치를 계산합니다."""
        x = sum(tl.get_location().x for tl in junction_group) / len(junction_group)
        y = sum(tl.get_location().y for tl in junction_group) / len(junction_group)
        z = sum(tl.get_location().z for tl in junction_group) / len(junction_group)
        return carla.Location(x, y, z)

    def close(self):
        self._cleanup_actors()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        print("CARLA environment closed.")


def create_carla_adjacency_matrix(net_file: str, tls_ids: list, threshold: float) -> np.ndarray:
    """
    SUMO 네트워크 파일의 좌표를 직접 사용하여 인접 행렬을 생성합니다.
    이 방식은 CARLA에 해당 신호등 액터가 없어도 안정적으로 동작합니다.
    """
    net_path = os.path.abspath(net_file)
    if not os.path.exists(net_path):
        raise FileNotFoundError(f"SUMO 네트워크 파일 '{net_path}'를 찾을 수 없습니다.")

    net = sumolib.net.readNet(net_path)
    offsetX, offsetY = net.getLocationOffset()

    valid_tls_ids_in_sumo_net = []
    locations = {}
    for jid in tls_ids:
        try:
            node = net.getNode(jid)
            jloc = node.getCoord()
            carla_x = jloc[0] - offsetX
            carla_y = -(jloc[1] - offsetY)
            locations[jid] = (carla_x, carla_y)
            valid_tls_ids_in_sumo_net.append(jid)
        except KeyError:
            print(f"경고: SUMO 네트워크 파일에 정의되지 않은 TLS ID '{jid}'가 있습니다. 해당 노드는 인접 행렬에서 제외됩니다.")
            continue

    num_junctions = len(valid_tls_ids_in_sumo_net)
    adj_matrix = np.eye(num_junctions, dtype=int)

    for i in range(num_junctions):
        for j in range(i + 1, num_junctions):
            loc_i = locations[valid_tls_ids_in_sumo_net[i]]
            loc_j = locations[valid_tls_ids_in_sumo_net[j]]
            dist = np.sqrt((loc_i[0] - loc_j[0]) ** 2 + (loc_i[1] - loc_j[1]) ** 2)
            if dist <= threshold:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

    print(f"SUMO 좌표 기반 인접 행렬 생성 완료 ({num_junctions}x{num_junctions})")
    return adj_matrix


def main():
    """메인 검증 함수"""
    vec_env = None
    try:
        raw_env = CarlaValidationEnv()

        sumo_tls_ids_for_matrix = REAL_TLS_IDS_SUMO_LEARNING
        sumo_net_path = "../Learning_SUMO/sumo_files/Town05.with_tls.net.xml"

        adj_matrix = create_carla_adjacency_matrix(sumo_net_path, sumo_tls_ids_for_matrix, GNN_ADJACENCY_THRESHOLD)

        policy_kwargs = dict(
            features_extractor_class=GATFeatureExtractor,
            features_extractor_kwargs=dict(adjacency_matrix=adj_matrix, features_dim=128),
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )

        env = Monitor(raw_env)
        vec_env = DummyVecEnv([lambda: env])

        print(f"CARLA 환경의 관측 공간 형태: {env.observation_space.shape}")

        vec_env = VecNormalize.load(STATS_PATH, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

        model = PPO.load(MODEL_PATH, env=vec_env, custom_objects={'policy_kwargs': policy_kwargs})

        print("\n--- CARLA 환경에서 모델 검증 시작 ---")
        for i in range(3):  # 3개 에피소드 실행
            obs = vec_env.reset()
            done = False
            print(f"\n--- 에피소드 {i + 1} 시작 ---")
            step_count = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, infos = vec_env.step(action)
                info = infos[0]  # VecEnv에서 반환되는 infos 리스트의 첫 번째 요소 (단일 환경)
                step_count += 1
                if step_count % 100 == 0:  # 100 스텝마다 로그 출력
                    print(
                        f"에피소드 {i + 1} - 스텝 {step_count}: Current Phases = {info['current_logical_phases']}, Actions = {info['actions_applied']}")
            print(f"--- 에피소드 {i + 1} 종료 (총 {step_count} 스텝) ---")

    except Exception as e:
        print(f"\n오류 발생: {e}")
        traceback.print_exc()
    finally:
        if vec_env:
            vec_env.close()


if __name__ == "__main__":
    main()