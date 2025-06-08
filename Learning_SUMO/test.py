import os
import numpy as np
from sumo_ppo_traffic_v1 import SumoMultiPhaseEnv, REAL_TLS_IDS, MAX_STEPS_PER_EPISODE

def main():
    # GUI 없이 포트 8813으로 단일 환경 생성
    env = SumoMultiPhaseEnv(tls_ids=REAL_TLS_IDS, use_gui=True, port=8813)
    obs, _ = env.reset()
    print("reset 완료, obs shape:", obs.shape)

    total_reward = 0.0
    for step in range(100):  # 최대 100 스텝만 테스트
        action = env.action_space.sample()  # 랜덤 행동
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        print(f"[{step:03d}] reward={reward:.3f}, done={done}")
        if done:
            print(f"==> 에피소드 종료 (step={step+1}), 누적 reward={total_reward:.3f}")
            break

    env.close()
    print("환경 종료")

if __name__ == "__main__":
    main()