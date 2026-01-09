import gymnasium as gym
import os
import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from battle_city_env import BattleCityEnv
import config

def test():
    print("--- BATTLE CITY AI - TESTING MODE ---")
    
    # Setup Environment
    # We use DummyVecEnv for a single environment interaction
    env_kwargs = {'use_vision': config.USE_VISION, 'stack_size': config.STACK_SIZE}
    env = DummyVecEnv([lambda: BattleCityEnv(**env_kwargs)])

    # Load Model
    model_path = f"{config.MODEL_DIR}/battle_city_final.zip"
    if not os.path.exists(model_path):
        model_path = f"{config.MODEL_DIR}/battle_city_interrupted.zip"
        if not os.path.exists(model_path):
            print(f"Error: No model found in {config.MODEL_DIR}")
            return

    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path, env=env)

    obs = env.reset()
    
    cv2.namedWindow("Battle City AI Test", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("Battle City AI Test", 50, 50)

    print("Playing... Press 'q' or Ctrl+C to stop.")

    try:
        while True:
            # Predict action
            # deterministic=True makes the agent pick the best action (no exploration noise)
            action, _states = model.predict(obs, deterministic=True)
            
            obs, rewards, dones, infos = env.step(action)
            
            # Rendering
            try:
                frame = infos[0].get('render')
                if frame is not None:
                    # Resize for visibility (84x84 -> 672x672)
                    frame_img = frame.astype('uint8')
                    frame_big = cv2.resize(frame_img, (672, 672), interpolation=cv2.INTER_NEAREST)
                    
                    # Convert to BGR
                    display_frame = cv2.cvtColor(frame_big, cv2.COLOR_GRAY2BGR)
                    
                    # Add Info Text
                    kills = infos[0].get('kills', 0)
                    score = infos[0].get('score', 0.0)
                    
                    cv2.putText(display_frame, f"KILLS: {kills}", (20, 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv2.putText(display_frame, f"SCORE: {score:.1f}", (400, 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                    cv2.imshow("Battle City AI Test", display_frame)
                    
                    # Wait 20ms (approx 50 FPS)
                    if cv2.waitKey(20) & 0xFF == ord('q'):
                        break
            except Exception as e:
                print(f"Render Error: {e}")

            if dones[0]:
                obs = env.reset()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test()
