import gymnasium as gym
from gymnasium import spaces
from nes_py import NESEnv
from nes_py.wrappers import JoypadSpace
import cv2
import numpy as np
from collections import deque
import glob
import os

class BattleCityEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None, use_vision=False, stack_size=4):
        super(BattleCityEnv, self).__init__()
        
        ROM_PATH = 'BattleCity_fixed.nes' 
        self.render_mode = render_mode
        self.USE_VISION = use_vision
        self.STACK_SIZE = stack_size
        self.MAX_STEPS = 100_000_000 # Unlimited (almost).
        self.steps_in_episode = 0
        
        # Raw Env
        self.raw_env = NESEnv(ROM_PATH)
        
        # Define Restricted Actions: Move, Fire, Move+Fire
        # NO START/SELECT!
        actions = [
            ['NOOP'],
            ['up'],
            ['down'],
            ['left'],
            ['right'],
            ['A'], # Fire
            ['up', 'A'],
            ['down', 'A'],
            ['left', 'A'],
            ['right', 'A']
        ]
        
        # Wrap in JoypadSpace
        self.env = JoypadSpace(self.raw_env, actions)
        
        # FIX: JoypadSpace uses old 'gym' spaces. We need 'gymnasium' spaces.
        # We manually redefine the action space to match.
        self.action_space = spaces.Discrete(len(actions))

        # Dynamic Observation Space
        # SMART FEATURES (Coordinates + Line of Sight)
        # 40 Features per frame:
        # [Px, Py,  (Ex1, Ey1, Rx1, Ry1, LoS1), ... (E4...), Metadata...]
        self.FEATURES_DIM = 40 
        ram_size = self.FEATURES_DIM * self.STACK_SIZE
        
        if self.USE_VISION:
            # DICT: Screen + Features
            self.observation_space = spaces.Dict({
                "screen": spaces.Box(low=0, high=255, shape=(84, 84, self.STACK_SIZE), dtype=np.uint8),
                "ram": spaces.Box(low=0.0, high=1.0, shape=(ram_size,), dtype=np.float32)
            })
        else:
            # BOX: Features Only
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(ram_size,), dtype=np.float32)
        
        # RAM Stack Buffer
        self.ram_stack = deque(maxlen=self.STACK_SIZE)

        # Load Templates (With Colab Fail-Safe)
        self.game_over_tmpl = None
        try:
            self.game_over_tmpl = cv2.imread('templates/game_over.png', cv2.IMREAD_GRAYSCALE)
            self.base_destroyed_tmpl = cv2.imread('templates/base_destroyed.png', cv2.IMREAD_GRAYSCALE)
            
            if self.base_destroyed_tmpl is None:
                print("Warning: templates/base_destroyed.png not found!")

            if self.game_over_tmpl is None:
                print("Warning: templates/game_over.png not found. Switched to RAM Game Over (Experimental).")
                
            # Load Enemy Templates (All 16x16)
            self.enemy_templates = []
            tmpl_files = glob.glob('templates/enemies/*.png')
            for f in tmpl_files:
                t = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                if t is not None:
                    self.enemy_templates.append(t)
            print(f"Loaded {len(self.enemy_templates)} enemy templates (16x16).")
            
        except Exception as e:
            print(f"Error loading templates: {e}")
            self.game_over_tmpl = None

        # RAM Addresses
        self.ADDR_LIVES = 0x51
        self.ADDR_STATE = 0x92
        self.ADDR_KILLS = [0x73, 0x74, 0x75, 0x76] 
        self.ADDR_BONUS = 0x62
        self.ADDR_STAGE = 0x85
        
        self.prev_lives = 3
        self.prev_kills = [0, 0, 0, 0]
        self.prev_bonus = 0
        self.prev_stage = 0
        self.prev_x = 0
        self.prev_y = 0
        
        # Idle Penalty Vars
        self.ADDR_X = 0x0090
        self.ADDR_Y = 0x0098
        self.idle_steps = 0
        self.IDLE_THRESHOLD = 30 # 30 steps * 4 frames = 120 frames (~2 sec)
        
        # Step Logic
        self.steps_in_episode = 0
        
        # Frame Stack Buffer
        self.frames = deque(maxlen=self.STACK_SIZE)

    def _process_frame(self, obs):
        """Resize to 84x84 and Grayscale"""
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized # (84, 84)

    def _get_obs(self):
        # 1. Vision Stack (Updated via step/reset)
        
        # 2. Smart Features Stack (Coordinates + LoS)
        ram = self.raw_env.ram
        screen = self.raw_env.screen # (240, 256, 3)
        
        # Build Feature Vector (40 floats)
        features = np.zeros(self.FEATURES_DIM, dtype=np.float32)
        
        # Player
        px_ram = ram[0x90]
        py_ram = ram[0x98]
        features[0] = px_ram / 255.0 # X
        features[1] = py_ram / 255.0 # Y
        
        # Enemies (Slots 1-4)
        # Each enemy gets 5 features: [Ex, Ey, RelX, RelY, LoS]
        base_idx = 2
        
        for i in range(1, 5):
            # CAST TO INT to prevent uint8 overflow (Crashing bug fix)
            ex_ram = int(ram[0x90+i])
            ey_ram = int(ram[0x98+i])
            px_int = int(px_ram)
            py_int = int(py_ram)
            
            # 1. Absolute Coords
            features[base_idx]     = ex_ram / 255.0
            features[base_idx + 1] = ey_ram / 255.0
            
            # 2. Relative Coords
            features[base_idx + 2] = (ex_ram - px_int) / 255.0
            features[base_idx + 3] = (ey_ram - py_int) / 255.0
            
            # 3. Line of Sight (Raycast)
            los = 0.0
            
            # GAME MECHANIC CHECK: Axis Alignment
            # Tanks cannot shoot diagonally. Only check LoS if aligned.
            dx = abs(ex_ram - px_int)
            dy = abs(ey_ram - py_int)
            is_aligned = (dx < 12) or (dy < 12) # ~12px width leniency
            
            if (ex_ram != 0 or ey_ram != 0) and is_aligned:
                los = 1.0
                # DENSER SAMPLING (10 points) to catch thin walls
                for t in np.linspace(0.1, 0.9, 10):
                    sx = int(px_int + (ex_ram - px_int) * t)
                    sy = int(py_int + (ey_ram - py_int) * t)
                    
                    # Bounds check
                    if 0 <= sx < 256 and 0 <= sy < 240:
                        pixel = screen[sy, sx] # RGB
                        # Brick (Red/Orange), Concrete (White/Grey) -> High Red.
                        # Trees (Green), Water (Blue) -> Low Red.
                        if pixel[0] > 60:  # Lowered threshold slightly to catch dark bricks
                             los = 0.0 # Blocked
                             break
            
            features[base_idx + 4] = los
            base_idx += 5
            
        # Metadata (starts at index 2 + 4*5 = 22)
        features[22] = ram[self.ADDR_LIVES] / 10.0 # Lives
        current_kills = sum([ram[k] for k in self.ADDR_KILLS])
        features[23] = current_kills / 20.0 # Kills
        features[24] = ram[self.ADDR_STAGE] / 35.0 # Stage
        
        # Add to stack
        self.ram_stack.append(features)
        
        # Fill if empty (first frame)
        while len(self.ram_stack) < self.STACK_SIZE:
             self.ram_stack.append(features)
             
        # Flatten Stack: [Feat1, Feat2, Feat3, Feat4] -> Vector
        ram_obs = np.concatenate(self.ram_stack) 
        
        if not self.USE_VISION:
            return ram_obs # Return ONLY vector (Box Space)

        # 1. Screen (Only if USE_VISION)
        if len(self.frames) < self.STACK_SIZE:
            screen_obs = np.zeros((self.STACK_SIZE, 84, 84), dtype=np.uint8)
        else:
            screen_obs = np.array(self.frames, dtype=np.uint8)
            
        # Transpose to (H, W, C) for Stable Baselines CnnPolicy
        screen_obs = np.moveaxis(screen_obs, 0, -1)
            
        return {
            "screen": screen_obs,
            "ram": ram_obs
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # self.raw_env is accessible.
        self.raw_env.reset()
        
        self.steps_in_episode = 0
        self.episode_score = 0.0 # Reset score
        self.visited_sectors = set() # Track visited 16x16 zones
        self.steps_in_episode = 0 # Reset Time Counter
        
        # Clear Buffers
        self.frames.clear()
        self.ram_stack.clear()
        
        # Auto-Skip Menu using RAW ENV actions (byte)
        # Start button is 0x08 (bit 3) -> 8
        
        # 2. Hardcoded Start Sequence (3 Presses for Level Select)
        # Sequence: Title -> [Start] -> Mode -> [Start] -> Level Select? -> [Start] -> Game
        
        # 1. Wait for Title (Robust Buffer)
        for _ in range(80): self.raw_env.step(0)
            
        # 2. Press Start (Title -> Mode)
        for _ in range(10): self.raw_env.step(8) # Hold longer
        for _ in range(30): self.raw_env.step(0) # Wait for fade
        
        # 3. Press Start (Mode -> Stage)
        for _ in range(10): self.raw_env.step(8)
        for _ in range(30): self.raw_env.step(0)

        # 4. Press Start (Stage -> Game)
        for _ in range(10): self.raw_env.step(8)
        
        # 5. Wait for Curtain (Game Start)
        for _ in range(60): self.raw_env.step(0)

        # Check debug
        state = int(self.raw_env.ram[self.ADDR_STATE])
        lives = int(self.raw_env.ram[self.ADDR_LIVES])
            
        self.prev_lives = int(self.raw_env.ram[self.ADDR_LIVES])
        self.prev_kills = [int(self.raw_env.ram[addr]) for addr in self.ADDR_KILLS]
        self.prev_bonus = int(self.raw_env.ram[self.ADDR_BONUS])
        self.prev_stage = int(self.raw_env.ram[self.ADDR_STAGE])
        self.prev_x = int(self.raw_env.ram[self.ADDR_X])
        self.prev_y = int(self.raw_env.ram[self.ADDR_Y])
        self.idle_steps = 0
        
        # Initial Frame Processing
        obs = self.raw_env.screen # Define obs
        processed = self._process_frame(obs)
        for _ in range(self.STACK_SIZE):
            self.frames.append(processed)
            
        # Fill RAM Stack with FRESH Smart Features (not old raw RAM)
        # Just call _get_obs() once - it auto-fills the stack with current state
        # (The while-loop inside _get_obs handles empty stack)
            
        return self._get_obs(), {} # Return (Obs, Info) for Gymnasium
        
    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        # Frame Skip x4
        for _ in range(4):
            # Use Wrapped Env step (maps action index to button press)
            obs, r, d, i = self.env.step(action)
            total_reward += r 
            if d:
                done = True
                break
        
        self.steps_in_episode += 1 
        
        # Process and Push new frame (Optimized)
        # Process and Push new frame (Optimized)
        processed = self._process_frame(obs)
        if self.USE_VISION:
             self.frames.append(processed)
        
        info['render'] = processed 
        
        ram = self.raw_env.ram # Access RAM from raw env
        reward = 0 
        
        # 0. Time Penalty
        if self.steps_in_episode >= self.MAX_STEPS:
            truncated = True # Time Limit = Truncated
            info["TimeLimit.truncated"] = True

        # 1. Kill Rewards
        curr_kills = [int(ram[addr]) for addr in self.ADDR_KILLS]
        for i in range(4):
            diff = curr_kills[i] - self.prev_kills[i]
            if diff > 0 and diff < 10:
                base_scores = [1.0, 1.5, 2.0, 3.0] 
                pts = base_scores[i]
                reward += pts * diff 
        self.prev_kills = curr_kills
        
        # 2. Bonus
        curr_bonus = int(ram[self.ADDR_BONUS])
        if curr_bonus > self.prev_bonus:
             reward += 0.5 # Normalized (+5.0 -> +0.5)
             
        # Trigger on Stage Reset (0->1 etc) - usually level num increases
        curr_stage = int(ram[self.ADDR_STAGE])
        if curr_stage > self.prev_stage:
             reward += 2.0 # Normalized (+20.0 -> +2.0)
        self.prev_stage = curr_stage
        
        # 3. Death
        curr_lives = int(ram[self.ADDR_LIVES])
        if curr_lives < 10 and self.prev_lives < 10:
             if curr_lives < self.prev_lives:
                reward -= 1.0 # Normalized (-0.5 -> -0.1 -> -1.0) Symmetric to Kill
        self.prev_lives = curr_lives
        
        # 4. Game Over Logic (Specific)
        # A. Base Destroyed (Vision)
        if self.base_destroyed_tmpl is not None:
             gray_full = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
             res = cv2.matchTemplate(gray_full, self.base_destroyed_tmpl, cv2.TM_CCOEFF_NORMED)
             _, max_val, _, _ = cv2.minMaxLoc(res)
             
             if max_val > 0.8: 
                  terminated = True
                  reward -= 10.0 # HUGE PENALTY for losing base
                  info['game_over_reason'] = 'base_destroyed'
                  
        # B. Out of Lives (RAM)
        if curr_lives == 0:
             terminated = True
             reward -= 5.0 # Extra penalty for losing all lives
             info['game_over_reason'] = 'out_of_lives'
            
        # 5. Idle Penalty (Coordinate Based)
        curr_x = int(ram[self.ADDR_X])
        curr_y = int(ram[self.ADDR_Y])
        
        # --- DISTANCE CALCULATION ---
        player_cv, enemies_data = self._detect_enemies(obs)
        if player_cv: info['player_cv'] = player_cv
        
        info['enemies_detected'] = len(enemies_data)
        if enemies_data:
            # enemies_data: list of (x, y, tmpl_id, score, is_visible)
            dists = [np.sqrt((curr_x - ex)**2 + (curr_y - ey)**2) for (ex, ey, *_) in enemies_data]
            closest_dist = min(dists)
            info['closest_enemy_dist'] = closest_dist
            
            # 8. HUNT REWARD (Distance Shaping)
            # If closer than before, +Reward
            if closest_dist < self.prev_dist:
                reward += 0.005 # Getting closer
            
            self.prev_dist = closest_dist
        else:
            info['closest_enemy_dist'] = 999.0
            self.prev_dist = 999.0
            
        # 7. Add Enemy Positions to Info (For render script)
        info['enemy_positions'] = enemies_data
            
        # 6. MOVEMENT REWARD (Normalized)
        if curr_lives > 0:
            if curr_x != self.prev_x or curr_y != self.prev_y:
                 self.idle_steps = 0
                 
                 # 7. GRID EXPLORATION (New!)
                 sec_x = curr_x // 16
                 sec_y = curr_y // 16
                 sector = (sec_x, sec_y)
                 
                 if sector not in self.visited_sectors:
                     reward += 0.1 # Discovery Bonus! (Increased)
                     self.visited_sectors.add(sector)
            else:
                 self.idle_steps += 1
                 
            # Threshold: 10 steps
            if self.idle_steps > 10:
               reward -= 0.002 # Tiny penalty (restored)

        else:
            self.idle_steps = 0 # Reset if dead
            
        self.prev_x = curr_x
        self.prev_y = curr_y
        
        # Debug info
        curr_state = int(ram[self.ADDR_STATE])
        info['idle_steps'] = self.idle_steps
        info['ram_state'] = curr_state
        info['x'] = curr_x
        info['y'] = curr_y
        info['kills'] = sum(curr_kills)
        
        self.episode_score += reward
        info['score'] = self.episode_score
        
        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode='human'):
        # Robust Render: Try with mode, then without
        try:
            self.env.render(mode=mode)
        except TypeError:
            self.env.render() # Try no-args (New Gym/NesPy updates)

    def close(self):
        self.env.close()

    # --- ENEMY DETECTION HELPER ---
    def _detect_enemies(self, obs):
        """
        Detects enemies using RAM (100% Accurate).
        RAM Map:
        X Coords: 0x90 (Player), 0x91-0x94 (Enemies)
        Y Coords: 0x98 (Player), 0x99-0x9C (Enemies)
        """
        ram = self.raw_env.ram
        
        # 1. Player (Slot 0)
        px = int(ram[0x90])
        py = int(ram[0x98])
        # Return Center (RAM + 8) to match Enemy format
        player_pos = (px + 8, py + 8)
        
        # 2. Enemies (Slots 1-4)
        enemies = []
        screen = self.raw_env.screen # Access screen for Raycast
        
        for i in range(1, 6): # Check up to 5 slots
            # CAST TO INT (Critical Fix)
            ex = int(ram[0x90 + i])
            ey = int(ram[0x98 + i])
            px_int = int(px)
            py_int = int(py)
            
            if ex == 0 and ey == 0:
                continue
            
            # --- LoS Check (Duplicate of _get_obs logic for Visual Debug) ---
            is_visible = False # Default False
            
            # GAME MECHANIC: Axis Alignment
            dx = abs(ex - px_int)
            dy = abs(ey - py_int)
            is_aligned = (dx < 12) or (dy < 12)
            
            if is_aligned:
                is_visible = True
                # DENSER SAMPLING
                for t in np.linspace(0.1, 0.9, 10):
                    sx = int(px_int + (ex - px_int) * t)
                    sy = int(py_int + (ey - py_int) * t)
                    if 0 <= sx < 256 and 0 <= sy < 240:
                        pixel = screen[sy, sx] 
                        if pixel[0] > 60: # Wall
                            is_visible = False
                            break
            
            # Format: (x_center, y_center, slot_id, confidence, is_visible)
            enemies.append((ex + 8, ey + 8, i, 1.0, is_visible))
            
        return player_pos, enemies
