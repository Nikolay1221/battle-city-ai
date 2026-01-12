import gymnasium as gym
import pygame
import numpy as np
import sys
import os

# Ensure we can import battle_city_env
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from battle_city_env import BattleCityEnv

def main():
    print("Initializing Battle City...")
    
    # Initialize Pygame
    pygame.init()
    
    # Scale factor for visibility (NES is 256x240, 84x84 crop used in AI)
    # The raw NESEnv renders full resolution usually (256x240).
    # Let's check what env.render() returns. 
    # BattleCityEnv.render() calls self.env.render(), which usually opens a window in nes-py.
    # But we want to capture it or control it.
    
    # Actually, nes_py's render(mode='human') creates its own pyglet window.
    # Mixing Pygame input with Pyglet window might be messy.
    # BETTER APPROACH:
    # Use env with render_mode='rgb_array' -> get frame -> blit to Pygame window.
    
    env = BattleCityEnv(render_mode='rgb_array', use_vision=False)
    obs, info = env.reset()
    
    # Get initial frame to determine size
    frame = env.raw_env.screen.copy() # (240, 256, 3) usually for NES
    h, w, c = frame.shape
    
    SCALE = 3
    screen = pygame.display.set_mode((w * SCALE, h * SCALE))
    pygame.display.set_caption("Battle City - Human Mode (Arrows + Z)")
    
    clock = pygame.time.Clock()
    running = True
    
    print("\n CONTROLS:")
    print(" [Arrows] : Move")
    print(" [Z]      : Fire")
    print(" [Esc]    : Quit")
    
    action_map = {
        # Keys to Action Index
        # 0: NOOP
        # 1: Up
        # 2: Down
        # 3: Left
        # 4: Right
        # 5: A (Fire)
        # 6: Up+A
        # 7: Down+A
        # 8: Left+A
        # 9: Right+A
    }
    
    # Create Stats Window
    import cv2
    from collections import deque
    
    cv2.namedWindow("Reward Log", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("Reward Log", w * SCALE + 20, 50) # Position next to game
    
    # Init Font
    frame_font = pygame.font.SysFont('Arial', 12, bold=True)
    
    msg_log = deque(maxlen=20)
    total_score = 0.0
    
    curr_frame = 0
    while running:
        curr_frame += 1
        if curr_frame == 300:
             pygame.image.save(screen, "debug_auto.png")
             print("DEBUG: Auto-saved debug_auto.png")
             
        # 1. Handle Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_s:
                    # Save Screenshot
                    # surf is scaled, let's save the raw frame if possible or just the surf
                    # We can save surf
                    pygame.image.save(screen, "debug_game_screen.png")
                    print("Saved debug_game_screen.png")
        
        # 2. Get Input State
        keys = pygame.key.get_pressed()
        
        up    = keys[pygame.K_UP]
        down  = keys[pygame.K_DOWN]
        left  = keys[pygame.K_LEFT]
        right = keys[pygame.K_RIGHT]
        fire  = keys[pygame.K_z]
        
        # Determine Action
        action = 0 # NOOP
        
        if up:
            if fire: action = 6
            else:    action = 1
        elif down:
            if fire: action = 7
            else:    action = 2
        elif left:
            if fire: action = 8
            else:    action = 3
        elif right:
            if fire: action = 9
            else:    action = 4
        elif fire:
            action = 5
        
        # 3. Step Env
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- REWARD LOGGING ---
        if abs(reward) > 0.001:
            total_score += reward
            # Identify Event
            event_text = f"{reward:+.2f}"
            color = (255, 255, 255) # White
            
            if reward >= 1.0: 
                event_text = f"KILL! ({reward:+.1f})"
                color = (0, 255, 0) # Green
            elif reward == 0.5:
                event_text = f"BONUS ({reward:+.1f})"
                color = (0, 255, 255) # Yellow
            elif reward == 0.02:
                event_text = f"Explore ({reward:+.2f})"
                color = (200, 200, 200) # Grey
            elif reward <= -1.0:
                event_text = f"DIED ({reward:+.1f})"
                color = (0, 0, 255) # Red
            elif reward == -0.02:
                event_text = f"Idle ({reward:+.2f})"
                color = (0, 0, 100) # Dark Red
            
            msg_log.append((event_text, color))
            
        # Draw Stats Window
        stats_bg = np.zeros((400, 300, 3), dtype=np.uint8)
        
        # Header
        cv2.putText(stats_bg, f"Score: {total_score:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                   
        # Distance Info
        dist = info.get('closest_enemy_dist', 999)
        num_enemies = info.get('enemies_detected', 0)
        
        dist_color = (255, 255, 255)
        if dist < 50: dist_color = (0, 0, 255) # Red (Danger!)
        elif dist < 100: dist_color = (0, 255, 255) # Yellow
        
        cv2.putText(stats_bg, f"Enemies: {num_enemies}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(stats_bg, f"Dist: {dist:.1f}", (150, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, dist_color, 1)
        
        # Log
        y = 90
        for text, col in reversed(msg_log):
            # Convert RGB (Pygame/Logical) to BGR (OpenCV)
            bgr = (col[2], col[1], col[0])
            cv2.putText(stats_bg, text, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 1)
            y += 25
            
        cv2.imshow("Reward Log", stats_bg)
        cv2.waitKey(1)
        
        # 4. Render Game
        # Get raw screen from env (full res)
        frame = env.raw_env.screen # RGB numpy array
        
        # Convert to Pygame Surface
        surf = pygame.surfarray.make_surface(frame.swapaxes(0,1))
        
        # Scale
        surf = pygame.transform.scale(surf, (w * SCALE, h * SCALE))
        
        # Blit
        screen.blit(surf, (0, 0))
        
        # --- DEBUG OVERLAY ---
        # Draw Player (RAM - Blue Box)
        if 'player_cv' in info:
           px_c, py_c = info['player_cv']
           # Centroid to Top-Left
           px = int((px_c - 8) * SCALE)
           py = int((py_c - 8) * SCALE)
           
           pygame.draw.rect(screen, (0, 0, 255), (px, py, 16*SCALE, 16*SCALE), 2)
           
           lbl = frame_font.render("PLAYER", True, (0, 255, 255)) # Cyan text
           screen.blit(lbl, (px, py - 15))
            
        # Draw Enemies (Red Box + ID)
        if 'enemy_positions' in info:
            for enemy_data in info['enemy_positions']:
                # Unpack 5 elements
                ex, ey, tmpl_id, score, is_visible = enemy_data
                
                ex_s = int((ex - 8) * SCALE)
                ey_s = int((ey - 8) * SCALE)
                
                # Draw Box (Red)
                pygame.draw.rect(screen, (255, 0, 0), (ex_s, ey_s, 16*SCALE, 16*SCALE), 2)
                
                # Draw LoS Line (Green=Clear, Red=Blocked)
                if 'player_cv' in info:
                    line_color = (0, 255, 0) if is_visible else (255, 50, 50)
                    thickness = 2 if is_visible else 1
                    pygame.draw.line(screen, line_color, (px + 8*SCALE, py + 8*SCALE), (ex_s + 8*SCALE, ey_s + 8*SCALE), thickness)

                if tmpl_id != -1:
                    status = "YES" if is_visible else "NO"
                    lbl = frame_font.render(f"RAM {tmpl_id} | LoS: {status}", True, (255, 255, 0))
                    screen.blit(lbl, (ex_s, ey_s - 15))
                
        pygame.display.flip()
        
        # 5. Cap FPS
        clock.tick(60)
        
        if terminated or truncated:
            msg_log.append(("--- EPISODE END ---", (100, 100, 255)))
            env.reset()
            
    env.close()
    pygame.quit()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
