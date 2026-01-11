
import pickle
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
import config
import os

def view_graph():
    history_path = f"{config.MODEL_DIR}/score_history.pkl"
    
    if not os.path.exists(history_path):
        print("No history file found yet.")
        return

    try:
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
    except Exception as e:
        print(f"Error reading history: {e}")
        return

    print(f"Loaded {len(history)} games.")

    # Data Parsing
    scores = []
    scores = []
    steps = []
    
    # Robust mixed-format handling
    for i, item in enumerate(history):
        try:
            if isinstance(item, tuple) or isinstance(item, list):
                # New format: (score, step)
                if len(item) >= 2:
                    scores.append(float(item[0]))
                    steps.append(int(item[1]))
                elif len(item) == 1:
                    scores.append(float(item[0]))
                    steps.append(i) # Fallback step
            else:
                # Old format: score scalar
                scores.append(float(item))
                steps.append(i) # Fallback step
        except Exception:
            # Skip corrupt data
            continue
            
    # Ensure lengths match (redundant but safe)
    min_len = min(len(scores), len(steps))
    scores = scores[:min_len]
    steps = steps[:min_len]
    
    if not scores:
        print("History is empty.")
        return

    # Moving Average
    window_size = 50
    moving_avg = []
    if len(scores) >= window_size:
        moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        ma_steps = steps[window_size-1:]
    else:
        moving_avg = scores
        ma_steps = steps

    # Setup Plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplots_adjust(bottom=0.25) # Make room for slider
    
    ax.set_title("Battle City AI - History (Use Slider to Scroll)")
    ax.set_xlabel("Game / Steps")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='white', linestyle='--', alpha=0.3)

    # Polt Lines
    l_score, = ax.plot(steps, scores, color='cyan', alpha=0.4, linewidth=1, label="Score")
    l_avg, = ax.plot(ma_steps, moving_avg, color='yellow', linewidth=2, label=f"Avg ({window_size})")
    
    ax.legend(loc='upper left')

    # Initial Zoom (Last 500 games or all if less)
    view_width = 500
    total_games = len(steps)
    start_idx = max(0, total_games - view_width)
    end_idx = total_games - 1
    
    # Set initial X-limits to the logic values (step numbers), not indices
    if total_games > 0:
        ax.set_xlim(steps[start_idx], steps[end_idx])
        # Auto-scale Y based on visible data
        visible_scores = scores[start_idx:]
        if visible_scores:
            ax.set_ylim(min(visible_scores)-1, max(visible_scores)+1)

    # Slider Setup
    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor='lightgoldenrodyellow')
    
    # Slider represents the END INDEX of the view
    slider = widgets.Slider(
        ax_slider, 'Scroll', 
        valmin=min(view_width, total_games), 
        valmax=total_games, 
        valinit=total_games, 
        valstep=10
    )

    def update(val):
        end_view_idx = int(slider.val)
        start_view_idx = max(0, end_view_idx - view_width)
        
        # Get start/end Step Values
        x_min = steps[start_view_idx]
        x_max = steps[end_view_idx - 1] if end_view_idx > 0 else steps[0]
        
        ax.set_xlim(x_min, x_max)
        
        # Recalculate Y based on visible view
        visible_subset = scores[start_view_idx:end_view_idx]
        if visible_subset:
             ymin = min(visible_subset)
             ymax = max(visible_subset)
             margin = (ymax - ymin) * 0.1 if ymax != ymin else 1.0
             ax.set_ylim(ymin - margin, ymax + margin)
             
        fig.canvas.draw_idle()

    slider.on_changed(update)
    
    print("Graph opened. Use the slider at the bottom to scroll through history.")
    plt.show()

if __name__ == "__main__":
    view_graph()
