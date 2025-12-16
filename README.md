# Atari DQN Agent - Breakout

This project demonstrates a Deep Q-Network (DQN) agent trained to play the Atari game Breakout using **Stable-Baselines3** and **PyTorch with GPU acceleration**.

![Breakout Demo](media/breakout.gif)

## Project Structure

```
atari-dqn/
│
├── Models/                 # Trained models
│   └── dqn_breakout_gpu.zip
├── breakout_train.py       # Training script
├── watch_agent.py          # Watch the trained agent play
├── media/                  # GIF
│   └── breakout.gif
├── requirements.txt        # Dependencies
└── README.md               # Project overview and instructions
```

## Training

To train the agent:

```bash
pip install -r requirements.txt
python breakout_train.py
```

* The agent is trained with pixel-based input and frame stacking.
* GPU is **recommended** for faster training.

## Watching the Trained Agent

To watch the trained agent play:

```bash
python watch_agent.py
```

Controls:

* `q` → Quit
* `p` → Pause/Resume
* `f` → Faster playback
* `s` → Slower playback

The display shows **episode number** and **current reward**. The arrow key overlay is removed for a clean view.

## Dependencies

* Python 3.9+
* gymnasium[atari], ale-py
* stable-baselines3
* PyTorch (GPU recommended)
* OpenCV, ImageIO

## Notes

* Training uses `render_mode="rgb_array"` for speed.
* Watching uses OpenCV (`cv2.imshow`) for resized, clean display.
* Models are saved in `Models/`.
