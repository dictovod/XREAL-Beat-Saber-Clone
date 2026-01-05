# 🎮 XREAL Beat Saber Clone

A rhythm game for XREAL One Pro AR glasses with hand motion controls via IMU sensors.

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)

## 🎯 Features

- **Full-Screen 3D Gameplay** - Immersive OpenGL rendering
- **Hand Motion Controls** - Use gyroscope data to detect hand swings
- **Color-Coded Cubes**:
  - 🔴 **Red** = Left hand only
  - 🔵 **Blue** = Right hand only
  - 🟢 **Green** = Any hand
- **Combo System** - Score multiplier for consecutive hits
- **Progressive Difficulty** - Speed increases over time
- **Visual Effects** - Slash trails and hit animations
- **Real-Time Statistics** - Score, combo, accuracy tracking

## 🎥 Demo

![Gameplay Screenshot](https://via.placeholder.com/800x450/1a1a2e/ffffff?text=XREAL+Beat+Saber+Gameplay)

*Swing your hands to slash flying cubes in rhythm!*

## 📋 Requirements

### Hardware
- **XREAL One Pro** AR glasses (or compatible XREAL device)
- Windows PC with USB connection
- NCM (Network Control Model) driver installed

### Software
- Python 3.7 or higher
- OpenGL support
- Required Python packages (see Installation)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/dictovod/XREAL-Beat-Saber-Clone.git
cd XREAL-Beat-Saber-Clone
```

### 2. Install Dependencies

```bash
pip install pygame PyOpenGL PyOpenGL_accelerate numpy opencv-python
```

Or use requirements.txt:

```bash
pip install -r requirements.txt
```

### 3. Connect XREAL Glasses

1. Connect your XREAL One Pro glasses via USB
2. Ensure the NCM network interface is active
3. Verify the glasses IP is `169.254.2.1` (default)

### 4. Test IMU Connection (Optional)

```bash
python imu_reader.py
```

You should see live IMU data streaming. Press `Ctrl+C` to stop.

## 🎮 How to Play

### Launch the Game

```bash
python xreal_beat_saber.py
```

The game will automatically:
- Enter full-screen mode
- Connect to XREAL IMU sensor
- Start spawning cubes

### Controls

| Action | Control |
|--------|---------|
| **Hit Cubes** | Swing your hands (left/right) |
| **Pause** | `SPACE` |
| **Reset** | `R` |
| **Quit** | `ESC` |

### Gameplay Rules

1. **Cubes fly towards you** on three lanes (left, center, right)
2. **Match the color**:
   - Red cubes → Swing left hand
   - Blue cubes → Swing right hand
   - Green cubes → Swing any hand
3. **Hit in the zone** - Yellow line marks the hit zone
4. **Build combos** - Consecutive hits multiply your score
5. **Miss penalty** - Missing resets your combo

### Scoring

- **Base Hit**: 100 points
- **Combo Multiplier**: 100 × (combo + 1)
- **Example**: 5th hit in combo = 600 points

## 🛠️ Project Structure

```
XREAL-Beat-Saber-Clone/
├── xreal_beat_saber.py      # Main game file
├── imu_reader.py             # IMU sensor reader
├── config.py                 # Configuration constants
├── live_video_viewer.py      # Camera viewer (bonus)
├── main.py                   # GUI test application
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Network settings
GLASSES_IP_PRIMARY = "169.254.2.1"
PORT_IMU = 52998

# Game settings (in xreal_beat_saber.py)
spawn_interval = 1.2          # Seconds between cubes
hit_cooldown = 0.2            # Seconds between swings
```

## 🐛 Troubleshooting

### IMU Not Connecting

**Problem**: "IMU NOT CONNECTED" message

**Solutions**:
1. Check USB connection
2. Verify NCM driver is installed
3. Ping glasses: `ping 169.254.2.1`
4. Try secondary IP: Change `GLASSES_IP_PRIMARY` to `"169.254.1.1"` in config.py
5. Restart glasses and PC

### Low Frame Rate

**Problem**: Game stutters or lags

**Solutions**:
1. Close other GPU-intensive applications
2. Update graphics drivers
3. Lower resolution (edit full-screen mode settings)
4. Check OpenGL support: `glxinfo | grep OpenGL` (Linux) or use GPU-Z (Windows)

### Hand Detection Issues

**Problem**: Swings not detected

**Solutions**:
1. **Swing faster** - Detection threshold is ~4.0 rad/s
2. Check IMU data: Run `python imu_reader.py` and observe gyro values
3. Adjust sensitivity in `detect_hand_strikes()` method
4. Ensure glasses are worn properly for accurate motion tracking

### Full-Screen Problems

**Problem**: Can't exit full-screen

**Solutions**:
- Press `ESC` to quit
- Press `Alt+F4` (Windows)
- Press `Alt+Tab` to switch windows, then close

## 🔧 Advanced Customization

### Adjust Hand Detection Sensitivity

In `xreal_beat_saber.py`, modify the `detect_hand_strikes()` method:

```python
# More sensitive (easier)
if left_speed > 2.5:  # Default: 4.0
    left_strike = 'left'

# Less sensitive (harder)
if left_speed > 6.0:  # Default: 4.0
    left_strike = 'left'
```

### Change Game Speed

```python
# In update() method
cube.z += 15 * dt  # Default: 20 (slower = easier)
cube.z += 30 * dt  # Faster = harder
```

### Modify Spawn Rate

```python
self.spawn_interval = max(0.4, 1.2 - self.game_time * 0.02)  # Faster progression
```

## 📊 Technical Details

### IMU Data Processing

- **Gyroscope**: Angular velocity (rad/s) for motion detection
- **Accelerometer**: Linear acceleration (m/s²) for gravity reference
- **Sampling Rate**: ~100 Hz
- **Protocol**: TCP on port 52998

### Hand Detection Algorithm

1. Collect last 5 IMU samples (~50ms window)
2. Calculate average angular velocity
3. Detect left swing: positive Z-axis rotation (>4.0 rad/s)
4. Detect right swing: negative Z-axis rotation (>4.0 rad/s)
5. Apply cooldown (200ms) to prevent double-hits

### 3D Rendering

- **Engine**: OpenGL via PyOpenGL
- **Perspective**: 60° FOV
- **Camera**: Fixed position (0, 3, 8)
- **Lighting**: Single directional light with ambient
- **Effects**: Alpha blending for trails and hit animations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Ideas for Improvements

- [ ] Add music synchronization
- [ ] Multiple difficulty levels
- [ ] Custom beatmaps/songs
- [ ] Leaderboard/high scores
- [ ] Two-player mode
- [ ] VR mode with head tracking
- [ ] Power-ups and special cubes
- [ ] Haptic feedback (if supported)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on reverse engineering from [6dofXrealWebcam](https://github.com/alvr-org/ALVR/discussions/1974)
- Inspired by [Beat Saber](https://beatsaber.com/) by Beat Games
- XREAL IMU protocol documentation from [SamiMitwalli/One-Pro-IMU-Retriever-Demo](https://github.com/SamiMitwalli/One-Pro-IMU-Retriever-Demo)

## 📧 Contact

**Project Maintainer**: [dictovod](https://github.com/dictovod)

**Issues**: [GitHub Issues](https://github.com/dictovod/XREAL-Beat-Saber-Clone/issues)

---

## 🎯 Quick Start Summary

```bash
# 1. Install
git clone https://github.com/dictovod/XREAL-Beat-Saber-Clone.git
cd XREAL-Beat-Saber-Clone
pip install -r requirements.txt

# 2. Connect XREAL glasses via USB

# 3. Play!
python xreal_beat_saber.py

# Controls: Swing hands to hit cubes, ESC to quit
```

**Enjoy the game! 🎮🥽✨**
