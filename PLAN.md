# PLAN.md - Development Roadmap

## What We've Done

### Phase 1: Core Implementation ✅
- [x] Set up project structure with MediaPipe Tasks API
- [x] Downloaded ASL Alphabet dataset from Kaggle (87k images, 29 classes)
- [x] Implemented hand landmark extraction (21 points × 3 coords)
- [x] Built Random Forest classifier
- [x] Created CLI interface (`main.py`)
- [x] Achieved 91.76% baseline accuracy

### Phase 2: Feature Engineering ✅
- [x] Identified limitation: 2D landmarks don't capture thumb front/behind
- [x] Added 30 engineered features:
  - Thumb-to-fingertip distances (4)
  - Adjacent fingertip distances (4)
  - Z-depth comparisons (5)
  - Finger curl angles (5)
  - Finger spread angles (4)
  - Thumb-index spatial relationship (3)
  - Fingertip-to-wrist distances (5)
  - Cross-product orientation (1)
- [x] Improved accuracy to 97.53%

### Phase 3: Repository Setup ✅
- [x] Moved to dedicated repo: `github.com/bendsp/sign_detection`
- [x] Created README with setup instructions
- [x] Added .gitignore for data/models/venv
- [x] Initial commit and push

---

## What's Next

### Priority 1: Improve Weak Letters (M, N, Q)

**Problem**: M (82%), N (80%) still confused due to subtle thumb positioning.

**Options**:
1. **More training data** - Currently using 200 samples/class, dataset has 3000
   ```bash
   python main.py extract      # Use full dataset
   python main.py train
   ```

2. **Additional engineered features**
   - Thumb tip to palm plane distance
   - Finger overlap detection (which finger is on top)
   - Thumb angle relative to palm normal

3. **Different classifier**
   - Try SVM with RBF kernel
   - Small neural network (2-3 layers)
   - Gradient boosting (XGBoost)

### Priority 2: Motion-Based Letters (J, Z)

**Problem**: J and Z require hand movement, not just static poses.

**Solution**: Implement temporal features
- Track landmark positions over N frames
- Compute velocity/trajectory of key points
- Classify motion patterns

**Implementation sketch**:
```python
class MotionDetector:
    def __init__(self, buffer_size=15):
        self.landmark_buffer = deque(maxlen=buffer_size)
    
    def add_frame(self, landmarks):
        self.landmark_buffer.append(landmarks)
    
    def get_motion_features(self):
        # Compute trajectory of index fingertip
        # Return direction, curvature, speed
```

### Priority 3: Performance Optimization

- [ ] Profile real-time inference speed
- [ ] Consider model quantization
- [ ] Batch processing for video files
- [ ] GPU acceleration if available

### Priority 4: User Experience

- [ ] Add letter spelling mode (accumulate letters into words)
- [ ] Audio feedback (text-to-speech)
- [ ] Training mode (show correct hand position)
- [ ] Gesture recording for custom training data

### Priority 5: Alternative Approaches

**If accuracy plateau persists**:

1. **Deep learning on raw images**
   - Fine-tune MobileNet/EfficientNet
   - End-to-end learning might capture subtle features
   - Requires more compute, larger dataset

2. **Depth camera integration**
   - Intel RealSense or similar
   - True depth instead of estimated z-coordinates
   - Would dramatically help M/N/T disambiguation

3. **Multi-angle training**
   - Collect data from multiple camera angles
   - Data augmentation with synthetic rotations

---

## Quick Experiments to Try

### Experiment 1: Full Dataset Training
```bash
python main.py extract        # ~20 min, uses all 3000 samples/class
python main.py train
python main.py realtime       # Test improvement
```

### Experiment 2: Different Classifier
Edit `src/train.py`:
```python
from sklearn.svm import SVC
clf = SVC(kernel='rbf', probability=True)
```

### Experiment 3: Feature Importance Analysis
```python
import pickle
with open('models/asl_classifier.pkl', 'rb') as f:
    data = pickle.load(f)
clf = data['classifier']
importances = clf.feature_importances_
# Analyze which features matter most
```

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Overall accuracy | 97.5% | 99% |
| M accuracy | 82% | 95% |
| N accuracy | 80% | 95% |
| J/Z (with motion) | N/A | 90% |
| Real-time FPS | ~30 | 30+ |
| Inference latency | <50ms | <30ms |

---

## Notes

- Dataset: Kaggle ASL Alphabet (grassknoted/asl-alphabet)
- MediaPipe model: `hand_landmarker.task` (float16, auto-downloaded)
- Python: 3.10+ required
- Tested on: macOS (Apple Silicon)
