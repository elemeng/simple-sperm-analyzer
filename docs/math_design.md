
## 🔬 Mathematical Design

### Detection Algorithm

The detection pipeline uses morphological operations to identify sperm heads:

1. **Preprocessing**: Gaussian blur with radius `σ` to reduce noise
2. **Thresholding**: Binary threshold at level `T` to separate objects from background
3. **Morphological Filtering**: Remove objects based on size, shape, and solidity criteria

**Object Classification Criteria:**

* **Area**: `A_min ≤ Area ≤ A_max` (reject debris and clusters)
* **Aspect Ratio**: `AR_min ≤ Length/Width ≤ AR_max` (sperm heads are elongated)  
* **Solidity**: `Solidity ≥ S_min` (compactness measure)

### Tracking Algorithm - Motion-Aware Cascade Matching

Our tracking system implements a **direction-first** multi-factor matching approach:

#### 1. Motion Estimation

For each track `i` at frame `t`, estimate motion parameters:

```python
v_i^prev = p_i^(t) - p_i^(t-2)    # Direction vector (2-frame window)
s_i^prev = |v_i^prev| / 2        # Speed estimate
```

#### 2. Candidate Gating (Hard Constraints)

Detection `j` is a candidate for track `i` if:

**Distance Constraint:**

```python
|p_j - p_i^(t)| ≤ d_max
```

**Direction Constraint:**

```python
Δθ_ij = arccos((v_i^prev · v_ij) / (|v_i^prev| |v_ij|)) ≤ θ_hard
```

where `v_ij = p_j - p_i^(t)` and `θ_hard ≈ 120°`

#### 3. Weighted Assessment (Soft Constraints)

For each candidate, compute matching score:

```python
S_ij = 0.35 × W_dir + 0.30 × W_dist + 0.20 × W_speed + 0.15 × W_morph
```

**Weight Components:**

* **Direction Weight**: `W_dir = exp(-(Δθ_ij/σ_θ)²)` with `σ_θ = 45°`
* **Distance Weight**: `W_dist = exp(-(d_ij/σ_d)²)` with `σ_d = d_max/2`
* **Speed Weight**: `W_speed = exp(-((s_ij - s_i^prev)/σ_s)²)`
* **Morphology Weight**: Based on area/shape similarity

#### 4. Global Assignment

Solve the assignment problem using:

* **Greedy Assignment**: Fast, good for dense scenes (default)
* **Hungarian Algorithm**: Globally optimal, better for sparse scenes

#### 5. Post-Assignment Validation

Re-validate assignments against hard constraints to prevent numerical leakage.

### Analysis Algorithm - Standard CASA Parameters

Computes industry-standard motility parameters from tracked trajectories:

#### Velocity Parameters

* **VCL** (Curvilinear Velocity): Total path length / time
* **VSL** (Straight-line Velocity): Straight-line distance / time  
* **VAP** (Average Path Velocity): Smoothed path length / time
