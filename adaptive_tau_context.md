# Context for agent mode: adaptive tau calculation for ROI merging benchmark

## Goal
Implement dynamic / online adaptation of the ROI merging threshold `tau` in the video object detection benchmark.

The benchmark currently uses cost-aware ROI merging where two regions are merged if the expected reduction in inference cost is positive. The merge threshold is based on the linear inference-time model:

```math
T(A) = K_t + c_t A
```

where:

- `A` is processed area in pixels;
- `K_t` is fixed per-inference overhead;
- `c_t` is per-pixel processing cost;
- `tau = K_t / c_t` is the maximum additional area that is still worth accepting to remove one detector invocation.

The new feature should make `tau` adaptive during video processing instead of using a fixed value from offline profiling.

## Motivation
On unstable CPU/mobile/edge devices, especially laptops on battery, measured `tau` can fluctuate heavily. Example observed by the user:

- `tau` varies from about `4000` px to `14000` px;
- most frequent values are closer to `6000` px.

This is too large to treat as measurement noise. Static `tau` can therefore be suboptimal.

Reasons for variation:

- dynamic CPU frequency scaling;
- thermal throttling;
- battery power limits;
- OS background load;
- changing memory/cache behavior.

## Scientific framing
Do not adapt `tau` directly from FPS. Instead, estimate the current cost model parameters online:

```math
T(A) = K_t + c_t A
```

Then compute:

```math
tau_t = K_t^{(t)} / c_t^{(t)}
```

Optionally smooth the threshold:

```math
tau_t = (1 - alpha) * tau_{t-1} + alpha * (K_t^{(t)} / c_t^{(t)})
```

Recommended `alpha`: start with `0.05` to `0.2`.

## Important observation about full-frame inference
The benchmark periodically runs full-frame detection, for example every 10 frames.

This helps online regression because full-frame points have much larger area than ROI points. Therefore the regression sees a broader range of `A` values:

- full-frame measurements: large `A`;
- ROI measurements: small/medium `A`.

Example for key-frame / full-frame interval = 10:

- per 100 frames: at least 10 full-frame measurements;
- up to 90 ROI-mode frames;
- if each ROI-mode frame has multiple ROI detector calls, the number of ROI measurements can be larger than 90.

This should improve statistical stability compared with estimating the model only from similarly sized ROIs.

## Implementation design
Add an online cost model estimator class, e.g.:

```python
class OnlineCostModel:
    def __init__(
        self,
        window_size: int = 100,
        min_samples: int = 30,
        min_fullframe_samples: int = 3,
        min_area_span_ratio: float = 3.0,
        alpha_tau: float = 0.1,
        tau_init: float | None = None,
        tau_min: float | None = None,
        tau_max: float | None = None,
        robust: bool = True,
    ):
        ...
```

It should store recent observations:

```python
(A_i, T_i, mode_i, frame_idx)
```

where:

- `A_i`: processed area in pixels;
- `T_i`: measured inference time in seconds;
- `mode_i`: `"full_frame"` or `"roi"`;
- `frame_idx`: video frame index.

Use a sliding window of the most recent `N` observations. Good starting values:

- `window_size = 100` or `300` observations;
- `min_samples = 30`;
- `min_fullframe_samples = 3` or `5`;
- update every frame or every `M` frames, but only if conditions are satisfied.

## Linear regression formula
For simple 2-parameter least squares:

```math
c_t = sum((A_i - mean(A)) * (T_i - mean(T))) / sum((A_i - mean(A))^2)
K_t = mean(T) - c_t * mean(A)
```

Then:

```math
tau_raw = K_t / c_t
```

Validation checks:

- `c_t > 0`;
- `K_t > 0`;
- denominator is not near zero;
- enough samples;
- enough full-frame samples;
- enough area diversity.

Area diversity check example:

```python
area_span_ratio = max(A) / max(min(A), 1)
area_span_ratio >= min_area_span_ratio
```

or use percentile span for robustness:

```python
p90(A) / max(p10(A), 1) >= min_area_span_ratio
```

## Robustness / outlier handling
Timing measurements can contain outliers from OS scheduling and background processes.

Recommended simple approach:

1. Fit linear model on current window.
2. Compute residuals.
3. Remove samples with residual larger than `k * MAD`, where `k = 3` or `4`.
4. Refit.

Simpler first implementation: use ordinary least squares plus tau smoothing and clamping.

## Tau smoothing and clamping
After computing raw tau:

```python
tau_raw = K_t / c_t
```

apply optional clamp:

```python
tau_raw = min(max(tau_raw, tau_min), tau_max)
```

Then smooth:

```python
if tau_current is None:
    tau_current = tau_raw
else:
    tau_current = (1 - alpha_tau) * tau_current + alpha_tau * tau_raw
```

Suggested initial clamp for Raspberry PI experiment:

- `tau_min = 1200`;
- `tau_max = 20000`;

or make these config values.

## Integration points in benchmark
Find the place where detector inference is timed. After every detector call, log the observation:

```python
cost_model.add_observation(
    area=processed_area_px,
    time_sec=inference_time_sec,
    mode="full_frame" or "roi",
    frame_idx=frame_idx,
)
```

Then before greedy merging on the next frame:

```python
tau = cost_model.get_tau(default_tau=config.merge_tau)
```

Use this `tau` in the cost-aware greedy ROI merge.

Important: do not run extra benchmark inferences just for tau adaptation. Use only telemetry already generated by normal full-frame and ROI inference.

## What counts as area
Use the actual detector input area, not just the original crop area, if resizing / padding affects compute.

For ROI-SSD variable input:

```python
A = input_height * input_width
```

For YOLO with shape rounding to stride multiple:

```python
A = effective_input_height * effective_input_width
```

If the model always resizes every ROI to a fixed size, area-based linear model may not be valid. In that case, area should reflect the actual tensor area passed to the network, not the semantic crop area.

## Logging fields to add
For each frame or summary row, log:

- `adaptive_tau_enabled`;
- `tau_current`;
- `tau_raw` if available;
- `K_t_current`;
- `c_t_current`;
- `cost_model_sample_count`;
- `cost_model_fullframe_count`;
- `cost_model_roi_count`;
- `cost_model_area_span_ratio`;
- `cost_model_valid`;
- `tau_update_count`;
- optionally `tau_min`, `tau_max`, `alpha_tau`, `window_size`.

This will make plots and debugging easier.

## Expected behavior
If `K_t` dominates and per-pixel cost is relatively low:

- `tau` should increase;
- merging becomes more aggressive;
- fewer detector calls;
- larger merged ROIs.

If per-pixel cost becomes high:

- `tau` should decrease;
- merging becomes more conservative;
- avoids excessive ROI expansion.

## Main risks
1. Regression instability if all areas are too similar.
   - Mitigation: periodic full-frame detection gives large-area anchor points.

2. Outlier timing samples.
   - Mitigation: sliding window, smoothing, clamping, optional robust regression.

3. Feedback loop.
   - Merging changes ROI areas and detector times, which affect the model estimate.
   - Mitigation: slow adaptation with small `alpha_tau`; update from a window, not one frame.

4. Wrong area definition.
   - Use actual network input tensor area.

5. Negative or nonsensical fitted parameters.
   - Reject update if `K_t <= 0` or `c_t <= 0`.

## Minimal pseudocode

```python
class OnlineCostModel:
    def __init__(self, window_size=100, min_samples=30, min_fullframe_samples=3,
                 min_area_span_ratio=3.0, alpha_tau=0.1,
                 tau_init=None, tau_min=None, tau_max=None):
        self.samples = deque(maxlen=window_size)
        self.alpha_tau = alpha_tau
        self.tau = tau_init
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.K_t = None
        self.c_t = None
        self.valid = False
        self.update_count = 0

    def add_observation(self, area, time_sec, mode, frame_idx):
        if area <= 0 or time_sec <= 0:
            return
        self.samples.append((float(area), float(time_sec), mode, frame_idx))
        self.try_update()

    def try_update(self):
        if len(self.samples) < self.min_samples:
            self.valid = False
            return

        full_count = sum(1 for _, _, mode, _ in self.samples if mode == "full_frame")
        if full_count < self.min_fullframe_samples:
            self.valid = False
            return

        A = np.array([s[0] for s in self.samples], dtype=np.float64)
        T = np.array([s[1] for s in self.samples], dtype=np.float64)

        p10, p90 = np.percentile(A, [10, 90])
        area_span = p90 / max(p10, 1.0)
        if area_span < self.min_area_span_ratio:
            self.valid = False
            return

        A_mean = A.mean()
        T_mean = T.mean()
        denom = ((A - A_mean) ** 2).sum()
        if denom <= 1e-9:
            self.valid = False
            return

        c_t = ((A - A_mean) * (T - T_mean)).sum() / denom
        K_t = T_mean - c_t * A_mean

        if c_t <= 0 or K_t <= 0:
            self.valid = False
            return

        tau_raw = K_t / c_t
        if self.tau_min is not None:
            tau_raw = max(tau_raw, self.tau_min)
        if self.tau_max is not None:
            tau_raw = min(tau_raw, self.tau_max)

        if self.tau is None:
            self.tau = tau_raw
        else:
            self.tau = (1.0 - self.alpha_tau) * self.tau + self.alpha_tau * tau_raw

        self.K_t = K_t
        self.c_t = c_t
        self.valid = True
        self.update_count += 1

    def get_tau(self, default_tau):
        return self.tau if self.tau is not None else default_tau
```

## Recommended config fields

```yaml
merging:
  type: greedy
  tau: 6000
  adaptive_tau: true
  adaptive_tau_window_size: 100
  adaptive_tau_min_samples: 30
  adaptive_tau_min_fullframe_samples: 3
  adaptive_tau_min_area_span_ratio: 3.0
  adaptive_tau_alpha: 0.1
  adaptive_tau_min: 3000
  adaptive_tau_max: 20000
```

## Suggested article wording
The adaptive threshold can be described as:

> Periodic full-frame inference performs a dual role: it restores global scene information and provides large-area timing observations for online estimation of the computational cost model. Since full-frame areas are substantially larger than typical ROI areas, these observations widen the range of processed areas and improve the statistical stability of estimating `K_t` and `c_t`. The resulting adaptive threshold `tau_t = K_t^{(t)} / c_t^{(t)}` allows the ROI merging strategy to adjust to temporal changes in the computational platform, such as battery power limits, thermal throttling, and background load.
