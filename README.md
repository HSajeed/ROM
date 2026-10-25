# ROM: POD-LSTM Reduced-Order Modeling of a Cylinder Wake

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HSajeed/ROM/blob/main/ROM_pod_spod.ipynb)

Data-driven reduced-order model for the two-dimensional incompressible cylinder wake at $Re=100$. The notebook compresses the streamwise velocity field with proper orthogonal decomposition (POD), trains an LSTM to forecast the POD coefficients, and reconstructs the full field from sparse point measurements.

**Scope note:** Despite the `ROM_pod_spod` filename, the current notebook implements POD, not SPOD. It is a research and teaching prototype, not a CFD solver, a physics-informed neural network, or a production digital twin.

## Abstract

This project studies three connected operations on a canonical wake dataset:

1. **Compression:** POD/SVD extracts energy-ranked spatial modes from 200 snapshots of the streamwise velocity field.
2. **Forecasting:** an LSTM learns the temporal evolution of the first 10 POD coefficients and predicts the held-out snapshots autoregressively.
3. **Sparse sensing:** the same POD basis reconstructs the full field from a small number of point measurements, comparing random placement with QR-pivoted placement.

The notebook is intentionally compact. It is meant to show the full workflow, the shape conventions, and the main failure modes clearly, rather than to report state-of-the-art forecasting accuracy.

## Results at a Glance

| Component | Verified result |
|---|---|
| POD | 10 modes capture **99.9996%** of the streamwise-velocity energy |
| LSTM forecast | **12.84%** mean and **29.32%** maximum relative field error over 40 held-out snapshots |
| Sparse sensing | **0.227%** relative error with 10 QR-pivoted sensors on frame 150 |
| Underdetermined baseline | **40.31%** error with 5 random sensors on frame 159 |

The forecast horizon is 4.0 s (40 snapshots at $\Delta t=0.1$ s). The sparse-sensing result is close to the POD truncation floor: the global relative L2 error of the rank-10 reconstruction is 0.207%.

## Data

The notebook downloads the benchmark cylinder-wake dataset from the PINNs repository:

- `U_star`: `(5000, 2, 200)` — 5000 spatial points, two velocity components, 200 snapshots
- `X_star`: `(5000, 2)` — spatial coordinates
- `t`: `(200, 1)` — time stamps

Only the streamwise component `u = U_star[:, 0, :]` is modeled. The spatial domain is $x \in [1, 8]$, $y \in [-2, 2]$, and the time interval is $t \in [0, 19.9]$ with $\Delta t = 0.1$.

Dataset source: [maziarraissi/PINNs](https://github.com/maziarraissi/PINNs/blob/master/main/Data/cylinder_nektar_wake.mat)

## Method

### 1. POD compression

The snapshot matrix is $X \in \mathbb{R}^{5000 \times 200}$, with space along rows and time along columns. An economy SVD gives

$$
X \approx U_r \Sigma_r V_r^\top .
$$

The first $r=10$ spatial modes are retained. Let $A = V_r \Sigma_r$; its $t$-th row is the coefficient vector $a_t^\top$. The field snapshot is reconstructed as

$$
x_t \approx U_r a_t .
$$

Keeping the $\Sigma_r$ factor in the coefficients means the LSTM sees the relative importance of each mode directly.

### 2. LSTM forecasting

The LSTM receives a lookback window of 10 coefficient vectors and predicts the next one. The rollout is autoregressive: each prediction is appended to the input window for the next step.

| Setting | Value |
|---|---|
| Input per step | 10 POD coefficients |
| Lookback | 10 |
| LSTM units | 64 |
| Output | 10 coefficients |
| Loss | Mean squared error |
| Optimizer | Adam, learning rate 0.01 |
| Epochs | 100 |
| Batch size | 16 |
| Training sequences | 150 |
| Held-out sequences | 40 |

The first 150 sequences are used for training and the remaining 40 for validation. The field is reconstructed from the predicted coefficients as $x_t \approx U_r a_t$.

### 3. Sparse-sensor reconstruction

For a snapshot, let $y \in \mathbb{R}^p$ be measurements at $p$ spatial locations and let $C = U_r[\mathcal{S}, :]$ contain the POD modes at those locations. The coefficients are estimated by least squares:

$$
\hat{a} = C^{+} y,
\qquad
\hat{x} = U_r \hat{a}.
$$

If $p < r$, the system is underdetermined and `pinv` returns the minimum-norm solution, which is not guaranteed to recover the true coefficients. If $p \ge r$ and $C$ has full column rank, the system is square or overdetermined.

Sensor locations are selected with QR factorization with column pivoting on $U_r^\top$. This is a greedy selection heuristic, not a global optimum. The notebook also compares against random sensor placement.

## Results

### POD energy capture

| Modes $r$ | Cumulative energy |
|---:|---:|
| 1 | 97.2591% |
| 2 | 98.5837% |
| 5 | 99.9702% |
| 10 | 99.9996% |
| 20 | 100.0000% |

The first two modes already capture 98.58% of the energy. Ten modes are retained as a compromise between compactness and resolving the smaller structures used by the forecasting and sparse-sensing experiments.

### LSTM forecast

The autoregressive forecast over the 40 held-out snapshots gives:

| Metric | Value |
|---|---:|
| Mean relative field error | 0.1284 |
| Maximum relative field error | 0.2932 |

The error is computed per snapshot as

$$
\frac{\lVert x_{\text{true}} - x_{\text{pred}} \rVert_2}{\lVert x_{\text{true}} \rVert_2}.
$$

The forecast degrades over the 4.0 s horizon. This is a baseline result; no hyperparameter search, sequence-to-sequence model, or uncertainty quantification is included.

### Sparse reconstruction with QR-pivoted sensors

Frame 150, $r=10$ modes:

| Sensors $p$ | Relative error | Condition number $\kappa(C)$ |
|---:|---:|---:|
| 5 | 0.9875 | 2.08 |
| 9 | 0.9199 | 3.52 |
| 10 | 0.002273 | 5.78 |
| 12 | 0.002075 | 4.06 |
| 20 | 0.002036 | 3.62 |

For reference, 5 random sensors on frame 159 give a relative error of 0.4031. The two cases use different frames, so the comparison is illustrative rather than a controlled ranking.

The sharp transition at $p=10$ is a rank effect. With 10 retained modes, 9 sensors cannot uniquely determine the coefficients. For $p \ge 10$, the error is close to the POD truncation floor (0.207% global relative L2 error for $r=10$), so adding more sensors gives diminishing returns.

QR pivoting is designed for the square or overdetermined regime. It is not guaranteed to be optimal, and it can perform poorly when $p < r$; the table above should not be read as a general ranking of sensor-placement methods.

## Quick Start

### Option 1: Google Colab

Use the Colab badge at the top of this page. The notebook downloads the dataset automatically.

### Option 2: Local execution

```bash
git clone https://github.com/HSajeed/ROM.git
cd ROM

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install numpy scipy matplotlib tensorflow-cpu jupyter
jupyter notebook ROM_pod_spod.ipynb
```

Run all cells. The first cell downloads `cylinder_nektar_wake.mat` (about 23 MB) if it is not already present. Training 100 epochs on CPU takes a few minutes; a GPU is not required.

Tested with Python 3.11.15, TensorFlow 2.21.0 (CPU), NumPy 2.4.6, SciPy 1.17.1, and Matplotlib 3.11.2.

## Project Structure

```
ROM/
├── README.md
├── ROM_pod_spod.ipynb
└── rom_pod_spod.py
```

`rom_pod_spod.py` is a script export of the notebook. The notebook is the primary artifact; the script is provided for readers who prefer plain Python.

## Reproducibility

- Random seeds are fixed at 42 for NumPy, Python `random`, and TensorFlow.
- The SVD sign of a mode is arbitrary; a mode may flip sign across runs or libraries. Reconstructions and error metrics are unaffected.
- Exact metrics can vary slightly with hardware, TensorFlow version, and numerical libraries.
- The repository version of the notebook has no stored outputs. Run it to regenerate the plots and `pod_lstm_forecast.gif`.
- Dataset SHA-256: `99d9e1f9da7d2dff113ff2723373237b455e2de6c6747ac681846fe86fbba3d0`

## Limitations

- Only the streamwise velocity component $u$ is modeled. The cross-stream velocity $v$ and pressure field are not used.
- The LSTM is a small baseline trained on 150 sequences. It has no hyperparameter search, regularization study, or uncertainty estimate.
- Forecast error grows with horizon; the reported 12.84% mean error should not be interpreted as long-horizon accuracy.
- Sparse reconstruction assumes a known POD basis, fixed sensor locations, and noise-free measurements.
- QR pivoting is greedy and is not a global optimum.
- SPOD is not implemented, despite the `ROM_pod_spod` filename.
- The notebook is not a CFD solver, a PINN, or a digital twin.

## Future Work

1. Implement SPOD to separate coherent structures by frequency and compare it with POD.
2. Compare the LSTM against DMD, operator inference, and sequence-to-sequence models.
3. Include the cross-stream velocity and pressure fields.
4. Study robust sensor placement under measurement noise and sensor failure.
5. Add hyperparameter sweeps and uncertainty quantification for the forecast.
6. Refactor the notebook into reusable modules with tests and a pinned requirements file.

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.
2. Taira, K., Brunton, S. L., Dawson, S. T. M., Rowley, C. W., Colonius, T., McKeon, B. J., Schmidt, O. T., Gordeyev, S., Theofilis, V., & Ukeiley, L. S. (2017). Modal analysis of fluid flows: An overview. *AIAA Journal*, 55(12), 4013–4041.
3. Manohar, K., Brunton, B. W., Kutz, J. N., & Brunton, S. L. (2018). Data-driven sparse sensor placement for reconstruction: Demonstrating the benefits of exploiting known patterns. *IEEE Control Systems Magazine*, 38(3), 63–86.
4. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
5. Brunton, S. L., & Kutz, J. N. (2022). *Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control* (2nd ed.). Cambridge University Press.

## Author

**Sajeed Hussain**  
Aerospace Engineering  
GitHub: [HSajeed](https://github.com/HSajeed)  
Email: [sajeedhs26@gmail.com](mailto:sajeedhs26@gmail.com)
