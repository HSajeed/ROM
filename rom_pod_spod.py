import os
import urllib.request

# URL to the official raw .mat file
url = "https://github.com/maziarraissi/PINNs/raw/master/main/Data/cylinder_nektar_wake.mat"
filename = "cylinder_nektar_wake.mat"

if not os.path.exists(filename):
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print("Download complete!")
else:
    print("File already exists.")

import random
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

os.makedirs("docs", exist_ok=True)

# Load the file
data = scipy.io.loadmat('cylinder_nektar_wake.mat')

# Print keys and shapes
print("Keys in .mat file:", data.keys())

# Check the shape of the main velocity variable
print("\nShape of U_star:", data['U_star'].shape)
print("Shape of X_star:", data['X_star'].shape)
print("Shape of t:", data['t'].shape)

# Extract exact shapes based on inspection
# Shape: (Space=5000, Comp=2, Time=200) -> We want (5000, 200)

X_all = data['U_star'][:, 0, :]
Coords = data['X_star']          # (x, y) coordinates for plotting

print(f"   Shape extracted: {X_all.shape} (Rows=Space, Cols=Time)")

# 1. DATA LOADING (Benchmark Dataset)

def load_real_data():
    data = scipy.io.loadmat('cylinder_nektar_wake.mat')

    # U_star shape is (N_points, N_components, N_timesteps)
    # Use the streamwise component 'u' and keep (space, time) orientation.
    U_star = data['U_star']
    X_star = data['X_star']  # Spatial coordinates (x, y)
    t_star = data['t']       # Time

    U_data = U_star[:, 0, :]  # (N_points, N_timesteps)

    return U_data, X_star, t_star

print("Loading data...")
X_all, Coords, Time = load_real_data()
n_points, n_snapshots = X_all.shape
print(f"Loaded: {n_snapshots} snapshots with {n_points} spatial points each.")


# --- 2. POD (COMPRESSION) ---

# X_all is (space, time); SVD expects (features x samples).
X_matrix = X_all

print("Computing SVD (Modes)...")
# U: Spatial Modes, S: Energy, VT: Temporal Dynamics
U, S, VT = np.linalg.svd(X_matrix, full_matrices=False)
V = VT.T

# Desired truncation rank, capped by the number of available singular values.
# For this dataset len(S)=200, so r=10.
r = min(10, len(S))

U_r = U[:, :r]      # The "Eigen-Wakes"
Sigma_r = np.diag(S[:r])
V_r = V[:, :r]      # The time coefficients to predict

# Visualization of Energy
energy = np.cumsum(S**2) / np.sum(S**2)
print(f"Top {r} modes capture {energy[r-1]*100:.4f}% of the flow energy.")


# --- ADAPTIVE VISUALIZATION ---
print("Diagnosing shapes...")
print(f"U shape: {U.shape}")
print(f"VT shape: {VT.shape}")

# LOGIC: Find which variable has 5000 rows (Space)
if U.shape[0] == 5000:
    print(">> Spatial modes are in U. Plotting U...")
    Modes = U
elif VT.shape[1] == 5000:
    print(">> Spatial modes are in VT. Plotting VT.T...")
    Modes = VT.T
else:
    raise ValueError("Could not find dimension 5000 in U or VT!")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, mode_idx, title in zip(
    axes,
    [0, 1],
    ["Mode 1 (mean-flow-like)", "Mode 2 (vortex shedding)"],
):
    sc = ax.scatter(
        Coords[:, 0], Coords[:, 1],
        c=Modes[:, mode_idx], cmap="RdBu_r", s=5,
    )
    fig.colorbar(sc, ax=ax)
    ax.set_title(title)
    ax.set_xlim([-1, 8])
    ax.set_ylim([-3, 3])
    ax.set_aspect("equal")

fig.suptitle(f"POD modes of the streamwise velocity field (r={r})", fontsize=14)
fig.tight_layout()
plt.savefig("docs/pod_modes.png", dpi=200, bbox_inches="tight")
plt.show()

# --- 3. LSTM TRAINING ---

# time_coeffs = V_r @ Sigma_r, shape (200, r).
# Sigma is included here so the LSTM learns energy-weighted coefficients.
time_coeffs = V_r @ Sigma_r

print(f"Training data shape: {time_coeffs.shape} (should be 200 x {r})")

# 2. PREPARE SEQUENCES
lookback = 10
X_train_lstm = []
y_train_lstm = []

for i in range(len(time_coeffs) - lookback):
    X_train_lstm.append(time_coeffs[i:i+lookback])
    y_train_lstm.append(time_coeffs[i+lookback])

X_train_lstm = np.array(X_train_lstm)
y_train_lstm = np.array(y_train_lstm)

# 3. SPLIT
split = 150
x_tr, x_val = X_train_lstm[:split], X_train_lstm[split:]
y_tr, y_val = y_train_lstm[:split], y_train_lstm[split:]

# 4. MODEL
model = Sequential([
    LSTM(64, input_shape=(lookback, r), return_sequences=False),
    Dense(r)
])
model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

print("Training LSTM...")
history = model.fit(x_tr, y_tr, epochs=100, batch_size=16, verbose=1, validation_data=(x_val, y_val))
print("Model Trained.")

# Plot loss to check for overfitting
fig, ax = plt.subplots(figsize=(8, 4.5))
epochs = range(1, len(history.history["loss"]) + 1)
ax.plot(epochs, history.history["loss"], label="Training loss")
ax.plot(epochs, history.history["val_loss"], label="Validation loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE loss on POD coefficients")
ax.set_title("LSTM training and validation loss")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
plt.savefig("docs/lstm_training.png", dpi=200, bbox_inches="tight")
plt.show()

# --- 4. PREDICTION & RECONSTRUCTION ---

# Autoregressive loop
current_batch = x_val[0].reshape(1, lookback, r)
predicted_coeffs = []

for i in range(len(x_val)):
    pred = model.predict(current_batch, verbose=0)[0]
    predicted_coeffs.append(pred)
    current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)

predicted_coeffs = np.array(predicted_coeffs)

# time_coeffs already includes Sigma, so x = U_r a (do not apply Sigma twice).
X_reconstructed = (U_r @ predicted_coeffs.T).T

# Field-level forecast error over the held-out horizon.
start_idx = split + lookback
truth_forecast = X_all[:, start_idx:start_idx + len(predicted_coeffs)].T
err_forecast = (
    np.linalg.norm(X_reconstructed - truth_forecast, axis=1)
    / np.linalg.norm(truth_forecast, axis=1)
)
print(f"Mean relative forecast error: {err_forecast.mean():.4f}")
print(f"Max relative forecast error:  {err_forecast.max():.4f}")


# --- 6. VISUALIZATION (GIF) ---
print("Generating forecast animation...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Align timestamps: the first prediction corresponds to split + lookback.
# start_idx is defined in the prediction cell.

# 1. Ground Truth Plot
# X_all is (5000, 200), so we slice columns: X_all[:, index]
scat1 = ax1.scatter(Coords[:, 0], Coords[:, 1],
                    c=X_all[:, start_idx], cmap='jet', vmin=-1, vmax=1.5, s=5)
ax1.set_title("Ground truth (CFD)")
ax1.set_xlim([-1, 8]); ax1.set_ylim([-3, 3])
ax1.set_aspect('equal')

# 2. Prediction Plot
# X_reconstructed is (N_preds, 5000), so we slice rows: X_reconstructed[0, :]
scat2 = ax2.scatter(Coords[:, 0], Coords[:, 1],
                    c=X_reconstructed[0, :], cmap='jet', vmin=-1, vmax=1.5, s=5)
ax2.set_title(f"POD-LSTM forecast\nModes: {r}")
ax2.set_xlim([-1, 8]); ax2.set_ylim([-3, 3])
ax2.set_aspect('equal')

def update(frame):
    # Update Ground Truth (column slice)
    scat1.set_array(X_all[:, start_idx + frame])

    # Update Prediction (row slice)
    scat2.set_array(X_reconstructed[frame, :])
    return scat1, scat2

ani = animation.FuncAnimation(fig, update, frames=len(predicted_coeffs), blit=False)
ani.save('docs/pod_lstm_forecast.gif', writer='pillow', fps=15)
print("Done! GIF saved as 'docs/pod_lstm_forecast.gif'")

# --- 1. LOAD DATA & MODES ---

# Reload so Part 2 can run independently of Part 1.
data = scipy.io.loadmat('cylinder_nektar_wake.mat')
X_all = data['U_star'][:, 0, :]   # (5000, 200)
Coords = data['X_star']           # (5000, 2)
U, S, VT = np.linalg.svd(X_all, full_matrices=False)
r = min(10, len(S))
U_r = U[:, :r]

# --- 2. PLACE SENSORS ---
# Let's pick 5 random locations in the wake
# We use a fixed seed so we get the same sensors every time
np.random.seed(42)
n_sensors = 5

# Choose random indices from the 5000 available points
sensor_idx = np.random.choice(X_all.shape[0], n_sensors, replace=False)
sensor_locs = Coords[sensor_idx]

print(f"Placed {n_sensors} sensors at indices: {sensor_idx}")

# --- 3. COMPUTE SENSOR MATRIX ---
# We extract the POD modes ONLY at the sensor locations
# Shape: (n_sensors, r) -> (5, 10)
C = U_r[sensor_idx, :]

# Pre-compute the Pseudo-Inverse: (C^T * C)^-1 * C^T
# This matrix converts "5 sensor readings" -> "10 mode coefficients"
Theta = np.linalg.pinv(C)

print(f"Reconstruction Matrix Shape: {Theta.shape}")

# --- 4. RUN RECONSTRUCTION ---
# Reconstruct frame 159 from 5 random sensor measurements.
true_snapshot = X_all[:, 159]            # Full field (5000 points)
measurements = true_snapshot[sensor_idx] # Sensor readings (5 points)

# Estimate POD coefficients from sparse data
a_est = Theta @ measurements

# Reconstruct full field
X_sparse_rec = U_r @ a_est

err_sparse = np.linalg.norm(true_snapshot - X_sparse_rec) / np.linalg.norm(true_snapshot)
print(f"Relative reconstruction error with {n_sensors} random sensors: {err_sparse:.4f}")


# --- 5. VISUALIZATION ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot True Field
scat1 = ax1.scatter(Coords[:,0], Coords[:,1], c=true_snapshot, cmap='jet', vmin=-1, vmax=1.5, s=5)
ax1.set_title("Ground Truth (Full Data)")
ax1.set_xlim([-1, 8]); ax1.set_ylim([-3, 3])
ax1.set_aspect('equal')

# Plot Reconstructed Field + Sensor Locations
scat2 = ax2.scatter(Coords[:,0], Coords[:,1], c=X_sparse_rec, cmap='jet', vmin=-1, vmax=1.5, s=5)
# Overlay the sensor positions as big black dots
ax2.scatter(sensor_locs[:,0], sensor_locs[:,1], c='black', s=100, marker='x', label='Sensors')
ax2.legend(loc='upper right')
ax2.set_title(f"Reconstruction from {n_sensors} random sensors")
ax2.set_xlim([-1, 8]); ax2.set_ylim([-3, 3])
ax2.set_aspect('equal')

plt.show()


import scipy.linalg

# --- QR-PIVOTED SENSOR PLACEMENT ---
print("Finding QR-pivoted sensor locations...")

# Use the spatial POD modes U_r. QR with column pivoting on U_r.T
# orders spatial points by their pivoting importance.
Q, R, P = scipy.linalg.qr(U_r.T, pivoting=True)

# Use as many sensors as retained modes for a square sensor matrix.
s = r
sensor_indices = P[:s]
sensor_locs_qr = Coords[sensor_indices]

print(f"Indices of {s} QR-pivoted sensors: {sensor_indices}")

# --- RECONSTRUCTION WITH QR-PIVOTED SENSORS ---
C_opt = U_r[sensor_indices, :]
Theta_opt = np.linalg.pinv(C_opt)

measurements_opt = X_all[sensor_indices, 150]
a_opt = Theta_opt @ measurements_opt
X_smart_rec = U_r @ a_opt

err_smart = np.linalg.norm(X_all[:, 150] - X_smart_rec) / np.linalg.norm(X_all[:, 150])
print(f"Relative reconstruction error with {s} QR-pivoted sensors: {err_smart:.3e}")

# --- VISUALIZATION ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

scat1 = ax1.scatter(Coords[:,0], Coords[:,1], c=X_all[:, 150], cmap='jet', vmin=-1, vmax=1.5, s=5)
ax1.set_title("Ground truth (frame 150)")
ax1.set_xlim([-1, 8]); ax1.set_ylim([-3, 3])
ax1.set_aspect('equal')
fig.colorbar(scat1, ax=ax1)

scat2 = ax2.scatter(Coords[:,0], Coords[:,1], c=X_smart_rec, cmap='jet', vmin=-1, vmax=1.5, s=5)
ax2.scatter(sensor_locs_qr[:,0], sensor_locs_qr[:,1], c='lime', s=150, marker='D', edgecolors='black', label='QR-pivoted sensors')
ax2.legend(loc='upper right')
ax2.set_title(f"Reconstruction from {s} QR-pivoted sensors\nrelative error = {err_smart:.3%}")
ax2.set_xlim([-1, 8]); ax2.set_ylim([-3, 3])
ax2.set_aspect('equal')
fig.colorbar(scat2, ax=ax2)

fig.tight_layout()
plt.savefig('docs/qr_reconstruction.png', dpi=200, bbox_inches='tight')
plt.show()

# --- SENSOR COUNT SWEEP ---
print("Running sensor sweep (1 to 20)...")

errors = []
sensor_counts = list(range(1, 21))
ground_truth = X_all[:, 150]  # Target frame

# Compute the QR pivoting order once.
Q, R, P = scipy.linalg.qr(U_r.T, pivoting=True)

for p in sensor_counts:
    current_indices = P[:p]
    C_p = U_r[current_indices, :]
    Theta_p = np.linalg.pinv(C_p)
    measurements = ground_truth[current_indices]
    a_est = Theta_p @ measurements
    X_rec = U_r @ a_est
    error = np.linalg.norm(ground_truth - X_rec) / np.linalg.norm(ground_truth)
    errors.append(error)

for p in [9, 10, 20]:
    print(f"p={p:2d}: relative error = {errors[p-1]:.3e}")

plt.figure(figsize=(10, 6))
plt.plot(sensor_counts, errors, 'bo-', linewidth=2)
plt.axvline(x=r, color='r', linestyle='--', label=f'r = {r} modes')
plt.title(f"Reconstruction error vs. number of sensors (r={r})")
plt.xlabel("Number of QR-pivoted sensors")
plt.ylabel("Relative reconstruction error")
plt.grid(True, which='both', alpha=0.3)
plt.legend()
plt.yscale('log')
plt.savefig('docs/sensor_sweep.png', dpi=200, bbox_inches='tight')
plt.show()

# --- SENSOR-MATRIX CONDITIONING ---
print("Checking sensor-matrix conditioning...")

cond_numbers = []
sensor_counts = list(range(1, 15))

Q, R, P = scipy.linalg.qr(U_r.T, pivoting=True)

for p in sensor_counts:
    current_indices = P[:p]
    C_p = U_r[current_indices, :]
    cond_numbers.append(np.linalg.cond(C_p))

plt.figure(figsize=(10, 6))
plt.plot(sensor_counts, cond_numbers, 'r-o', linewidth=2)
plt.axvline(x=r, color='k', linestyle='--', label=f'r={r}')
plt.yscale('log')
plt.title("Sensor-matrix condition number vs. number of sensors")
plt.xlabel("Number of sensors")
plt.ylabel(r"Condition number $\kappa(C)$")
plt.grid(True)
plt.legend()
plt.show()
