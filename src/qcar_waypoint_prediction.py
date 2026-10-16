"""
QCar - N-step state prediction with a discretized Ackermann (kinematic bicycle) model.

The continuous model is discretized with two integrators, explicit Euler and
4th-order Runge-Kutta (RK4), and evaluated from several waypoints of a recorded
(pre-run) trajectory.

For each selected waypoint:
  - the real pose (x, y, yaw) is used as the initial condition X(k)
  - X(k+1) ... X(k+N_STEPS) are predicted with Euler and with RK4 (step DT)
  - both predictions are compared against the real trajectory and against a
    fine-step RK4 reference (step DT_REF)

Usage (from the repository root):
    pip install -r requirements.txt
    python src/qcar_waypoint_prediction.py
"""
import os
import numpy as np
import pandas as pd
import casadi as ca
import matplotlib.pyplot as plt
from mcap_ros2.reader import read_ros2_messages

# ---------------------------------------------------------------- Paths
# Paths are resolved relative to the repository root, not the working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAG_FILE = os.path.join(REPO_ROOT, "data", "rosbag", "qcar_bag_20261009_164244",
                        "qcar_bag_20261009_164244_0.mcap")
WAYPOINTS = os.path.join(REPO_ROOT, "data", "waypoints", "waypoints.csv")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------- Parameters
L = 0.256            # [m] QCar wheelbase
DT = 0.050           # [s] discretization step
DT_REF = 0.001       # [s] fine reference step (1 ms)
N_STEPS = 30         # predictions per waypoint: X(k+1) ... X(k+N_STEPS)
N_WAYPOINTS = 10     # number of starting waypoints (5-10)
V_THRESH = 0.05      # [m/s] threshold used to detect start/end of motion

# Steering calibration (provisional): the steering command does not match the
# actual wheel angle directly, nor does it act at the same instant.
STEER_DELAY = 2.2    # [s] delay between command and response
STEER_GAIN = 1.6     # [-] delta_real = STEER_GAIN * command

# ---------------------------------------------------------------- 1. Model
x, y, th = ca.SX.sym("x"), ca.SX.sym("y"), ca.SX.sym("theta")
z = ca.vertcat(x, y, th)
v, delta = ca.SX.sym("v"), ca.SX.sym("delta")
u = ca.vertcat(v, delta)
dt = ca.SX.sym("dt")

z_dot = ca.vertcat(v * ca.cos(th), v * ca.sin(th), v * ca.tan(delta) / L)
f = ca.Function("f", [z, u], [z_dot])

euler_step = ca.Function("euler", [z, u, dt], [z + dt * f(z, u)])

k1 = f(z, u)
k2 = f(z + 0.5 * dt * k1, u)
k3 = f(z + 0.5 * dt * k2, u)
k4 = f(z + dt * k3, u)
rk4_step = ca.Function("rk4", [z, u, dt], [z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)])

# ---------------------------------------------------------------- 2. Read the rosbag
def read_topic(topic):
    t, data = [], []
    for m in read_ros2_messages(BAG_FILE, topics=[topic]):
        t.append(m.log_time_ns * 1e-9)
        data.append([m.ros_msg.vector.x, m.ros_msg.vector.y])
    return np.array(t), np.array(data)

t_vel, vel = read_topic("/qcar/velocity")       # vel[:,0] = measured speed [m/s]
t_cmd, cmd = read_topic("/qcar/user_command")   # cmd[:,1] = steering command [rad]
t0 = t_vel[0]
t_vel, t_cmd = t_vel - t0, t_cmd - t0
v_meas, steer = vel[:, 0], cmd[:, 1]

idx = np.where(v_meas > V_THRESH)[0]
t_start, t_end = t_vel[idx[0]], t_vel[idx[-1]]

# ---------------------------------------------------------------- 3. Real trajectory
wp = pd.read_csv(WAYPOINTS)
mask = (t_vel >= t_start) & (t_vel <= t_end)
tv, vv = t_vel[mask], v_meas[mask]
s_t = np.concatenate([[0], np.cumsum(0.5 * (vv[1:] + vv[:-1]) * np.diff(tv))])   # s(t)
s_wp = np.concatenate([[0], np.cumsum(np.hypot(np.diff(wp.x), np.diff(wp.y)))])
s_wp *= s_t[-1] / s_wp[-1]
yaw_wp = np.unwrap(wp.yaw.values)
t_wp = np.interp(s_wp, s_t, tv)                  # time associated with each waypoint

def true_state(t):
    s = np.interp(t, tv, s_t)
    return np.array([np.interp(s, s_wp, wp.x), np.interp(s, s_wp, wp.y),
                     np.interp(s, s_wp, yaw_wp)])

# ---------------------------------------------------------------- 4. Waypoint selection
horizon = N_STEPS * DT
valid = np.where(t_wp + horizon <= t_end)[0]     # horizon must fit inside the motion
if len(valid) == 0:
    raise SystemExit(
        f"The horizon N_STEPS*DT = {N_STEPS} x {DT} = {horizon:.1f} s is longer than the "
        f"recorded motion ({t_end - t_start:.1f} s), so there is no real data to compare "
        f"against. Reduce N_STEPS or DT (N_STEPS*DT must be below ~{t_end - t_start:.0f} s).")
wp_idx = valid[np.linspace(0, len(valid) - 1, N_WAYPOINTS).astype(int)]

# ---------------------------------------------------------------- 5. Prediction
def inputs_at(t):
    return [np.interp(t, t_vel, v_meas),
            STEER_GAIN * np.interp(t - STEER_DELAY, t_cmd, steer)]

def predict(step_fn, t_init, z_init, dt_s, n):
    zk = np.array(z_init, dtype=float)
    traj = [zk]
    for k in range(n):
        zk = np.array(step_fn(zk, inputs_at(t_init + k * dt_s), dt_s).full()).flatten()
        traj.append(zk)
    return np.array(traj)

pred = []
for i in wp_idx:
    ti = t_wp[i]
    z0 = np.array([wp.x[i], wp.y[i], yaw_wp[i]])
    e = predict(euler_step, ti, z0, DT, N_STEPS)
    r = predict(rk4_step, ti, z0, DT, N_STEPS)
    n_ref = int(round(N_STEPS * DT / DT_REF))
    ref = predict(rk4_step, ti, z0, DT_REF, n_ref)[:: int(round(DT / DT_REF))]  # same instants
    real = np.array([true_state(ti + k * DT) for k in range(N_STEPS + 1)])
    pred.append(dict(i=i, t=ti, euler=e, rk4=r, ref=ref, real=real))

# ---------------------------------------------------------------- 6. Plots
# 6a. Full trajectory with the predictions from each waypoint
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(wp.x, wp.y, color="limegreen", lw=3, label="Real trajectory (pre-run)")
for j, p in enumerate(pred):
    ax.plot(p["euler"][:, 0], p["euler"][:, 1], "r-", lw=2, label="Euler" if j == 0 else None)
    ax.plot(p["rk4"][:, 0], p["rk4"][:, 1], "b--", lw=2, label="RK4" if j == 0 else None)
    ax.plot(p["euler"][0, 0], p["euler"][0, 1], "o", color="darkred", ms=8,
            label="Starting waypoint of each prediction" if j == 0 else None)
    ax.annotate(str(j + 1), p["euler"][0, :2], textcoords="offset points", xytext=(6, 6), fontsize=9)
ax.set_title(f"Predictions X(k+1)...X(k+{N_STEPS}) from {N_WAYPOINTS} waypoints "
             f"(dt = {DT*1000:.0f} ms, horizon = {horizon:.2f} s)")
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.axis("equal")
ax.grid(True, ls="--", alpha=0.5); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "prediction_global.png"), dpi=150)

# 6b. Zoom on each waypoint
ncol = 5
nrow = int(np.ceil(N_WAYPOINTS / ncol))
fig2, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow), squeeze=False)
for j, (p, ax) in enumerate(zip(pred, axs.flat)):
    ax.plot(p["real"][:, 0], p["real"][:, 1], "k-", lw=3, label="Real")
    ax.plot(p["ref"][:, 0], p["ref"][:, 1], "g:", lw=2, label=f"RK4 ref. {DT_REF*1000:.0f} ms")
    ax.plot(p["euler"][:, 0], p["euler"][:, 1], "r.-", ms=4, label="Euler")
    ax.plot(p["rk4"][:, 0], p["rk4"][:, 1], "b.--", ms=4, label="RK4")
    ax.plot(*p["real"][0, :2], "o", color="darkred", ms=7)
    ax.set_title(f"Waypoint {j+1} (t = {p['t']-t_start:.1f} s)", fontsize=10)
    ax.axis("equal"); ax.grid(True, ls="--", alpha=0.5)
    if j == 0:
        ax.legend(fontsize=7)
for ax in list(axs.flat)[N_WAYPOINTS:]:
    ax.axis("off")
fig2.suptitle(f"Detail of each prediction ({N_STEPS} steps of {DT*1000:.0f} ms)", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "prediction_zoom.png"), dpi=150)

# 6c. Errors
k_axis = np.arange(N_STEPS + 1)
err = lambda a, b: np.linalg.norm(a[:, :2] - b[:, :2], axis=1)
fig3, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
for p in pred:
    a1.plot(k_axis, err(p["euler"], p["real"]), "r", alpha=0.25)
    a1.plot(k_axis, err(p["rk4"], p["real"]), "b", alpha=0.25)
a1.plot(k_axis, np.mean([err(p["euler"], p["real"]) for p in pred], 0), "r", lw=2.5, label="Euler (mean)")
a1.plot(k_axis, np.mean([err(p["rk4"], p["real"]) for p in pred], 0), "b--", lw=2.5, label="RK4 (mean)")
a1.set_title("Error vs real trajectory (model + inputs + discretization)")
a1.set_xlabel("k (steps ahead)"); a1.set_ylabel("position error [m]")
a1.grid(True, ls="--", alpha=0.5); a1.legend()

eu = np.mean([err(p["euler"], p["ref"]) for p in pred], 0)
rk = np.mean([err(p["rk4"], p["ref"]) for p in pred], 0)
a2.semilogy(k_axis[1:], np.maximum(eu[1:], 1e-12), "r", lw=2.5, label="Euler (mean)")
a2.semilogy(k_axis[1:], np.maximum(rk[1:], 1e-12), "b--", lw=2.5, label="RK4 (mean)")
a2.set_title(f"Discretization error only (vs RK4 at {DT_REF*1000:.0f} ms)")
a2.set_xlabel("k (steps ahead)"); a2.set_ylabel("position error [m] (log scale)")
a2.grid(True, which="both", ls="--", alpha=0.5); a2.legend()
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "prediction_error.png"), dpi=150)

print(f"Waypoints used (CSV row indices): {[int(i) for i in wp_idx]}")
print(f"Prediction horizon: {horizon:.2f} s  ({N_STEPS} steps of {DT*1000:.0f} ms)")
print(f"Mean error at k={N_STEPS} vs real      -> Euler: "
      f"{np.mean([err(p['euler'], p['real'])[-1] for p in pred]):.4f} m | "
      f"RK4: {np.mean([err(p['rk4'], p['real'])[-1] for p in pred]):.4f} m")
print(f"Mean error at k={N_STEPS} vs 1 ms ref. -> Euler: {eu[-1]:.2e} m | RK4: {rk[-1]:.2e} m")
plt.show()
