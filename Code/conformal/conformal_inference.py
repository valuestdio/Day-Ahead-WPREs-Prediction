import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import brentq

# The data has a change point halfway through.
folder_path = './SDWPF/Hiformer/Denormalization/'
file_path = folder_path + 'test_deno.csv'
data = pd.read_csv(file_path)

# Extract the columns and save as numpy arrays
mean = data['Mean'].to_numpy()
var = data['Var'].to_numpy()
gts = data['gts'].to_numpy()

a = 0.5  # 调整幅度参数
k = 20 # 对数的底数
var = 0.5 + a * (np.log(var + 1) / np.log(k))
# Visualize the average accuracy,计算移动平均，将一个数组x卷积全是1的窗口w
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# Problem setup
alpha = 0.1 # 1-alpha is the desired coverage
K=144; weights = np.ones((K,)); # Take a fixed window of K （1000个1）
# 权重从0.99^999到0.99^0
exponent_weights = 0.99 ** (np.arange(K)[::-1])
wtildes = exponent_weights / (exponent_weights.sum() + 1)

scores = np.abs(mean - gts) / var

def get_weighted_quantile(scores, T):
    score_window = scores[T-K:T]

    def critical_point_quantile(q):
        return (wtildes * (score_window <= q)).sum() - (1 - alpha)

    return brentq(critical_point_quantile, 0, 100)

# =========================
# Weighted conformal
# =========================
start_weighted = time.time()

qhats = np.array([
    get_weighted_quantile(scores, t)
    for t in range(K, scores.shape[0])
])

prediction_sets = [
    mean[K:] - var[K:] * qhats,
    mean[K:] + var[K:] * qhats
]

prediction_sets = [
    np.clip(prediction_sets[0], 0, 200),
    np.clip(prediction_sets[1], 0, 200)
]

end_weighted = time.time()
print(f"Weighted Conformal Prediction Time: {end_weighted - start_weighted:.4f} seconds")

# =========================
# Naive conformal
# =========================
start_naive = time.time()

naive_qhats = np.array([
    np.quantile(scores[:t], np.ceil((t + 1) * (1 - alpha)) / t, method='higher')
    for t in range(K, scores.shape[0])
])

naive_prediction_sets = [
    mean[K:] - var[K:] * naive_qhats,
    mean[K:] + var[K:] * naive_qhats
]

naive_prediction_sets = [
    np.clip(naive_prediction_sets[0], 0, 200),
    np.clip(naive_prediction_sets[1], 0, 200)
]

end_naive = time.time()
print(f"Naive Conformal Prediction Time: {end_naive - start_naive:.4f} seconds")

# =========================
# Coverage
# =========================
gts_eff = gts[K:]

covered = (gts_eff >= prediction_sets[0]) & (gts_eff <= prediction_sets[1])
coverage_over_time = moving_average(covered, 500)

naive_covered = (gts_eff >= naive_prediction_sets[0]) & (gts_eff <= naive_prediction_sets[1])
naive_coverage_over_time = moving_average(naive_covered, 500)

coverage_mean = coverage_over_time.mean()
naive_coverage_mean = naive_coverage_over_time.mean()
coverage_variance = coverage_over_time.var()
naive_coverage_variance = naive_coverage_over_time.var()

df = pd.DataFrame({
    'timestamp': range(len(coverage_over_time)),
    'weighted': coverage_over_time,
    'unweighted': naive_coverage_over_time
})
df.to_csv(folder_path + 'coverage_results.csv', index=False)

print(f"Average coverage over time: {coverage_mean}")
print(f"Average naive coverage over time: {naive_coverage_mean}")
print(f"Variance of coverage over time: {coverage_variance}")
print(f"Variance of naive coverage over time: {naive_coverage_variance}")

prediction_width = (prediction_sets[1] - prediction_sets[0]).mean()
naive_prediction_width = (naive_prediction_sets[1] - naive_prediction_sets[0]).mean()

print(f"Average width of prediction_sets: {prediction_width}")
print(f"Average width of naive_prediction_sets: {naive_prediction_width}")

# =========================
# Save intervals
# =========================
predictions_df = pd.DataFrame({
    'lower bound': prediction_sets[0],
    'upper bound': prediction_sets[1],
    'width': prediction_sets[1] - prediction_sets[0]
})

naive_predictions_df = pd.DataFrame({
    'lower bound': naive_prediction_sets[0],
    'upper bound': naive_prediction_sets[1],
    'width': naive_prediction_sets[1] - naive_prediction_sets[0]
})

var_df = pd.DataFrame(var)

predictions_df.to_csv(folder_path + 'my_prediction_intervals.csv', index=False)
naive_predictions_df.to_csv(folder_path + 'naive_prediction_intervals.csv', index=False)
var_df.to_csv(folder_path + 'var.csv', index=False)

print("\n========== Mechanism Point Coverage ==========")

block_size = 4
Ksize = K

lower = prediction_sets[0]
upper = prediction_sets[1]
naive_lower = naive_prediction_sets[0]
naive_upper = naive_prediction_sets[1]

num_small_blocks = len(gts_eff) // block_size
blocks_per_big_block = Ksize // block_size
num_big_blocks = num_small_blocks // blocks_per_big_block

small_block_records = []
records = []

for b in range(num_small_blocks):
    s = b * block_size
    e = s + block_size
    blk = gts_eff[s:e]

    blk_max_idx = np.argmax(blk)
    blk_min_idx = np.argmin(blk)

    blk_max = blk[blk_max_idx]
    blk_min = blk[blk_min_idx]

    t_max = s + blk_max_idx
    t_min = s + blk_min_idx

    cover_max = int(lower[t_max] <= blk_max <= upper[t_max])
    cover_min = int(lower[t_min] <= blk_min <= upper[t_min])

    naive_cover_max = int(naive_lower[t_max] <= blk_max <= naive_upper[t_max])
    naive_cover_min = int(naive_lower[t_min] <= blk_min <= naive_upper[t_min])

    small_block_records.append({
        'cover_max': cover_max,
        'cover_min': cover_min,
        'naive_cover_max': naive_cover_max,
        'naive_cover_min': naive_cover_min
    })

    max_global_ts = K + t_max
    min_global_ts = K + t_min

    records.append({
        "block_id": b,
        "type": "max",
        "global_timestamp": max_global_ts,
        "idx_in_block": blk_max_idx,
        "gts_value": blk_max,
        "lower": lower[t_max],
        "upper": upper[t_max],
        "covered_weighted": cover_max,
        "covered_naive": naive_cover_max
    })

    records.append({
        "block_id": b,
        "type": "min",
        "global_timestamp": min_global_ts,
        "idx_in_block": blk_min_idx,
        "gts_value": blk_min,
        "lower": lower[t_min],
        "upper": upper[t_min],
        "covered_weighted": cover_min,
        "covered_naive": naive_cover_min
    })

df_ext = pd.DataFrame(records)
df_ext.to_csv(folder_path + "extreme_points_detailed.csv", index=False)
print(f"[INFO] Saved detailed extreme point CSV → {folder_path}extreme_points_detailed.csv")

mech_cover = []
naive_mech_cover = []

for B in range(num_big_blocks):
    s = B * blocks_per_big_block
    e = s + blocks_per_big_block
    subset = small_block_records[s:e]

    covered_max = sum(x['cover_max'] for x in subset)
    covered_min = sum(x['cover_min'] for x in subset)

    naive_covered_max = sum(x['naive_cover_max'] for x in subset)
    naive_covered_min = sum(x['naive_cover_min'] for x in subset)

    total = 2 * len(subset)

    mech_cover.append((covered_max + covered_min) / total)
    naive_mech_cover.append((naive_covered_max + naive_covered_min) / total)

mech_cover = np.array(mech_cover)
naive_mech_cover = np.array(naive_mech_cover)

print(f"Mechanism-point coverage (weighted): {mech_cover.mean():.4f}")
print(f"Mechanism-point coverage (naive): {naive_mech_cover.mean():.4f}")

df_mech = pd.DataFrame({
    'block_id': np.arange(num_big_blocks),
    'weighted_mech_coverage': mech_cover,
    'naive_mech_coverage': naive_mech_cover
})
df_mech.to_csv(folder_path + 'mechanism_point_coverage.csv', index=False)

# =========================
# Plot mechanism coverage
# =========================
figm, axm = plt.subplots(figsize=(15, 8))

expanded_mech_cover = np.repeat(mech_cover, Ksize)
expanded_naive_mech_cover = np.repeat(naive_mech_cover, Ksize)
expanded_time = np.arange(len(expanded_mech_cover)) + K

axm.plot(expanded_time, expanded_mech_cover, label='weighted mechanism-point coverage')
axm.plot(expanded_time, expanded_naive_mech_cover, label='naive mechanism-point coverage')

sns.despine(ax=axm, top=True, right=True)
axm.set_xlabel("Time Points")
axm.set_ylabel("Mechanism Point Coverage")
axm.legend()

plt.tight_layout()
plt.savefig(folder_path + 'mechanism_point_coverage.pdf')
plt.close(figm)

# =========================
# Plot coverage
# =========================
plt.rcParams.update({'font.size': 12})

fig1, ax1 = plt.subplots(figsize=(15, 10))
ax1.plot(coverage_over_time, label='weighted')
ax1.plot(naive_coverage_over_time, label='unweighted')

sns.despine(ax=ax1, top=True, right=True)
ax1.set_xlabel('timestamp')
ax1.set_ylabel('coverage\n(size 500 sliding window)')
ax1.legend()

plt.tight_layout()
plt.savefig(folder_path + 'coverage_results.pdf')
plt.close(fig1)

# =========================
# Plot prediction interval
# =========================
fig2, ax2 = plt.subplots(figsize=(15, 10))

start_point = 5000

timestamps = np.arange(K + start_point, len(mean))

ax2.plot(timestamps, mean[K + start_point:], color='#000000', label='prediction')
ax2.plot(timestamps, gts[K + start_point:], color='#00FF00', label='ground truth')

sns.despine(ax=ax2, top=True, right=True)

ax2.fill_between(
    timestamps,
    prediction_sets[0][start_point:],
    prediction_sets[1][start_point:],
    color='#D3D3D3',
    label='weighted conformal interval'
)

ax2.set_ylim(-250, 500)
ax2.locator_params(tight=True, nbins=4)
ax2.set_xlabel('timestamp')
ax2.set_ylabel(r'wind power')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(folder_path + 'prediction_results.pdf')
plt.close(fig2)

# =========================
# GARCH interval
# =========================
start_garch = time.time()

residuals = gts - mean

garch_model = arch_model(residuals, vol='Garch', p=1, q=1, dist='normal')
garch_res = garch_model.fit(disp='off')

garch_forecast = garch_res.forecast(horizon=len(residuals))
sigma_forecast = np.sqrt(garch_forecast.variance.values[-1, :])

resid_eff = residuals[K:]
sigma_eff = sigma_forecast[K:]
mean_eff = mean[K:]
gts_eff = gts[K:]

target_coverage = 0.9
z_low, z_high = 0.0, 5.0

for _ in range(20):
    z = (z_low + z_high) / 2

    lower_tmp = np.clip(mean_eff - z * sigma_eff, 0, 200)
    upper_tmp = np.clip(mean_eff + z * sigma_eff, 0, 200)

    coverage_tmp = ((gts_eff >= lower_tmp) & (gts_eff <= upper_tmp)).mean()

    if coverage_tmp > target_coverage:
        z_high = z
    else:
        z_low = z

z_opt = (z_low + z_high) / 2
print(f"Optimal z for target 0.9 coverage: {z_opt:.4f}")

garch_lower = np.clip(mean_eff - z_opt * sigma_eff, 0, 200)
garch_upper = np.clip(mean_eff + z_opt * sigma_eff, 0, 200)

end_garch = time.time()
print(f"GARCH Prediction Time: {end_garch - start_garch:.4f} seconds")

garch_predictions_df = pd.DataFrame({
    'lower bound': garch_lower,
    'upper bound': garch_upper,
    'width': garch_upper - garch_lower
})
garch_predictions_df.to_csv(folder_path + 'garch_prediction_intervals.csv', index=False)

garch_covered = (gts_eff >= garch_lower) & (gts_eff <= garch_upper)
garch_coverage_over_time = moving_average(garch_covered, 500)

print(f"Average GARCH coverage rate: {garch_coverage_over_time.mean():.4f}")

garch_prediction_width = (garch_upper - garch_lower).mean()
print(f"Average width of GARCH prediction_sets: {garch_prediction_width:.4f}")

# =========================
# Plot all coverage
# =========================
timestamps_plot = np.arange(len(coverage_over_time))

plt.figure(figsize=(15, 10))
plt.plot(timestamps_plot, coverage_over_time, label='Weighted Conformal')
plt.plot(timestamps_plot, naive_coverage_over_time, label='Naive Conformal')
plt.plot(timestamps_plot, garch_coverage_over_time, label='GARCH', linestyle='--')

sns.despine(top=True, right=True)
plt.xlabel('timestamp')
plt.ylabel('Coverage (500-point sliding window)')
plt.legend(loc='lower left')

plt.tight_layout()
plt.savefig(folder_path + 'coverage_comparison_all_methods.pdf')
plt.close()

# =========================
# GARCH mechanism-point coverage
# =========================
num_small_blocks = len(gts_eff) // block_size
blocks_per_big_block = Ksize // block_size
num_big_blocks = num_small_blocks // blocks_per_big_block

garch_small_records = []

for b in range(num_small_blocks):
    s = b * block_size
    e = s + block_size
    blk = gts_eff[s:e]

    blk_max_idx = np.argmax(blk)
    blk_min_idx = np.argmin(blk)

    blk_max = blk[blk_max_idx]
    blk_min = blk[blk_min_idx]

    t_max = s + blk_max_idx
    t_min = s + blk_min_idx

    cover_max = int(garch_lower[t_max] <= blk_max <= garch_upper[t_max])
    cover_min = int(garch_lower[t_min] <= blk_min <= garch_upper[t_min])

    garch_small_records.append({
        'cover_max': cover_max,
        'cover_min': cover_min
    })

garch_mech_cover = []

for B in range(num_big_blocks):
    s = B * blocks_per_big_block
    e = s + blocks_per_big_block
    subset = garch_small_records[s:e]

    covered_max = sum(x['cover_max'] for x in subset)
    covered_min = sum(x['cover_min'] for x in subset)

    total = 2 * len(subset)

    garch_mech_cover.append((covered_max + covered_min) / total)

garch_mech_cover = np.array(garch_mech_cover)

print(f"Mechanism-point coverage (GARCH): {garch_mech_cover.mean():.4f}")

df_garch_mech = pd.DataFrame({
    'block_id': np.arange(num_big_blocks),
    'garch_mech_coverage': garch_mech_cover
})
df_garch_mech.to_csv(folder_path + 'garch_mechanism_point_coverage.csv', index=False)

# =========================
# Plot all mechanism coverage
# =========================
figm_all, axm_all = plt.subplots(figsize=(15, 8))

expanded_weighted = np.repeat(mech_cover, Ksize)
expanded_naive = np.repeat(naive_mech_cover, Ksize)
expanded_garch = np.repeat(garch_mech_cover, Ksize)

min_len = min(len(expanded_weighted), len(expanded_naive), len(expanded_garch))

expanded_weighted = expanded_weighted[:min_len]
expanded_naive = expanded_naive[:min_len]
expanded_garch = expanded_garch[:min_len]

expanded_time = np.arange(min_len) + K

axm_all.plot(expanded_time, expanded_weighted, label='Weighted Conformal')
axm_all.plot(expanded_time, expanded_naive, label='Naive Conformal')
axm_all.plot(expanded_time, expanded_garch, label='GARCH', linestyle='--')

sns.despine(ax=axm_all, top=True, right=True)
axm_all.set_xlabel("Time Points")
axm_all.set_ylabel("Mechanism Point Coverage")
axm_all.legend()

plt.tight_layout()
plt.savefig(folder_path + 'mechanism_point_coverage_all_methods.pdf')
plt.close(figm_all)


