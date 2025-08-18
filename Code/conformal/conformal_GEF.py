import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import brentq

# The data has a change point halfway through.
folder_path = './GEFcom/Hiformer/Denormalization/'
file_path = folder_path + 'test_deno.csv'
data = pd.read_csv(file_path)
# Extract the columns and save as numpy arrays
mean = data['Mean'].to_numpy()
var = data['Var'].to_numpy()
gts = data['gts'].to_numpy()
a = 0.5 # 调整幅度参数
k = 2 # 对数的底数
var = 1 + a * (np.log(var + 1) / np.log(k))

# Visualize the average accuracy,计算移动平均，将一个数组x卷积全是1的窗口w
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# Problem setup
alpha = 0.1 # 1-alpha is the desired coverage
K=24; weights = np.ones((K,)); # Take a fixed window of K （1000个1）
# 权重从0.99^999到0.99^0
exponent_weights = 0.99 ** (np.arange(K)[::-1])  # 生成从0.99^999到0.99^0的数组
# 归一化
wtildes = exponent_weights / (exponent_weights.sum()+1)
# Use the uncertainty scalars method to get conformal scores
#用残差计算损失，这里用pred_uncertainty做加权处理，使其具有非交换性
scores = np.abs(mean-gts)/var
# Get the weighted score quantile at each time step
#计算每一个时间点上的conformal score，并进行窗口化切片，然后比较和q的大小，小于q的变成1/1001，使用brentq求解使得总和为1-alpha
def get_weighted_quantile(scores,T):
    score_window = scores[T-K:T]
    def critical_point_quantile(q): return (wtildes * (score_window <= q)).sum() - (1 - alpha)
    return brentq(critical_point_quantile, 0, 100)

start_weighted = time.time()

qhats = np.array( [get_weighted_quantile(scores, t) for t in range(K+1, scores.shape[0])] )#得到每个windows的加权分位数

#得到一个q数组
# Deploy (output=lower and upper adjusted quantiles)
prediction_sets = [mean[K+1:] - var[K+1:]*qhats, mean[K+1:] + var[K+1:]*qhats]#刻画不确定集合
#反归一化
prediction_sets = [
    np.clip(prediction_sets[0], 0, 10),  # 对下界限制范围
    np.clip(prediction_sets[1], 0, 10)   # 对上界限制范围
]

end_weighted = time.time()
print(f"Weighted Conformal Prediction Time: {end_weighted - start_weighted:.4f} seconds")
#归一化
#prediction_sets = [
    #np.clip(prediction_sets[0], -12, 20),  # 对下界限制范围
    #np.clip(prediction_sets[1], -12, 20)   # 对上界限制范围
#]
start_naive = time.time()
# For comparison, run naive conformal
naive_qhats = np.array([
    np.quantile(scores[:t], np.ceil((t + 1) * (1 - alpha)) / t, method='higher')
    for t in range(K + 1, scores.shape[0])
])
naive_prediction_sets = [mean[K + 1:] - var[K+1:]*naive_qhats, mean[K + 1:] + var[K+1:]*naive_qhats]
#反归一化
naive_prediction_sets = [
    np.clip(naive_prediction_sets[0], 0, 10),  # 对下界限制范围
    np.clip(naive_prediction_sets[1], 0, 10)   # 对上界限制范围
]

#归一化
#naive_prediction_sets = [
    #np.clip(naive_prediction_sets[0], -12, 20),  # 对下界限制范围
    #np.clip(naive_prediction_sets[1], -12, 20)   # 对上界限制范围
#]
end_naive = time.time()
print(f"Naive Conformal Prediction Time: {end_naive - start_naive:.4f} seconds")

# Calculate coverage over time
covered = ( gts[K+1:] >= prediction_sets[0] ) & ( gts[K+1:] <= prediction_sets[1] )
coverage_over_time = moving_average(covered, 500)
naive_covered = ( gts[K+1:] >= naive_prediction_sets[0] ) & ( gts[K+1:] <= naive_prediction_sets[1] )
naive_coverage_over_time = moving_average(naive_covered, 500)

# 计算平均值并打印
coverage_mean = coverage_over_time.mean()
naive_coverage_mean = naive_coverage_over_time.mean()

# 计算方差
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
# 计算 prediction_sets 和 naive_prediction_sets 的宽度
prediction_width = (prediction_sets[1] - prediction_sets[0]).mean()
naive_prediction_width = (naive_prediction_sets[1] - naive_prediction_sets[0]).mean()

# 打印平均宽度
print(f"Average width of prediction_sets: {prediction_width}")
print(f"Average width of naive_prediction_sets: {naive_prediction_width}")

predictions_df = pd.DataFrame({
    'lower bound': prediction_sets[0],
    'upper bound': prediction_sets[1],
    'width' : prediction_sets[1] - prediction_sets[0]
})
naive_predictions_df = pd.DataFrame({
    'lower bound': naive_prediction_sets[0],
    'upper bound': naive_prediction_sets[1],
    'width' : naive_prediction_sets[1] - naive_prediction_sets[0]
})
var_df = pd.DataFrame(var)

# Save to CSV
predictions_df.to_csv(folder_path + 'my_prediction_intervals.csv', index=False)
naive_predictions_df.to_csv(folder_path + 'naive_prediction_intervals.csv', index=False)
var_df.to_csv(folder_path + 'var.csv', index=False)
# Plot prediction sets and coverage over time
plt.rcParams.update({'font.size': 12})

# Create and save the first subplot
fig1, ax1 = plt.subplots(figsize=(15, 10))  # Adjust the size if needed
ax1.plot(coverage_over_time, label='weighted')
ax1.plot(naive_coverage_over_time, label='unweighted')
sns.despine(ax=ax1, top=True, right=True)
ax1.set_xlabel('timestamp')
ax1.set_ylabel(f'coverage\n(size 500 sliding window)')
ax1.legend()
plt.tight_layout()
plt.savefig(folder_path + 'coverage_results.pdf')  # Save the first plot
plt.close(fig1)

# Create and save the second subplot
fig2, ax2 = plt.subplots(figsize=(15, 10))  # Adjust the size if needed
start_point = 7000
timestamps = np.array(range(start_point, mean[K+1:].shape[0] + K + 1))
ax2.plot(timestamps, mean[start_point:], color='#000000', label='prediction')
ax2.plot(timestamps, gts[start_point:], color='#00FF00', label='ground truth')
sns.despine(ax=ax2, top=True, right=True)
ax2.fill_between(
    timestamps,
    prediction_sets[0][start_point - K - 1:],
    prediction_sets[1][start_point - K - 1:],
    color='#D3D3D3',
    label='weighted conformal interval'
)
ax2.set_ylim(-20, 25)
ax2.locator_params(tight=True, nbins=4)
ax2.set_xlabel('timestamp')
ax2.set_ylabel(r'wind power')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig(folder_path + 'prediction_results.pdf')  # Save the second plot
plt.close(fig2)


