import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# ========== 第一步：解析数据 ==========
with open('data/isoflops_curves.json', 'r') as f:
    data = json.load(f)

grouped_data = defaultdict(list)
for item in data:
    grouped_data[item['compute_budget']].append(item)

# 按 final_loss 排序
for key in grouped_data:
    grouped_data[key].sort(key=lambda x: x['final_loss'])

# 取每个 C 下 loss 最小的点
optimal_points = []
for C in sorted(grouped_data.keys()):
    best = grouped_data[C][0]
    N_opt = best['parameters']
    D_opt = C / (6 * N_opt)
    optimal_points.append((C, N_opt, D_opt, best['final_loss']))

print("Optimal points (C, N_opt, D_opt, loss):")
for p in optimal_points:
    print(f"  C={p[0]:.0e}: N={p[1]:.2e}, D={p[2]:.2e}, loss={p[3]:.4f}")


# ========== 第二步：拟合幂律（纯 numpy，无需 scipy） ==========
C = np.array([p[0] for p in optimal_points])
N = np.array([p[1] for p in optimal_points])
D = np.array([p[2] for p in optimal_points])

# 核心：在 log-space 做线性回归
# log10(N) = b * log10(C) + log10(a)
# 等价于 y = slope * x + intercept，用 numpy.polyfit 拟合

log_C = np.log10(C)
log_N = np.log10(N)
log_D = np.log10(D)

# polyfit(degree=1) 返回 [slope, intercept]
# y = slope * x + intercept
coeffs_N = np.polyfit(log_C, log_N, 1)
coeffs_D = np.polyfit(log_C, log_D, 1)

b_N = coeffs_N[0]          # 指数
a_N = 10 ** coeffs_N[1]    # 系数
b_D = coeffs_D[0]
a_D = 10 ** coeffs_D[1]

print(f"\n拟合结果：")
print(f"  N_opt(C) = {a_N:.4e} * C^{b_N:.4f}")
print(f"  D_opt(C) = {a_D:.4e} * C^{b_D:.4f}")


# ========== 第三步：外推预测 ==========
for C_target in [1e23, 1e24]:
    N_pred = a_N * (C_target ** b_N)
    D_pred = a_D * (C_target ** b_D)
    print(f"\nC = {C_target:.0e}:")
    print(f"  N_opt = {N_pred:.2e} ({N_pred/1e9:.1f}B params)")
    print(f"  D_opt = {D_pred:.2e} ({D_pred/1e9:.1f}B tokens)")
    print(f"  Check: 6*N*D = {6*N_pred*D_pred:.2e}")


# ========== 第四步：绘图 ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# (a) Model size
C_plot = np.logspace(17, 25, 500)
N_plot = a_N * (C_plot ** b_N)
ax1.scatter(C, N, color='red', s=100, label='Optimal points', zorder=5)
ax1.plot(C_plot, N_plot, 'b-', linewidth=2, label=f'Fit: N={a_N:.2e}×C^{b_N:.3f}')
for C_t in [1e23, 1e24]:
    N_t = a_N * (C_t ** b_N)
    ax1.scatter([C_t], [N_t], color='green', s=150, marker='*', zorder=6)
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('C (FLOPs)'); ax1.set_ylabel('N_opt (params)')
ax1.set_title('(a) Model Size Scaling Law'); ax1.legend(); ax1.grid(True, alpha=0.3)

# (b) Dataset size
D_plot = a_D * (C_plot ** b_D)
ax2.scatter(C, D, color='red', s=100, label='Optimal points', zorder=5)
ax2.plot(C_plot, D_plot, 'b-', linewidth=2, label=f'Fit: D={a_D:.2e}×C^{b_D:.3f}')
for C_t in [1e23, 1e24]:
    D_t = a_D * (C_t ** b_D)
    ax2.scatter([C_t], [D_t], color='green', s=150, marker='*', zorder=6)
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('C (FLOPs)'); ax2.set_ylabel('D_opt (tokens)')
ax2.set_title('(b) Dataset Size Scaling Law'); ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('isoflops_scaling.png', dpi=300)
plt.show()