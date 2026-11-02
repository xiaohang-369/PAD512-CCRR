# E1/D1 Rank-aware Identity Loss 第一轮预检

日期：2026-11-02。独立分支 `experiment/e1d1-rankloss-20261102`，基于 V4 `d398fc1`。修改最初保存在独立 worktree 的临时分支 `experiment/e1d1-rank-preflight-20261102`，尚未提交时按用户要求从同一 HEAD 创建当前分支并保留全部内容。本目录只实现候选损失、数据审计和零更新诊断，未接入正式训练循环。没有调用优化器或写出新模型权重。V4 主线、既有权重和旧实验记录保留。

交付包的 `运行说明.md` 给出上传和执行方法；`审查与零更新结果.md` 给出源码审查、样本覆盖、负项投机分析、参数建议与当前证据边界。

## 实现

- `ccrr_v4/rank_identity.py`：草案原式 `live`、动态负项停梯度 `stop_negative`、固定 G0 负阈值 `anchor`；固定 fit 患者等权归约，支持无合法正样本的批次。
- `scripts/audit_fit.py`：fit 正样本覆盖、跨划分重叠、实际 fit 预处理哈希与完全重复检查。
- `scripts/local_zero.py`：真实 epoch120、真实 fit 图像、真实 ResNet50/VGG 的 CPU 零更新检查。
- `scripts/full_zero.py`：复用服务器原 PanDerm+8500+绑定统计/头，以及固定 RGB fit 库；逐项计算灰度/E1 梯度范数、夹角和权重扫描，不创建优化器。
- `scripts/check_assets.py`：按历史资产清单检查服务器文件存在性及 SHA256。

旧损失、原训练入口、网络和优化器源码均不修改。`full_zero.py` 可接收原联合 E1/D1 配置，但拒绝 resume；其配置中的 epochs/lr 仅保留来源记录，脚本不会执行训练。诊断默认 batch2，与正式训练 global batch16 区分；可显式做 batch16 零更新检查。

## 归约与图库约束

正样本要求同患者、同病灶、另一真实图像；同一原图视图或内容哈希相同者排除。同患者不同病灶既不是正样本，也不是负样本。所有查询和图库特征仅来自 fit，`val_tune` 只可能由原资产加载器读取其元数据与历史基线绑定，不进入本包梯度计算。

在所有 fit 图像上均匀取 batch 时，权重为 `w_i=N_fit/(N_valid_patients*n_valid_queries_of_patient_i)`，无合法正样本者为0；损失为 `sum(w_i*loss_i)/B`。这使全 fit 期望风险为患者等权，并兼容各卡有效查询数不等的 DDP 平均。本包零更新额外有效查询面板是诊断用富集样本，其损失均值不能视为完整 fit 风险的无偏估计。

损失保留草案原始 `softplus((a-b+m)/tau)` 尺度，没有额外乘 tau。若以后改为 `tau*softplus(...)`，必须同步换算 lambda，不能比较同一数值权重。

## 本地复现

在仓库根目录设置 `PYTHONPATH=src:v4:v3:v2`，使用已有 `ccrr-runtime` 运行 `python -m pytest v4/rank_preflight/tests -q`。测试覆盖导数方向、负项单独升高、归一化耦合反例、正样本掩码、空有效批次、固定患者权重/跨卡归约、有限差分、分段 VJP 及完整服务器诊断入口的微型模型替身。替身通过不代表服务器真实资产已执行。
