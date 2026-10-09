# PAD512-CCRR

医学图像能力约束表示的首版研究系统，依据 `PAD512-CCRR-v1.1`。工程同时包含模型、教师与 Purifier、私有码流、独立凭证、神经读写、训练、标定和评价；统一入口是 `ccrr train --stage all`。

**恢复契约：RGB 高保真近似恢复；认证成功后，选定量化等级的整数 latent 与 PAD-M1 字段精确恢复。** 代码和数值检查不能代替训练结果。此目录没有训练后的模型，也没有任何临床有效性、身份隐私或神经通信成功率的证明。

## 当前验证状态

- 已在本机 CPU 运行 512 尺寸的可逆主干、神经写入/读取及反向梯度检查。
- 已核验真实 PAD 清洗集 2,264 张图的预处理像素摘要，以及全部 metadata 的精确序列化往返。
- 已测试独立凭证、错误密钥、并发 nonce、残缺/重复记录、畸形熵流、容量限制、冻结教师的输入梯度及训练 checkpoint 保存/恢复。
- 有界 Python rANS64 与 CompressAI 1.2.8 原生实现已完成本机合法固定向量的双向互通及字节一致检查。生产接收不调用原生实现的无边界指针解析。
- 512 工程检查使用合成低熵 latent。数字协议成功，**未训练盲解码器返回 Extraction Failure**，不能将这次检查称为图像载荷恢复成功。
- 完整医学训练、全部阶段调度、PanDerm 真实权重加载、GPU 显存与吞吐、最终统计评价尚未实跑。正式 H100 环境需要独立检查。

当前证据在项目外 `/Users/zhuyanghang/Desktop/CCRR-experiments/`；代码内 `verification/implementation_report.json` 是本次工程检查索引。

## 安装与完整运行

建议使用独立 Python 3.11 环境，数据、权重、凭证和实验输出放在源码目录之外。当前本机环境为 `/Users/zhuyanghang/Desktop/ccrr-runtime`。正式 GPU 主机先配置对应的 PyTorch/CUDA 构建，不能把本机 macOS 锁文件当作 H100 的锁文件。

```bash
python -m pip install -e '.[test]'
python -m ccrr.cli inspect --config configs/pad512.yaml
python -m ccrr.cli preflight --config configs/pad512.yaml
python -m ccrr.cli train --config configs/pad512.yaml --stage all
```

`configs/pad512.yaml` 的权重项有意为空。填入经过核验的官方权重路径、SHA-256 和 PanDerm 固定源码；缺少资源时程序明确停止，不静默使用随机教师。官方 ImageNet 权重可用以下工具获取并生成**新的**配置文件：

```bash
python scripts/fetch_official_weights.py \
  --config configs/pad512.yaml \
  --directory /controlled/ccrr-weights \
  --panderm-weights /controlled/ccrr-weights/panderm_bb_data6_checkpoint-499.pth \
  --panderm-source /controlled/panderm/modeling_finetune.py \
  --out-config /controlled/pad512-local.yaml
```

PanDerm 必须使用设计指定的 Base 来源，不能替换为用本实验 val/test 微调的权重。源文件对应仓库 commit `fd7a80748ba7fc3e203fed88f909f4689d0d6f24`。预训练样本重叠仍须单独审计。

完整命令内部执行教师 → Purifier → 表示学习 → 通信 → 标定/联合训练 → 冻结终评。阶段是完整系统的训练依赖，不是取消教师或单独交付基础协议。未通过科学准入的阶段停止并写 `stopped.json` / `acceptance.json`，不会降低门槛继续制造“成功结果”。

已有阶段产物可通过 `--stage A|B|C|D|E|F|G|H|I|J` 进入相应阶段。此入口重新开始该阶段，**不是任意中间 step 的自动恢复**；低层 checkpoint 工具保存并支持加载模型、优化器与 Torch RNG，完整阶段内断点恢复仍需在目标环境进行中断/续跑验证。

四组预登记开发配置由 `scripts/run_registered_matrix.py` 生成。默认只写配置，显式 `--run` 才执行。先完成主实验，再进行消融或跨分科扩展。

## 发送与接收

冻结模型与整数 CDF 后，`export` 生成绑定资源摘要的模型包。研究快照与正式部署 profile 分开；正式 profile 需要相应验收证据。

```bash
ccrr export --checkpoint /run/checkpoints/H.pt --cdf /run/snapshots/H-final-accepted/cdfs.npy \
  --config /run/final-config.yaml --evidence /run/deployment-evidence.json --out /controlled/bundle

ccrr send --manifest /controlled/bundle/manifest.json \
  --image /controlled/source.png --metadata /controlled/metadata.json \
  --clinical-checkpoint /run/checkpoints/clinical.pt \
  --nonce-db /controlled/sender-state/nonce.sqlite --out /controlled/publication

ccrr receive --manifest /controlled/bundle/manifest.json \
  --image /controlled/publication/public.png \
  --key-c /controlled/publication/C.key --key-m /controlled/publication/M.key \
  --out /controlled/recovery
```

`send` 成功后只分享 `public.png`。C/M 凭证经独立受控渠道交付；它们不属于公开图像文件。接收端不读取原图、V、G0、连续 latent、发送预算、频带 gate 或临床教师。省略某条凭证不会妨碍另一条能力。

Nonce 由 SQLite `WAL + synchronous=FULL + BEGIN IMMEDIATE` 预留。不要复制/回滚该存储后继续用旧密钥发送。完整数据库及所有本地证据同时回滚不能靠数据库自身识别；恢复备份或丢失存储后须新建对象和密钥。默认存储和日志均位于受控实验输出目录。

## 工程布局

| 路径 | 职责 |
|---|---|
| `src/ccrr/system.py` | 发送、实际载体接收、能力独立返回、PNG 保存/重读 |
| `models/inn.py` | 三尺度 INN、Cayley 通道混合、共享解析逆、固定灰度对照 |
| `models/communication.py` | Carrier Encoder、预算/可靠性头、Haar 软频带 Writer、盲 Decoder |
| `models/teachers.py`, `models/purifier.py`, `supervision.py` | 教师适配与严格加载、Purifier、来源 mask 和局部几何监督 |
| `entropy.py`, `rans.py`, `metadata.py`, `crypto.py`, `records.py` | 多等级整数源编码、PAD-M1、AEAD、nonce、记录与路由 |
| `training.py`, `feature_audit.py`, `calibration.py` | 完整训练调度、独立特征读出、实测 probe、阶段验收 |
| `evaluation.py`, `final_study.py`, `metrics.py` | 实际链路结果、分类与隐私攻击、失败分母、患者重采样 |
| `robustness.py`, `comparisons.py`, `joint_b.py` | 凭证矩阵、有损通道检查、配对比较及 Joint-B 回退条件 |
| `optional_metrics.py`, `external.py` | LPIPS/FID/CLIP-IQA 接口、明确标注泛化层级的外部分科契约 |

LPIPS、FID 和 CLIP-IQA 不参与选型或临床准入。CLIP-IQA 需要提供固定版本官方实现的适配文件和权重；当前未运行这些探索性指标。跨分科数据仍需真实数据许可、标签映射、患者标识与新 metadata profile 核验，不能仅凭接口存在就声称跨分科实验完成。

## 复核

```bash
python -m pytest -q
python -m ccrr.cli smoke --size 512 --out /controlled/engineering-smoke
python -m ccrr.cli audit-data --root /controlled/PAD-UFES-20 --out /controlled/data-check.json
```

工程测试中的合成数据、随机初始化教师和直接提供的正确记录都明确隔离。只有从实际保存/重读的 Gpub 盲提取、认证、解码得到的结果才进入正式授权恢复指标。容量拒绝仍在全输入分母；已发布但认证失败的图仍在公共临床/隐私评价分母。
