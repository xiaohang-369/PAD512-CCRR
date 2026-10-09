# PAD512-CCRR

医学图像能力约束表示的首版研究系统，依据 `PAD512-CCRR-v1.1`。工程同时包含模型、教师与 Purifier、私有码流、独立凭证、神经读写、训练、标定和评价；统一入口是 `ccrr train --stage all`。

**恢复契约：RGB 高保真近似恢复；认证成功后，选定量化等级的整数 latent 与 PAD-M1 字段精确恢复。** 代码和数值检查不能代替训练结果。此目录没有训练后的模型，也没有任何临床有效性、身份隐私或神经通信成功率的证明。

## 当前：第一批代码正确性修复

在重构提交 `52665d5` 上建立独立分支 `fix/pad512-v1.1-correctness-batch1`，按最终稿修复 D01、D03、D04、D11、D12。最新完整工程验证为 **133 passed，0 failed，0 skipped，0 xfailed**（使用本机已有且核验摘要的原生编码参照）。原三个 strict xfail 已转为必须通过的回归测试。

详见 [第一批代码正确性修复报告](docs/PAD512-CCRR-v1.1_第一批代码正确性修复报告.md) 和 `verification/correctness_batch1/`。D11 有意删除 D_id/A_sem 多余的末端卷积，旧错误结构的 checkpoint 不兼容；未修改 checkpoint 容器格式。其余审查问题尚未解决，工程通过不等于教师能力、完整训练或科学指标通过。

复现本轮工程检查：`python scripts/verify_refactor.py --out verification/correctness_batch1`。原生参照为可选项，不提供时该互操作测试应跳过；不会自动下载。历史函数/参数等价性检查针对结构重构，本轮有意修正数学与网络行为，不能继续声称全部等价。

## 历史：架构审查与验证

本轮按最终设计稿进行了受控结构重构，工作分支为 `refactor/pad512-v1.1-architecture`，原始源码保存在 `baseline/original`（`b2e8187`）。原目录没有 Git，已先建立归档和基线再重构。

- [工程架构报告](docs/01_工程架构报告.md)
- [设计—代码追踪表](docs/02_设计代码追踪表.md)
- [完整验证报告与设计冲突](docs/03_完整验证报告.md)
- [H100 后续训练准备清单](docs/04_H100后续训练准备清单.md)

本次结果：**41 passed，3 strict xfailed**（启用已有 CompressAI 原生参照）。三个预期失败是既有设计冲突：occupancy 空洞、有效 batch 加权 CE、JS 风险量纲，不能计作验收通过。不配置可选原生参照时为 40 passed、1 skipped、3 xfailed。

真实 PAD 五角色 2,264 图的 512 预处理像素摘要及全部 metadata 往返已重新核验。512 非平凡 INN 正逆、参数/输入梯度、消息读写梯度、底层 checkpoint 下一次优化器更新一致性、包/CLI 和阶段分派检查通过。**阶段分派检查不是 A–J 完整训练。**

512 smoke 使用合成零 latent；正确记录 oracle 的数字协议成功，真实未训练盲 Decoder 的 C/M 均返回 `Extraction Failure`。没有真实教师权重加载、完整医学训练、H100 验证或科学指标达标证据。

除三个可执行反例外，还发现选点规则、标定背景/闭环、阶段恢复和最终统计等缺项，已在验证报告 D01–D24 登记。本轮保留科研行为，未改最终设计稿。**目前不能认为下载权重、填路径后就可直接开始符合最终稿的正式训练。**

本次证据在 `verification/refactor/`；原 `verification/implementation_report.json` 是保留的历史报告。新的复现入口是 `python scripts/verify_refactor.py`；它不下载模型、不启动正式训练。

## 安装与训练入口（先解决审查报告中的阻塞）

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

| 路径（相对 src/ccrr） | 职责 |
|---|---|
| `cli.py` | 参数解析、命令分派和错误出口 |
| `data/` | PAD 预处理、五角色、患者配对、外部数据契约 |
| `teachers/` | Clinical/MFM/ReID 与终评模型适配、严格权重加载 |
| `representation/` | 三尺度 INN、Purifier、来源 mask 与局部监督 |
| `codec/` | 统一量化、整数 CDF/rANS、PAD-M1、独立 AEAD/nonce/凭证 |
| `communication/` | Carrier/两类 heads、Haar Writer、盲 Decoder、记录/路由 |
| `runtime/` | send/receive/export 业务、真实 PNG、preflight、checkpoint 序列化、工程 smoke |
| `training/` | Experiment 资源上下文、显式 A–J runner、阶段优化/损失、独立读出和攻击者训练 |
| `calibration/` | 风险、probe/标签、阈值/预算/排序、G 工作流 |
| `evaluation/` | 实际链路/开发选点/冻结攻击评分、指标、失败分母、患者统计 |
| `tensor_ops.py`, `config.py`, `errors.py` | 共用张量与神经基础块、冻结配置、失败语义 |

旧内部 Python import 路径已迁移，没有转发壳；CLI 参数和 checkpoint/state_dict 格式保持。训练、测试、发送端共用同一套量化、记录、编码与接收机制。

LPIPS、FID 和 CLIP-IQA 不参与选型或临床准入。CLIP-IQA 需要提供固定版本官方实现的适配文件和权重；当前未运行这些探索性指标。跨分科数据仍需真实数据许可、标签映射、患者标识与新 metadata profile 核验，不能仅凭接口存在就声称跨分科实验完成。

## 复核

```bash
python -m pytest -q
python -m ccrr.cli smoke --size 512 --out /controlled/engineering-smoke
python -m ccrr.cli audit-data --root /controlled/PAD-UFES-20 --out /controlled/data-check.json
```

工程测试中的合成数据、随机初始化教师和直接提供的正确记录都明确隔离。只有从实际保存/重读的 Gpub 盲提取、认证、解码得到的结果才进入正式授权恢复指标。容量拒绝仍在全输入分母；已发布但认证失败的图仍在公共临床/隐私评价分母。
