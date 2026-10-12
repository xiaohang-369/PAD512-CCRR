# PAD512-CCRR

PAD512-CCRR-v1.1 医学图像双能力授权研究工程，包含 A–J 阶段源码、512×512 预处理、INN 表示、Purifier、双流编码与认证、盲读写、标定和评价。

**恢复契约：RGB 高保真近似恢复；认证后的量化整数 latent 与 PAD-M1 字段精确恢复。** 工程测试通过不代表教师能力、完整训练或科学指标达标。

## 当前状态

A 阶段基线 `fb68b6b` 已通过 169 项 Mac CPU 工程测试（包含独立原生 rANS 参考），三个真实教师权重的严格加载、RGB/灰度接口与输入梯度，以及冻结五角色 2,264 图的预处理核验。没有执行正式 A 训练，H100 CUDA、显存、吞吐与恢复一致性仍须服务器实测。

本次公开内容整理新增 3 项工具回归，完整工程检查为 **172 passed，0 failed/skipped/xfail**（含已有原生 rANS 参考）；没有重新训练或把历史真实权重验证冒充本次执行结果。

A 支持完整 optimizer 边界恢复。B–J 的完整恢复、部分选点、标定闭环及最终评价仍有未闭合问题；保留源码不表示这些阶段已通过研究验收。PanDerm 预训练数据重叠仍为 unknown，不构成无泄漏证明。未训练 Decoder 的真实盲提取应报告失败，不能用正确记录 oracle 替代。

## 安装与工程检查

使用独立 Python 3.11 环境。先按设备准备项目指定的 PyTorch 2.6.0 / torchvision 0.21.0（H100 需要匹配驱动的 CUDA 构建），再由用户执行安装：

```bash
python -m pip install -e '.[verify]'
python -m pip check
python -m ccrr.cli --help
python -m ccrr.cli inspect
python scripts/verify_refactor.py --out /controlled/ccrr-checks
```

`verify` 包含 pytest 和 ruff；核心版本固定在 `pyproject.toml`。`requirements-local-macos.lock` 只记录 Mac 环境，不能当作 H100 锁文件。验证脚本不下载权重、不启动正式训练，不需要内部报告或历史分支。

统一检查包含完整 pytest、包/CLI、协议、512 INN 正逆与梯度、Writer/Decoder 梯度、checkpoint、缺资源预期拒绝和未训练 smoke。独立原生 rANS 互操作参考可用 `--reference-ans PATH --reference-sha256 SHA256` 显式提供；没有它时该一项应跳过，固定字节向量测试仍必须通过。其他失败不能当作成功。

只检查源码架构可运行 `python scripts/audit_architecture.py --out /controlled/architecture.json`；历史比较仅在显式提供存在的 `--baseline REF` 时进行。

## 资源与 A 阶段入口

数据、官方源码、权重、凭证和所有检查/训练输出放在源码目录之外。将 `configs/stage_a_h100.yaml` 复制为外部私有配置（下例 `/srv/ccrr/a.yaml`），填写真实路径；不要改变冻结划分、512 输入、预算、类别权重或阈值。`configs/pad512.yaml` 是全流程模板，空权重项会被严格拒绝。

| 角色 | 资源 |
|---|---|
| Clinical Teacher | EfficientNet-B0 `IMAGENET1K_V1` |
| MFM | PanDerm-Base `panderm_bb_data6_checkpoint-499.pth` |
| ReID Teacher | ResNet-50 `IMAGENET1K_V2` |

PanDerm 官方源码 commit 固定为 `fd7a80748ba7fc3e203fed88f909f4689d0d6f24`。模板含三权重完整摘要及源码摘要；PanDerm 权重摘要绑定已核验实物，不冒称发布方提供的完整校验值。资源由用户获取，仓库不附带权重。`fetch_official_weights.py` 是显式下载辅助，包含 J 阶段的 ConvNeXt/Swin，不是 A 的自动准备步骤。

必须提供完整冻结数据及原有核验证据（源 metadata、study_v1、protocol_audit_20261009 和图像）。`src/ccrr/data/frozen_evidence.json` 只含摘要与汇总数，是运行依赖；不得重生成摘要来掩盖数据变化。Git clone 本身不包含可运行正式训练的患者数据或划分文件。

在 H100 上检查 `nvidia-smi`，设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。逐模型运行 `scripts/check_teacher_interface.py --device cuda`，通过 `--model`、`--weights`、`--sha256`、`--out` 提供模板对应资源；PanDerm 另需 `--source`。四组 RGB/灰度和有效矩形检查均须通过。把各输出报告的真实路径及 SHA-256 填入外部配置的 `interface_reports`，不能用 Mac CPU 报告替代。

PyTorch 2.6 的 CUDA 抗锯齿双线性缩放反向不支持严格确定性。教师适配对同尺寸缩放直接返回输入；需要缩放和输入梯度时，保持原生设备前向像素，用 CPU 上同参数线性缩放的伴随计算一阶输入梯度，再传回原设备。严格确定性仍开启，`antialias=True` 和冻结 PIL 数据预处理不变。这一路径增加 CPU/GPU 传输开销，不支持二阶梯度；需在目标 H100 实测正确性和耗时。接口报告绑定 `pad.py`、`resize.py` 与计算策略，旧报告须重跑。使用 `CUDA_VISIBLE_DEVICES` 限定一张已确认可用的卡后，先运行 `CCRR_TEST_DEVICE=cuda python -m pytest tests/test_teacher_resize.py -q`，再执行真实权重接口检查。

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m pytest -q
CCRR_TEST_DEVICE=cuda python -m pytest tests/test_stage_a_resume.py tests/test_stage_a_training_loops.py -q
python -m ccrr.cli preflight --config /srv/ccrr/a.yaml --stage A
python scripts/check_stage_a_host.py --config /srv/ccrr/a.yaml --model efficientnet_b0 --out /srv/ccrr/checks/clinical-batch16.json
python scripts/check_stage_a_host.py --config /srv/ccrr/a.yaml --model resnet50 --out /srv/ccrr/checks/reid-pairs16.json
```

preflight 必须退出 0、`ready_for_stage=true` 且 `missing=[]`。资源检查只有合成前后向、没有 optimizer 更新；AdamW 状态为额外估算，不等于完整训练显存与吞吐已验。检查均通过并确认正式训练资源后，才执行：

```bash
python -m ccrr.cli train --config /srv/ccrr/a.yaml --stage A
# 同代码、资源、环境、seed 和输出目录下中断恢复：
python -m ccrr.cli train --config /srv/ccrr/a.yaml --stage A --resume
```

固定顺序为 Clinical 双域及两种域内对照 → MFM probe → ReID/P1、P2 独立评价；门禁失败停止。检查私有输出的 `acceptance.json`、`validation/`、`teacher_freeze_record.json` 和 checkpoint，不能仅凭生成权重判定能力达标。预登记 seed 各自使用独立输出目录，不选最好 seed。

新 checkpoint 容器为 `CCRR-CHECKPOINT-2`；旧 v1/selected-only 权重不能作为完整 A 恢复。恢复使用最近持久 optimizer 边界，默认每 100 步及规定端点保存。早期错误 Purifier 结构权重与修正后的 D_id/A_sem 不兼容。

正在运行 A 时，可在另一张已确认空闲的 H100 上用 `scripts/probe_stage_a_speed.py --config /srv/ccrr/a.yaml --out /srv/ccrr/checks/new-speed-probe` 做隔离性能回放。输出目录必须是仓库及正式输出目录之外的新目录；内含私有 checkpoint 副本，不可提交。脚本复制最近一个未完成的临床子阶段 checkpoint，分别测试原路径、2 个 CPU 线程、合并梯度有限性检查及热缓存候选；仍用生产分类循环、microbatch=1、原 FP32/确定性/采样与优化器设置。每个候选必须与原路径的 loss、最终参数/梯度、AdamW 状态及 RNG 精确一致。各自最多回放 10 步，更新只发生在隔离副本，不计入正式训练。热缓存是预加载回放样本后的上界，计时不含加载、保存和验证，不能直接推算整个 Stage A 的加速比。该工具不部署优化、不修改运行中的进程，也不放宽源码恢复绑定；正式切换仍需实测和独立的恢复迁移验证。

可选的进程内 CPU 像素缓存通过外部配置 `pixel_cache_bytes` 启用，默认 `0` 关闭。例如 `8589934592` 是所有训练数据角色共享的 8 GiB 常驻张量上限（Python 对象和当前批次另计）。按首次访问懒加载，保留原 `PADData` 的预处理与逐图摘要校验；命中时检查源文件 stat 身份并返回独立张量副本，超限按 LRU 淘汰。它不写磁盘、不缓存增强结果、不消耗随机数，不改变训练循环、有效批量、microbatch、FP32、梯度检查或优化器。源文件必须保持不可变；stat 检查不是每次重新计算源文件内容摘要。完整实验配置和 checkpoint 绑定记录缓存容量。

**已有运行不要直接合并此源码后恢复。** A checkpoint 仍严格绑定全部生产源码及缓存设置，没有增加忽略绑定的开关。新增 `scripts/check_stage_a_cache.py` 用于从独立候选源码目录验证：指定 `--legacy-repo` 为原正式源码目录、`--config` 为原配置、`--out` 为仓库和正式输出之外的新私有目录，在另一张空闲 H100 上运行。它校验原绑定和源码差异白名单，仅在私有副本中显式翻译绑定，然后用未改动的生产分类循环比较无缓存、仅缓存、缓存保存后冷启动重载三条路径；完整比较模型参数/缓冲区、AdamW、调度器、step、selection/audit、全部 RNG，以及各视图 loss、梯度和梯度范数。回放每一步在独立目录实际保存，最后一步在 `scheduler.advance()` 和保存完成后停止。计时仍不含保存与验证；短回放通过不等于整个 Stage A 提速或正式迁移已通过。正式迁移仍须另行保护最新进度并验证，原正式输出和进程不由该脚本修改。

Stage A 另支持 `microbatch: 4` 或 `8` 的真正批量执行（实现接受 1/2/4/8/16）。Clinical 按图分组，每组分别计算 RGB/Gray；MFM probe 同样分组；ReID 按对分组，逐对保留采样域和翻转，并把逐对损失求和后除以完整的 16 对。临床加权 CE 始终使用整个有效批量及全部视图的权重和归一化，BN 继续冻结、FP32 与严格确定性继续开启。MFM 初始化统计和验证路径保持原样。模板默认仍为 1。

批量变化会改变浮点归约及 Dropout/stochastic-depth 的 RNG 消耗，不能声称与单图训练逐位一致。`microbatch` 已纳入恢复绑定；切换时使用新输出目录从官方基础权重开始，保留旧运行，不直接接续旧 checkpoint。相同配置内的完整保存/恢复仍需精确一致。可在另一张空闲 H100 使用 `scripts/check_stage_a_batch_speed.py --config /srv/ccrr/a.yaml --out /srv/ccrr/checks/new-batch-speed` 做一次 1/4/8 比较：使用真实已验收 fit 像素、CPU 热缓存、官方 EfficientNet-B0 和生产分类循环，每档重复两次并比较完整状态；为测量全骨干反向，仅在隔离测速中把调度器起点设为 750。测速不会读取或修改正式 checkpoint，所有更新丢弃，不能作为模型能力证据或全 Stage A 吞吐承诺。通过后再对新的外部配置执行正式 Preflight。

## 工程接口

`src/ccrr/` 按 data、teachers、representation、codec、communication、runtime、training、calibration、evaluation 分工。保留全部 A–J 实现；`scripts/run_registered_matrix.py` 默认只生成四组预登记配置，显式 `--run` 才启动训练。

`ccrr export`、`ccrr send`、`ccrr receive` 提供冻结部署与收发入口，参数见各子命令 `--help`。发送前必须具备通过相应验收的冻结系统；只分享输出的 `public.png`，C/M 凭证独立交付。接收端不使用原图、连续 latent 或发送端预算。nonce 数据库不得回滚后复用旧密钥，丢失状态须创建新对象与密钥。

请勿提交患者记录、逐图预测/配对日志、数据、权重、密钥、nonce 数据库、checkpoint 或实验结果。研究审查文档和历史验证日志仅在本地保留，不是安装、测试或部署依赖。
