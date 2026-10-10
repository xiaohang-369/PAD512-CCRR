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

## 工程接口

`src/ccrr/` 按 data、teachers、representation、codec、communication、runtime、training、calibration、evaluation 分工。保留全部 A–J 实现；`scripts/run_registered_matrix.py` 默认只生成四组预登记配置，显式 `--run` 才启动训练。

`ccrr export`、`ccrr send`、`ccrr receive` 提供冻结部署与收发入口，参数见各子命令 `--help`。发送前必须具备通过相应验收的冻结系统；只分享输出的 `public.png`，C/M 凭证独立交付。接收端不使用原图、连续 latent 或发送端预算。nonce 数据库不得回滚后复用旧密钥，丢失状态须创建新对象与密钥。

请勿提交患者记录、逐图预测/配对日志、数据、权重、密钥、nonce 数据库、checkpoint 或实验结果。研究审查文档和历史验证日志仅在本地保留，不是安装、测试或部署依赖。
