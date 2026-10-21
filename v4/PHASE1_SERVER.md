# V4 第一阶段服务器操作

本包只运行原 E→G0→D0 基线。作者的90＋30 epoch均属于V4第一阶段。没有身份/医学监督、加密、嵌入命令。先按下面1–4逐步检查、回传报告，再手动执行5；任何失败均保留现场，不自动换环境或改配置重跑。

## 1. 上传、解压、安装独立版本

将本次 `PAD512-CCRR-V4-phase1-<commit>.tar.gz` 和同名 `.sha256` 放到服务器自己的上传目录。以下 `<commit>` 使用包名中的7位提交号：

```bash
sha256sum -c PAD512-CCRR-V4-phase1-<commit>.tar.gz.sha256
tar -xzf PAD512-CCRR-V4-phase1-<commit>.tar.gz
cd PAD512-CCRR-V4-phase1-<commit>
bash install-code.sh
```

默认既有解释器：`/home/zwc/anaconda3/envs/ccrr_v2/bin/python`。
默认独立目录：`/data/zwc/zyh/PAD512-CCRR-V4`。只在这里建立release、VGG资源、配置、数据视图、运行结果；原数据只读，旧V2/V3不改，不安装任何依赖。使用已有Git bundle，服务器不需要访问GitHub或下载VGG。

如已有解释器/私有根目录不同，在所有命令前设置 `V4_PYTHON` / `V4_ROOT`；不要改系统环境或别人目录。出现缺依赖就回传具体报错，不在原环境盲目pip升级。

安装输出 `PHASE1_CONFIG=...` 与 `V4_RELEASE=...`。配置副本在：

```text
$V4_ROOT/configs/phase1-train-<commit>.json
$V4_ROOT/configs/phase1-resources-<commit>.json
```

## 2. 配置与资源核对

编辑上面的**配置副本**，不要编辑release中的默认文件或Python源码。

- `phase1-resources`：确认 `data_root` 对应冻结 PAD 数据；VGG路径、缓存和run目录均相对独立V4根目录，默认即可。权重已随包提供并绑定哈希。
- `phase1-train`：填写 `runtime.gpu_ids`。例如 `[2]` 为一张物理卡，`[0,2]` 为两张物理卡；只是语法示例，不代表已分配这些卡。也可用设备UUID字符串；启动器只对本进程设置CUDA_DEVICE_ORDER=PCI_BUS_ID及CUDA_VISIBLE_DEVICES。
- 全局batch固定8，支持1/2/4/8卡，对应每卡8/4/2/1，不自动放大学习率，不做梯度累积。卡数不能整除8会拒绝执行。
- `runtime.cpu_threads`、日志频率、验证batch、保存频率均在配置中。验证batch默认1。基线损失/日程完整列在JSON；本次严格源码模式拒绝将改过的损失/epoch冒称同一基线。
- 可用 `V4_CONFIG=/绝对路径/副本.json`、`V4_RESOURCES=/绝对路径/副本.json` 选择其他配置副本。

配置中的90和30就是作者两个训练段，并非用户方案中的第二阶段。当前配置运行到总120epoch自动停止。

## 3. 准备数据视图，无训练

在解压目录执行：

```bash
bash phase1.sh prepare
```

这一步在CPU校验冻结数据清单及源文件哈希；核验实际使用fit/val_tune的PAD512像素，创建独立256缓存；校验VGG。完成后输出 `PHASE1_PREPARED=.../prepared.json`。重复调用不会覆盖现有缓存；若已经成功，无需重建。若失败，请回传报错并保留目录。

默认新建：`$V4_ROOT/data/phase1-author256/`。数据适配定义见release的 `v4/PHASE1_AUDIT.md`。预期fit1433、val_tune169；不能擅自混入calibration/val_select/test扩充训练数据。

## 4. 所选单卡/多卡预检，零更新

```bash
bash phase1.sh preflight
```

按配置启动单进程或真实DDP。用全局batch8检查两段完整原损失、原E/D梯度、VGG冻结、真实8bit PNG读回；不调用optimizer.step，不保存训练checkpoint。输出唯一目录及 `PHASE1_REPORT=.../report.json`。

回传该 `report.json` 和安装/准备终端输出即可，不要回传患者图像。需看到 `passed=true`、`optimizer_updates=0`、`png_exact=true`、`vgg_frozen=true`，并确认world_size符合选卡。多卡GPU/NCCL是否可用以本步骤实测为准，本地仅完成CPU Gloo多进程检查。

## 5. 预检确认后，手动启动第一阶段完整训练

在你自己的持久终端会话中执行（命令在前台运行，终端断开可能终止进程）：

```bash
bash phase1.sh train
```

这条命令才会更新参数，执行90＋30epoch，默认21,600步。它不会进入身份/医学监督或任何后续阶段。输出在唯一的 `$V4_ROOT/runs/phase1/train.<时间>.<PID>/`，绝不覆盖旧run。

应保留：

- `binding.json`：固定代码、完整有效配置、数据缓存/VGG哈希、GPU和软件版本。
- `metrics.jsonl`：每epoch训练损失分项，连续链和8bit PNG链的val_tune恢复指标。
- `epoch-010.pt` 等检查点，以及 `epoch-090.pt`、`epoch-120.pt`。
- `checkpoints.json`、`report.json`、`status.json`。
- `preview-epoch-*.png`：固定前4个val_tune样例，五列为输入RGB、普通灰度、G0、PNG后恢复RGB、4倍绝对误差；保存在私有run内。

固定最终120epoch作为候选基线，不以val_tune选择最佳点或提前停止。完成后回传报告与曲线数据，先审核恢复和量化表现，不自动宣布医学/隐私有效。

## 中断恢复

默认不自动恢复。明确选定上次完整epoch检查点后，写入一个新的run目录：

```bash
bash phase1.sh train --resume /绝对路径/原run/epoch-090.pt
```

代码、缓存、VGG、损失日程必须匹配；可调整GPU分配但全局batch保持8。若第91epoch前恢复，第二段Adam重新开始，但LR global_step继续。文件损坏/配置变化会拒绝，不覆盖原run；已完成120epoch的checkpoint不会继续训练。

## 验证范围

本包交付的是经过本地测试的第一阶段实现和运行步骤，不是已经训练好的PAD模型。PyTorch源代码迁移不等于TF1.6运行时逐位复现。VGG使用层已与官方Caffe逐元素核对，但不可用Mega整文件字节身份未验证。详细论文/源码差异及原始行号见 `v4/PHASE1_AUDIT.md`。
