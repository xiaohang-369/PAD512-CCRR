# V2：医学特征与身份分离的独立验证代码

本目录是新的实验入口，分支名 **V2**。旧 `src/ccrr/training`、旧 INN/Purifier、旧运行输出均不修改、不续训。第一轮保留公开 PanDerm / ImageNet 基础权重，重新训练疾病探针、患者 ReID、S0，以及独立评价读出器和攻击者。

当前范围是**疾病任务与身份关联的前置开发实验**。这里尚无病灶/概念真值、P/E_C/D、Writer 或完整系统验收；训练完成不能据此宣布病灶保护、特征完全匿名或论文系统成功。

## 1. 代码隔离与复用边界

| 路径 | 职责 |
|---|---|
| `v2/ccrr_v2/models.py` | 新 S0、临床探针、全局/空间患者配对攻击者 |
| `v2/ccrr_v2/features.py` | 冻结 PanDerm 的真实 RGB/灰度特征缓存；fit 标准化 |
| `v2/ccrr_v2/learning.py` | 新探针与 S0 医学/联合对照训练循环 |
| `v2/ccrr_v2/reid.py` | 公开 ResNet-50 起点的新 ReID 训练与 P1/P2 能力评价 |
| `v2/ccrr_v2/loading.py` | ReID 按序多进程 CPU 加载、有限预取与设备传输 |
| `v2/ccrr_v2/audit.py` | 冻结表示后重新拟合独立 Linear/MLP 与全局/空间攻击者 |
| `v2/ccrr_v2/runtime.py` | 资源摘要、全新输出目录、提交绑定、运行状态 |
| `v2/run.sh` | 唯一推荐的 V2 入口；每次只执行一个显式阶段 |
| `v2/configs/first_validation.json` | 有限开发预算；不是正式科研通过配置 |

保留同一仓库的 Git 历史，因此 V2 分支仍能看到 V1 文件。V2 **不调用旧 Experiment、旧训练调度、旧 Purifier 或旧 INN**。只复用当前提交中的数据读取/坐标适配、公开模型适配器、患者配对及基础统计函数；这些代码在 V2 提交中固定，不从另一个工作目录动态导入。这样既隔离训练实现，又不凭新数据处理规则改变比较基础。

基础权重哈希固定；资源配置拒绝 `stage_a_source` 等旧实验字段。不会自动下载模型，也不会把旧教师 checkpoint 当预训练起点。患者数据、缓存和权重不得提交 Git。

## 2. 服务器目录约定

以下是**新规划路径，尚未在服务器创建或核验**：

```text
/data/zwc/zyh/PAD512-CCRR-V2/
├── source/                       # GitHub V2 的独立 clone；不从这里直接改代码训练
├── releases/<完整commit>/        # 从上述提交建立的独立只读工作 checkout
├── private/resources.json        # 数据与公开基础权重引用，权限仅本用户
├── plans/                       # 本轮配置的私有副本；修改必须另立实验
├── logs/<run-id>.log            # 标准输出；与对应运行目录一一绑定
└── runs/
    ├── A0-features/<run-id>/    # 真实空间特征、fit统计、清单及哈希
    ├── A1-probe/<run-id>/       # 新疾病探针、选择记录、RGB/灰度指标
    ├── A2-reid/<run-id>/        # 新患者ReID、各端点、独立P1/P2指标
    ├── B0-medical/<run-id>/     # 同结构S0，仅医学监督
    ├── B1-joint/<run-id>/       # S0医学+身份监督
    └── audits/<raw|medical|joint>/<run-id>/
        ├── linear/
        ├── mlp/
        ├── global/
        └── spatial/

/data/zwc/zyh/envs/ccrr_v2/       # V2私有环境
/data/zwc/zyh/data/PAD-UFES-20/   # 既有数据，只读复用，不重新划分
```

run-id 示例：`20261019T180000+0800_s20261019_<commit前12位>_r01`。实际开始时间在执行时生成，不提前把示例当运行记录。每阶段独立目录；已存在即报错，不能 `rm -rf` 后复用。GPU 按服务器实时空闲情况选择明确 UUID，不沿用旧物理卡编号，不抢占其他进程。

单独环境可以从用户已有私有 `ccrr_pad512` **复制**后核对依赖，不能在旧环境原位升级。当前依赖主版本沿用根目录 pyproject；新训练入口通过 V2 checkout 的 `PYTHONPATH` 运行，不对旧安装做 editable 覆盖。实际 CUDA/驱动/权重资源状态以服务器 preflight 回传为准。

## 3. 首轮实验预算与顺序

固定患者划分，fit 更新、val_tune 选 checkpoint、val_select 作一次开发报告。test 图像不载入；预检查可以核对 test 划分文件的完整性，这不是使用 test 性能选型。

| 阶段 | 新训练对象 | 预算与选择 |
|---|---|---|
| A0 features | 无参数更新 | 提取 fit/val_tune/val_select 两域特征；fit 两域有效 token 统计均值方差 |
| A1 probe | PanDerm 上新的六分类 Linear | 2,000更新，lr=1e-3；每次同批 RGB/灰度等权；val_tune 两域最小AUROC选点 |
| A2 reid | ImageNet ResNet-50＋新患者分类头 | 10,000更新，lr=1e-4，冻结BN统计；CE＋真实患者 batch-hard triplet（margin=.3）；RGB/灰度交替；P1三方向最小AUROC选点 |
| B0 medical | S0＋新六分类头 | 1,000医学预热＋3,000医学更新，lr=1e-4；固定终点 |
| B1 joint | 同初始化S0＋新头＋训练攻击者 | 同样4,000次医学更新；预热后额外1,000攻击更新，再2A:1S交替；身份权重1,000步升至.01 |
| 独立audit，每种表示 | 新Linear/MLP和全局/空间攻击者 | 每个临床读出2,000更新，每个攻击者5,000更新；分别在val_tune选择，val_select报告 |

batch=16 图或16对；ReID用8对真实正对构成16图。B0/B1医学样本随机流相同，攻击者随机流独立，不因多了攻击步骤而改变医学样本次序。权重/预算为首轮有限候选；单种子结果是开发证据，不是三种子正式研究。

**为什么先这样跑：** A1检验原特征疾病可读性；A2检验本数据患者关联能否学到；B0隔离降维的影响；B1检验身份目标是否带来额外收益；独立audit防止只欺骗训练头。A1失败则暂停S0解释；A2的P1失败不能把P2同病灶匹配当跨病灶患者能力。

原始/两组S0的audit必须同配置、同cache、同数据划分。逐一比较疾病AUROC/F1和P1/P2三方向最强攻击；配对区间与原门槛沿用设计规格。当前程序输出详细预测以供配对比较，**不自动产生科学通过证书**。它尚未包含ReID原图到S0的跨表示攻击、完整最强攻击联合bootstrap验收、定位和概念验证；这些证据补齐前，不把S0晋升为完整可信医学监督。第一轮仍能回答疾病/身份是否出现可重复的改善或退化。

## 4. 服务器准备（用户执行，不自动启动）

先只查询目录和GPU状态，确认不存在同名旧目录。只用正常 Git clone/fetch，不修改代理、TLS或其他用户环境。

```bash
ls -ld /data/zwc/zyh/PAD512-CCRR-V2 /data/zwc/zyh/envs/ccrr_v2
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv
```

确认新根路径后，以下准备不会训练：

```bash
set -euo pipefail
test ! -e /data/zwc/zyh/PAD512-CCRR-V2
test ! -e /data/zwc/zyh/envs/ccrr_v2
umask 077
mkdir /data/zwc/zyh/PAD512-CCRR-V2
git clone --branch V2 --single-branch https://github.com/xiaohang-369/PAD512-CCRR.git /data/zwc/zyh/PAD512-CCRR-V2/source
mkdir /data/zwc/zyh/PAD512-CCRR-V2/releases
mkdir /data/zwc/zyh/PAD512-CCRR-V2/private
mkdir /data/zwc/zyh/PAD512-CCRR-V2/plans
mkdir /data/zwc/zyh/PAD512-CCRR-V2/logs
mkdir /data/zwc/zyh/PAD512-CCRR-V2/runs
conda create --prefix /data/zwc/zyh/envs/ccrr_v2 --clone /home/zwc/anaconda3/envs/ccrr_pad512
```

复制既有私有环境只保留软件依赖，不复制旧实验checkpoint。源路径来自既有记录，执行前核实存在。目录已存在时停止核对，不覆盖或删除。

创建提交绑定的release；此后 source fetch 不改变该次实验：

```bash
V2_ROOT=/data/zwc/zyh/PAD512-CCRR-V2
V2_COMMIT=$(git -C "$V2_ROOT/source" rev-parse HEAD)
git -C "$V2_ROOT/source" worktree add --detach "$V2_ROOT/releases/$V2_COMMIT" "$V2_COMMIT"
V2_RELEASE="$V2_ROOT/releases/$V2_COMMIT"
export V2_PYTHON=/data/zwc/zyh/envs/ccrr_v2/bin/python
export PYTHONPATH="$V2_RELEASE/v2:$V2_RELEASE/src"
export PYTHONDONTWRITEBYTECODE=1
```

从旧配置只读导入**公开基础权重引用**，不是旧教师：

```bash
"$V2_PYTHON" "$V2_RELEASE/v2/tools/import_public_resources.py" \
  --old-config /data/zwc/zyh/ccrr-checks/stage-b/formal-purinit-rc1-mb16-20261015/check/candidate-config.yaml \
  --workspace-root "$V2_ROOT" \
  --output "$V2_ROOT/private/resources.json"
bash "$V2_RELEASE/v2/run.sh" preflight --resources "$V2_ROOT/private/resources.json"
```

导入器严格验证公开权重和数据清单摘要，拒绝已有同名配置。旧配置路径仍须服务器确认；若不存在，填写 `resources.example.json` 的真实公开权重路径，不能猜路径或改成旧 `clinical.pt/reid.pt`。

## 5. 逐阶段启动模板

preflight回传和GPU/批次运行检查通过后，按单个阶段运行。**以下模板不建议整段粘贴一次性启动；先审查每阶段结果。** 选择的GPU UUID必须来自当前服务器，不提供臆造UUID。

```bash
export CUDA_VISIBLE_DEVICES=实际选定的GPU_UUID
V2_RUN_ID="$(date +%Y%m%dT%H%M%S%z)_s20261019_${V2_COMMIT:0:12}_r01"
V2_FEATURES="$V2_ROOT/runs/A0-features/$V2_RUN_ID"
bash "$V2_RELEASE/v2/run.sh" features --resources "$V2_ROOT/private/resources.json" --output "$V2_FEATURES"
```

缓存约数GB，加载同时占用显存/内存；实际峰值与耗时尚未在H100测量。默认FP32，不临时换精度来宣称通过。数据预处理逐图核对已绑定像素哈希。

独立执行疾病探针或ReID：

```bash
bash "$V2_RELEASE/v2/run.sh" probe --resources "$V2_ROOT/private/resources.json" --features "$V2_FEATURES" --output "$V2_ROOT/runs/A1-probe/$V2_RUN_ID"
bash "$V2_RELEASE/v2/run.sh" reid --resources "$V2_ROOT/private/resources.json" --output "$V2_ROOT/runs/A2-reid/$V2_RUN_ID"
```

进入S0对照前检查上游结果；两组可顺序执行，各自独立目录：

```bash
bash "$V2_RELEASE/v2/run.sh" s0 --variant medical --resources "$V2_ROOT/private/resources.json" --features "$V2_FEATURES" --output "$V2_ROOT/runs/B0-medical/$V2_RUN_ID"
bash "$V2_RELEASE/v2/run.sh" s0 --variant joint --resources "$V2_ROOT/private/resources.json" --features "$V2_FEATURES" --output "$V2_ROOT/runs/B1-joint/$V2_RUN_ID"
```

冻结表示后分别审计；不能拿联合训练的医学头成绩作成功依据：

```bash
bash "$V2_RELEASE/v2/run.sh" audit --resources "$V2_ROOT/private/resources.json" --features "$V2_FEATURES" --output "$V2_ROOT/runs/audits/raw/$V2_RUN_ID"
bash "$V2_RELEASE/v2/run.sh" audit --resources "$V2_ROOT/private/resources.json" --features "$V2_FEATURES" --s0 "$V2_ROOT/runs/B0-medical/$V2_RUN_ID" --output "$V2_ROOT/runs/audits/medical/$V2_RUN_ID"
bash "$V2_RELEASE/v2/run.sh" audit --resources "$V2_ROOT/private/resources.json" --features "$V2_FEATURES" --s0 "$V2_ROOT/runs/B1-joint/$V2_RUN_ID" --output "$V2_ROOT/runs/audits/joint/$V2_RUN_ID"
```

各命令可通过 `set -o pipefail` 后接 `2>&1 | tee "$V2_ROOT/logs/<唯一阶段名>-<run-id>.log"` 保存终端日志；日志路径也须全新。首版不提供断点恢复命令，模型checkpoint不等于含优化器/RNG的完整恢复状态；中断时保留失败目录、报告已完成预算，不从模型快照悄悄续训。

## 6. 结果与下一步

### ReID 并行加载修订（2026-10-19）

服务器实际 V2 解释器为 `/home/zwc/anaconda3/envs/ccrr_v2/bin/python`。上文早期 `/data/zwc/zyh/envs/ccrr_v2` 是规划路径，已建立环境的服务器不应重新执行环境创建模板。

并行加载只对新版本的 `reid` 训练启用：在现有命令后添加 `--loader-workers 4 --prefetch-batches 2`。默认 workers=0 保留串行对照。每个工作进程最多预取2批（4个进程共8批），采用spawn，仅执行CPU读取、原预处理及原哈希验证；主进程生成配对并按序接收。工作进程torch线程数为1。CUDA运行启用pinned memory及非阻塞传输；没有新增CUDA预取流，不宣称H2D与GPU计算已实现完全重叠。

固定batch=16、原完整PairSampler随机调用、正对顺序、标签、灰度奇偶步、优化器与损失不变。DataLoader使用独立torch Generator，避免工作进程初始化改变模型随机状态。不删除未使用负对的随机调用，不做图像缓存、AMP或多GPU；这次仅改变加载调度。验证集评价仍使用原串行路径。预取可能提前推进采样器，当前仍不支持精确断点恢复。

运行binding.json记录worker数量、预取设置、顺序和spawn方式。新增training.jsonl记录每100步CE/triplet、窗口吞吐及主进程等待批次时间；第一窗口包含worker启动成本，评价与checkpoint写出不计入训练窗口。批次等待时间可能与上一批GPU工作重叠，不能将wall减去wait当作GPU计算时间。validation.jsonl及终端保存每1000步P1三个方向的val_tune AUROC，端点选择规则不变。

先在服务器新固定提交下运行有限CPU数据一致性检查（独立新输出目录）：

```bash
export V2_PYTHON=/home/zwc/anaconda3/envs/ccrr_v2/bin/python
export PYTHONPATH="$V2_RELEASE/v2:$V2_RELEASE/src"
export PYTHONDONTWRITEBYTECODE=1
CUDA_VISIBLE_DEVICES="" "$V2_PYTHON" "$V2_RELEASE/v2/tools/check_reid_loader.py" \
  --resources "$V2_ROOT/private/resources.json" \
  --output "$V2_ROOT/runs/loader-check/<全新run-id>" \
  --workers 4 --prefetch-batches 2 --batches 16
```

该检查使用相同fit采样序列，对串行和并行批次的图像、mask、标签、坐标及辅助字段逐批做内容摘要比较，错误直接失败；不加载模型、不更新参数、不读取test。CPU等待时间受首次启动、缓存和校验工作影响，不是训练提速倍数。检查成功后仍须另立新运行确认H100实测吞吐及数值，不覆盖正在运行的833fe4a快照，不把旧模型checkpoint当完整续训状态。软件依赖不变，无需在旧环境安装包。

每次产生 `binding.json`、`status.json` 及相应模型/指标。`completed`只表示该程序完成预算；`scientific_acceptance=not_evaluated`不能解读为通过或失败。报告含患者关联明细，留在私有服务器；回传仅必要汇总，不上传GitHub。

本轮先获取实际教师/探针和S0对照证据。未补齐区域标注、概念迁移、独立跨表示攻击和联合准入前，不自动启动灰度生成或Writer训练。论文故事保持，失败结果照实报告，不能通过修改数据划分/门槛获得通过。

## 7. 本地工程验证（2026-10-19）

18项CPU合成检查通过，包含S0更新、冻结攻击者的输入梯度、独立攻击训练、ReID嵌入接口、患者配对权重、输出防覆盖及旧checkpoint配置拒绝；Ruff、shell语法和CLI帮助检查通过。没有加载真实医学图像或公开模型权重，没有启动服务器训练。

本地临时环境为Python 3.11、PyTorch 2.6.0；macOS的SciPy 1.15.3二进制加载失败，因此仅本地临时环境改为SciPy 1.16.3完成测试。未修改根项目依赖、服务器环境或旧环境。不能用这次CPU检查代替服务器固定依赖版本、真实预训练权重、batch=16和CUDA的运行核验。
