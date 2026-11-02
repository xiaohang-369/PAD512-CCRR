# PAD512-CCRR · V4 stable

本分支整理已完成的原方法 E0/D0 与医学/身份监督 E1/D1 训练链，基于历史服务器版本 `d398fc11ff3044f352c8c9647152013e4552d48f`。本次为 Git 历史治理，不修改网络、损失、训练/评价或权重加载逻辑，不启动实验。main 的迁移需另行确认。

## 当前代码主线

- 第一阶段：RGB X → E0 → 单通道 G0 → D0 → RGB；512×512，global batch16，原方法90+30=120 epoch。
- 第二阶段：从 epoch120 复制 E1/D1，联合训练；冻结 PanDerm+8500 与公共 ResNet50，保留原恢复损失和医学/身份监督。历史30轮结束后另行续至60轮。
- `v4/ccrr_v4/original.py`、`losses.py`、`phase1/`、`phase2_g1.py`、`phase2_objectives.py`、`phase2_resume.py` 是稳定执行链。可训练 E1 直接接收 RGB；不是把 E0 的单通道输出再喂给 E1。
- 新 Rank-aware Identity Loss 只在独立实验分支 `experiment/e1d1-rankloss-20261102`；不包含在本稳定分支。

完整结构、资源和历史对应关系见 [V4 稳定版本说明](governance/V4_STABLE.md)。本仓库不分发权重、原始医学图像或私人资源文件。

## 入口与回归

在既有 Python3.11 / PyTorch2.6.0 环境中，先设置 `PYTHONPATH=v4:v3:v2:src`。V4 模块未被根 pyproject 的 src 包发现逻辑单独安装，不能只依靠 `pip install -e .` 找到 ccrr_v4。

- 原方法：`v4/scripts/run-phase1.py`（prepare / preflight / train；本次未执行）。
- 冻结基线推理/医学与检索评价：`v4/scripts/run-phase2-baseline.py`。
- E1/D1 联合训练与续训：`v4/scripts/run-phase2-g1.py`；历史配置位于 `v4/configs/phase2-g1.json` 和 `phase2-g1-resume60.json`。
- 只读检查点证据收集：`v4/scripts/collect-checkpoint-selection.py`，仅核对/导出报告和哈希，不加载张量、不推理残差、不启动下一阶段。
- V4 回归：`python -m pytest v4/tests -q`。完整兼容检查分别以独立 Python 进程运行 `tests`、`v2/tests`、`v3/tests`、`v4/tests` 四个目录；旧目录有同名测试和裸模块 fixture 导入，不合并到同一 pytest 进程。

不要直接把配置中的历史服务器路径当作本机已存在的资产；先核对路径与 SHA。训练入口要求干净固定 release，配置副本和新输出目录应放在 release 外部。读取既有权重不代表重新训练。

## 历史边界

`src/`、`v2/`、`v3/` 和 V4 既有残差/传输原型保留其历史实现与来源；其中部分模块仍被 V4 用作数据、教师或监督器依赖。旧残差方案、Zm/Km、多载荷、旧 Stage A/B 不是稳定默认执行主线，不能根据历史 README 自动启动。原 README 原样保存在 [history/](history/README-before-v4-stable.md)，旧 V4 时点说明见 [V4_README_20261021.md](history/V4_README_20261021.md)。

256-bit 水印属于后续独立模块方向，本稳定版本不声称已实现或验收该模块；新身份排序损失也未替换历史 KL。

Git 治理结果见 [63个待处理文件的处置](governance/PENDING_V4_FILES.md)、[证据来源](governance/PROVENANCE.json)、[实际验证结果与旧测试限制](governance/VALIDATION.md)、[main 安全迁移计划](governance/MAIN_MIGRATION_PLAN.md)。保留作者许可：[Invertible Grayscale LICENSE](v4/upstream/InvertibleGrayscale/LICENSE.txt)。
