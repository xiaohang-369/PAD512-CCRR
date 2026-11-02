# V4 稳定执行链

当前入口和完成状态以 [根 README](../README.md) 与 [稳定版本说明](../governance/V4_STABLE.md) 为准。旧版“尚未训练”等时点文字原样保存于 [history/V4_README_20261021.md](../history/V4_README_20261021.md)，不覆盖已完成120/30/60轮的回传事实。

本分支保留原 E0/D0、E1/D1 联合训练、冻结基线评价、历史KL身份损失及权重加载/恢复逻辑。旧 RESIDUAL_PLAN、DESIGN、INTERFACE_DESIGN、workflow-v4.json、residual_probe/transport 等只作历史资料或显式原型工具，不作为当前执行调度；本次不修改其历史实现，也不自动执行。

新增检查点收集器仅读取旧报告和权重文件哈希，导出的 residual 字段为旧表格空占位，不做残差推理或选点。排序损失实验不进入本分支。
