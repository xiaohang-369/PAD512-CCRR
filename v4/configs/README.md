# 配置独立于代码

用户要求：训练参数放configs供他人修改，GPU与卡数不写死。配置文件版本化，运行保存展开后的有效配置及原文件哈希。

| 文件 | 用途 | 可执行状态 |
|---|---|---|
| phase1-train.json | 原E/D、90＋30、完整损失、512×512、FP32、全局batch16 | 当前可执行的第一阶段训练配置 |
| phase1-resources.json | 只读PAD入口、已核验VGG、独立缓存/输出目录 | 当前第一阶段资源配置 |
| verify-cpu.json | 小尺寸CPU、来源/数字链/梯度工程验证 | 可执行 |
| verify-server.json | 每张所选卡batch16、512，无optimizer | 可执行；须显式选择GPU |
| probe-residual.json | 差值加密、基底恢复、PNG收发及容量拒绝诊断 | 可执行；纯CPU，无训练 |
| workflow-v4.json | 用户明确的四阶段顺序、统一符号和阶段产物 | 设计配置；当前从第一阶段开始 |
| resources.example.json | 已训练监督器、A0、数据资源的只读入口 | 历史路径示例；服务器校验后才确认 |
| train-author-reference.json | 抄录固定作者源码90+30与损失/Adam设置 | 文献代码参考，不能当PAD训练配置执行 |
| train-pad-design.json | 历史第一阶段准备与后续设计记录 | 不执行；当前用phase1-train.json |

可复制配置到服务器独立V4目录后编辑，不需要改Python源码。`--config`指定文件；`--resources`替换资源入口。

设备选择：`--gpus 2`或`--gpus 0,2`只是命令示例，不指定用户应使用哪张卡。启动器设置本进程CUDA_VISIBLE_DEVICES，验证器使用对应的`cuda:0`,`cuda:1`等局部编号。JSON里的cuda:0不是物理卡0。

旧验证命令多个设备逐卡验证；新第一阶段训练按全局batch定义，`global_batch = world_size × per_device_batch × accumulation`；默认不做梯度累积。卡数变化不自动倍增全局batch或学习率，不可整除时明确报错。第一阶段训练器现已实现DDP；配置runtime.gpu_ids选择设备，CPU Gloo全局TV梯度检查通过，服务器GPU/NCCL仍需preflight实测。旧run-verify入口仍为逐卡检查。

独立修改范围：模型/损失（方法配置）、数据与监督器（私有资源配置）、优化器/日程/保存（实验配置）、设备/输出路径（运行配置）。体系结构变更必须新模型版本，不把任意宽度/深度开关伪装成同一作者基线。

当前执行请见 `../PHASE1_SERVER.md`。历史train-pad-design保留但不执行；原参考文件不是训练CLI配置。
