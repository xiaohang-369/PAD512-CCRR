# V3：ICCV 式 PAD 前置监督开发实现

本目录从 V2 `21f941524736db74a16730fcb9a5d0517a221234` 独立新增，对应 V2-R4 设计。**仅支持有限开发实验，不自动颁发科研通过证书，不启动后续灰度或Writer训练。** 示例参数仍为草案，程序拒绝将未登记配置用于训练。

## 1. 本次实现范围

| 组件 | 实现 |
|---|---|
| `models.py` | 768→256→32编码、3残差块、硬整数STE、hard Gumbel互补Mask、共享特征解码、关联解码、共享三分量GMM |
| `pairs.py` | 仅P2真实同病灶不同照片；无P1资格依赖；跨患者负例、部分疾病／部位匹配负例与fallback计数 |
| `data.py` | 原A0绑定缓存只读复用、fit标准化、图像／标签／患者隔离验证；不读test图像 |
| `reid.py` | 公开ResNet50候选；可选新的P2适配；P2选点、原始初始化对照、冻结两域关联目标 |
| `learning.py` | 独立仅压缩／分离／额外对抗三组；完整编码预热、有限终点、冻结或联合任务头登记对照 |
| `evaluation.py` | 独立Linear/MLP与global/spatial配对攻击；P2 Rank-1、AP、pair-AUROC，固定图库查询患者区间 |
| `comparison.py` | 同队列／图库、同攻击预算的配对差值；无事后准入阈值 |
| `p3.py` | 普通灰度到确切原图的NCC来源匹配基线，仅val_tune，无训练 |
| `FrozenSupervisor` | 原始PanDerm特征→绑定fit标准化→确定硬分离→域内医学头；冻结参数但保留输入梯度 |
| `runtime.py` / `cli.py` | 独立V3输出、父产物摘要、一次一个显式阶段、无隐式续训或覆盖 |

残差块为Conv3/ReLU/Conv3加残差，无BN/dropout，是本适配的明确选择。内部取整为floor(x+0.5)，负半整数亦适用。推理Mask为二分类argmax，平局选semantic；生成训练可用软代理反向但保持硬、确定前向。共享GMM按通道参数、空间共享；完整Q与稠密Q_link全部有效位置均计码长，包括置零符号，分母不随Mask占用变化。

核心损失为完整特征MSE、单位化关联embedding距离、内部码长代理、疾病CE；完整特征重建不施加于semantic分支。冻结医学头和联合任务头不是隐式切换。额外对抗使用平衡BCE与均匀混淆，不叠加第二次GRL反号。

ICCV分离与原文扩散联合训练不同；本实现的显式医学目标只对应指定六分类。不是官方DCM-DeID完整复现，也不是MaMI复现。参考：[ICCV论文](https://openaccess.thecvf.com/content/ICCV2025/html/Tian_Semantic_versus_Identity_A_Divide-and-Conquer_Approach_towards_Adjustable_Medical_Image_ICCV_2025_paper.html)。

## 2. 不改变的边界

- 不修改旧`v2/`、`src/`、旧checkpoint或A0缓存；只复用冻结的数据适配、公开权重加载及基础统计工具。
- 医学采样覆盖合格fit全体；P2正对资格只影响关联采样。可选P2 ReID的CE仅在P2正对资格患者上训练，输出覆盖范围，不把单图患者伪造为配对。
- P2图库排除查询exact source，负例不使用同患者不同病灶。P3图库明确含exact source。P1不参与新选点或必需准入。
- Q、Q_sem、Q_link、Mask不传输；内部GMM不是q_C实际字节数。后续q_C固定4/8-bit时lambda_R=0。
- 语义表示低关联不表示公共图安全；后续仍要直接公共图P2/P3训练攻击、独立评价和双密钥验收。
- S0仅历史对照；真实PAD病灶／概念、最终P/E_C/D、Writer和完整系统没有在本实现中验证。

## 3. 目录、资源与执行登记

建议服务器新根`PAD512-CCRR-V3`，保留`source/`、`releases/<commit>/`、`private/`、`plans/`、`runs/<stage>/<unique-id>/`和`logs/`布局。资源配置强制V3根名、与数据及源码隔离；不会创建环境、安装包、下载权重或操作其他进程。解释器由`V3_PYTHON`显式指定，可先核验现有私有环境的兼容性，不自动升级V2环境。

`configs/resources.example.json`只引用公开PanDerm/ImageNet权重和原数据；按真实路径填写私有副本。`--features`可指向既有V2的已完成A0缓存，只读核验内容和绑定，不沿用旧训练教师。新输出不得位于V2根。

`configs/development.example.json`中步数、学习率、权重、Gumbel温度是**工程候选示例，不是已证明充分的训练预算**。运行前登记有限预算、比较范围及假设，将私有配置的`registration_status`设为`registered`并填写`registration_note`。程序仅支持development，不自动判医学／隐私合格。不要在代码中把样例改成默认已登记。

## 4. 显式阶段（不是一次性自动运行脚本）

所有命令均以`bash v3/run.sh <stage> --resources <private.json> --config <registered.json>`为共同前缀，真实运行前要求源码已提交、输出全新。每次仅一个阶段，无自动串联。

| stage | 额外参数 | 产物／作用 |
|---|---|---|
| `preflight` | 可选`--features` | CPU只读资源及数据检查；草案配置允许检查，不训练 |
| `probe` | `--features --output` | 分别拟合RGB／灰度医学头，val_tune两域最小AUROC选点，val_select开发报告 |
| `reid` | `--output`，可选`--workers 4` | 可选重新训练P2参照；从公开权重起步，三方向最低P2 AUROC选点，记录初始化对照；不自动优于初始化 |
| `targets` | `--features --output`，可选`--reference <V3-reid-run>` | 公共初始化或本次V3参照的冻结embedding和P2能力报告，**不自动合格** |
| `p3-baseline` | `--output` | val_tune普通灰度NCC精确来源基线，query与图库均使用公开相同灰度处理 |
| `train` | `--features --heads --variant compression --output` | 独立训练完整编码＋医学对照，不消费关联targets |
| `train` | `--features --heads --targets --reference-review --variant separation/adversarial --output` | 登记后的分离或额外对抗开发实验 |
| `audit` | `--features --view raw --output` | 原始表征独立读出与P2攻击 |
| `audit` | `--features --separator --view full/semantic/q_sem/q_link --output` | 冻结表示后重新拟合；compression无q_link |
| `compare` | `--before <audit> --after <audit> --output` | 匹配临床及关联差值区间，仍不自动颁发通过 |

训练所需的`reference-review`见样例：需绑定已审阅targets的artifact.json摘要，三方向都明确记录，决定为`allow_development_reference`并写出依据。它只是允许该候选进入开发比较的记录，不是正式教师通过证书；不得无视初始化／困难负例／有效性证据直接批量生成。旧V2已训练ReID不能通过`--reference`静默导入。

`train`仅加载fit/val_tune特征；`probe/audit`另用val_select开发报告。数据完整性核验可读取test角色元数据，但不读取test像素／特征或使用test性能。val_select已有开发历史，不能称新的无偏终测。

## 5. 评价解释与未闭合项

对照是独立训练的compression与separation，而不是同一模型中full与semantic的替代比较。附加对抗的作用单列。若要归因到学习Mask本身，还需实现并登记匹配随机／固定Mask对照；本版未实现该额外消融。

攻击方向仅在val_tune固定，在val_select不再反转或选点；同时报告双方向保守诊断，不能把低于0.5的AUROC视为安全。相同分数Rank-1与AP采用随机打破平局的期望，避免图像名排序制造成功。P3近1的NCC在1e-6范围合并为平局，以避免浮点误差使自身匹配虚高。

pair-AUROC按查询患者及每查询正负集合平衡；区间是**给定固定图库、已拟合攻击器时的查询患者聚类不确定性**，不包含重新生成图库或重训模型的方差。临床区间按患者聚类。缺类bootstrap报告无效比例；区间不足不得当通过。多seed、困难负例同查询子集报告、完整最强攻击联合准入、正式阈值仍需后续登记／完善，不能把当前compare.json当最终科学验收。

P3当前仅普通灰度NCC基线，未实现生成图的P3训练或局部特征攻击全集；其高匹配率不是不可行性证明。监督器灰度接口虽有输入梯度测试，真实PanDerm—生成像素链路尚待服务器检查。未提供任何“自动晋升合格监督包”功能。

## 6. 记录及工程验证

每运行写`binding.json`、`events.jsonl`、`status.json`及带内容SHA的`artifact.json`。运行状态completed只表示完成预算，`scientific_acceptance=not_evaluated`不变。父资源／cache／模型的内容与来源绑定，禁止路径逃逸、错误阶段或不同特征来源。输出已存在即拒绝，不删除重跑。

separator.pt保存模型、头、fit统计、优化器和主要RNG状态供审计；当前**不支持恢复命令**，不能仅凭这些字段宣称精确续训。P2参照保存选中模型而非完整恢复状态。模型输出、配对明细和患者标识只留私有服务器，不提交Git。

本地检查使用合成数据，可验证硬量化、互补Mask、GMM概率、输入梯度、冻结教师、三组短循环、P2／P3合法性、配对统计、hash防篡改、配置登记和输出隔离。合成更新是工程测试，不是PAD训练或科学结论。既有V2测试同时回归；实际结果以提交时实验记录为准。
