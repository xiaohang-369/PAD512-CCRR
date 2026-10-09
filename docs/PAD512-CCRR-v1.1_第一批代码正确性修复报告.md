# PAD512-CCRR-v1.1 第一批代码正确性修复报告

本轮按最终稿修复 **D01、D03、D04、D11、D12**，其余问题保持登记、未扩大修复范围。完整工程回归 **133 passed、0 failed、0 errors、0 skipped、0 xfailed**。这是本机 CPU 工程证据，不是 A–J 完整训练通过或科学指标达标。

唯一研究依据：[最终设计稿](../../医学图像双能力授权系统_完整方案_V1修订版.md)。源码定位参考此前的[设计—代码追踪表](02_设计代码追踪表.md)和[完整验证报告](03_完整验证报告.md)。没有修改最终稿、下载大型权重或启动正式训练。

## 1. 基线、范围和回滚

- 起点：重构分支 `refactor/pad512-v1.1-architecture` 的 `52665d5`；修改源码前检查工作区干净。
- 独立修复分支：`fix/pad512-v1.1-correctness-batch1`。原始 `baseline/original`（`b2e8187`）与重构分支均保留。需要对比/回滚时切换原分支或在独立 checkout 使用该提交；没有重置或覆盖它们。
- 顺序提交：D01 `6dc3c14` → D03 `ad6dcc5` → D04 `7a0ca64` → D11 `13d99cc` → D12 `5fedd6e`；后续提交仅补充测试、验证证据和文档。
- 修改前重跑：**41 passed、3 strict xfailed**，已有 CompressAI 原生参照启用。证据：[baseline.log](../verification/correctness_batch1/baseline.log)、[baseline.xml](../verification/correctness_batch1/baseline.xml)。
- 创建证据目录后保存的 `start_status.txt` 只出现该新增目录；此前首次 Git 检查无修改。原始状态文本保留，不覆盖成空文件。
- 最终稿 SHA-256 前后均为 `6319564fce9923f01d34f367e2e46ee20b9bf2dc90e0ae6a094f2863b9c5a549`。

本轮改动清单：`communication/records.py`；`training/losses.py,experiment.py,teachers.py,purifier.py,representation.py,joint.py`；`representation/supervision.py,purifier.py`；`codec/entropy.py`；`runtime/system.py`；原冲突测试、五份新回归测试；验证脚本结果说明、README 和报告。没有修改数据成员、512 输入、类别权重生成、量化步长、Router、Writer/Decoder 网络、INN、密码或消息协议、阶段预算和阈值。

## 2. 逐项根因、修复与追踪

### D01：occupancy 空洞

**设计：§22.3、§23 的独立 capability 故障处理。** 原 `logits_to_records` 对 occupancy 无效行直接 `continue`，随后有效行仍会进入记录集，等效压缩了空洞。

修复位置：[communication/records.py](../src/ccrr/communication/records.py) 的 `logits_to_records`。每个区域扫描时记住是否已经遇到空行；其后再出现有效行就登记 `Occupancy hole`，该行不进入记录集，不重新排序或补猜记录。

固定错误归属规则如下：

1. 空洞后的**有效行**按它解出的 alpha 归属 C 或 M，令该流出现结构错误；即使其他区域已提供了该流完整副本，也不能绕过错误。
2. 空行其他字段一律忽略，不据此猜测缺失流。空洞前合法前缀和其他区域合法记录仍保留。
3. 两种类型都出现在空洞后时，两流都报错；只出现一种时不自动让另一种失败。无法收齐记录的流仍由原重组检查拒绝。
4. `CCRR.receive_records` 原有流级检查将被标记流返回为 **Extraction Failure**，另一条流继续独立重组/认证。

这落实了最终稿要求固定的实现归属规则，没有新增接收端预算、发送表或密钥输入。测试覆盖所有 16 种四行模式 × 两种类型、同一区域异流合法前缀、C/M 混合空洞，以及实际 AEAD 完整对象的一流失败/另一流成功。供给解析器的 logits 是数字测试夹具，不是训练后 Decoder 的成功证据。

### D03：完整有效 batch 的类别加权 CE

**设计：§26.2；多视图 §6.6、§7.6；Joint 可行子集 §26.4。** 原 `clinical_loss` 在单图调用 PyTorch weighted mean，得到 `w_y CE / w_y = CE`，再由训练循环平均，类别权重被抵消。

修复后：令完整有效 batch 有 N 个视图样本，W=Σw_y，μw=W/N。含 m 个样本的 microbatch 返回

`Σ_micro(w_y CE)/(m μw) + 0.5 × mean_micro(KL)`。

调用者以 `m/N` 累积后，正好得到 `Σ_all(w_y CE)/W + 0.5 × mean_all(KL)`。当前正式循环仍使用规定的 microbatch=1；没有借此实现或宣称 D19 的任意配置生效。

| 路径 | 核对及处理 |
|---|---|
| `training/losses.py:weighted_cross_entropy,clinical_loss` | 前者提供唯一加权求和/显式总分母；后者接收 `effective_weight_mean`，KD 保持无权重样本均值和系数 0.5 |
| `representation/supervision.py:TeacherSupervision.loss` | 传入完整有效 batch 权重均值；局部临床项 0.1、局部身份项及冻结教师输入梯度不变 |
| `training/experiment.py:Experiment.class_weight_mean` | 从当前采样索引计算均值，重复抽样照计，不读取额外图像或改 fit 权重 |
| `training/representation.py:train_representation` D/E | 前向前固定本次采样列表及分母，再逐图累积；批量抽索引与旧逐次抽索引序列一致。C 无临床项，不增加监督 |
| `training/joint.py:_accumulate_message_gradients` H/I | 分母仅来自本次容量可行样本；空集合直接返回，原拒绝批逻辑保留 |
| `training/joint.py:_accumulate_representation_gradients` I | 分母来自全部 chosen 样本，独立于通信可行子集；不删除容量失败样本的表示项 |
| `training/teachers.py:train_classifier,train_mfm_probe` | 原已用全 batch 加权求和/分母；统一调用同一 CE 函数。Clinical RGB/灰度两视图、单域对照和 MFM 两视图分母保持正确 |
| `training/purifier.py:stage_b` | 原已用 16 图 × 3 语义视图全局分母；统一 CE 函数，RGB/gray/self/source-mask 三条任务监督仍等权。双域重建/占用/身份重建系数不改 |
| ReID Teacher 与攻击者 | 患者身份 CE 原为无类别权重的端点均值，BCE/contrastive 另有既定平衡规则；不把临床六分类权重套到身份损失 |
| `FeatureAudit.classifiers` / `FinalStudy.train_classifier` | 前者整批三视图一次 weighted mean，后者显式全批总分母；均无本问题，保持源码原样 |

数值和参数梯度测试比较单批与 `(1,1,1,1,1)`、`(2,3)`、`(4,1)`、`(5)` 划分，覆盖 1/2/3 视图、不均衡类别、KD 不回传 target、冻结监督输入梯度。另执行真实 D/E/H/I 累积函数，以轻量解析组件隔离临床项，核对实际 cohort 分母和梯度；在选点前停止，不产生任何能力准入报告。攻击者 2 次更新、detach、生成端冻结攻击者和负 BCE 的符号/系数未变；梯度隔离另有真实 Purifier/A_sem 回归测试。

### D04：JS 散度量纲

**设计：§8.5，调用于 §13、§29 的 probe。** 原实现自然对数 JS 最大约为 ln 2，没有除以 ln 2，和预登记绝对阈值量纲不同。

修复位置：[training/losses.py](../src/ccrr/training/losses.py) `js_divergence`：以 `log_softmax` 和 `logaddexp` 稳定构造混合分布，返回 `[KL(P||M)+KL(Q||M)]/(2 ln 2)`。零概率项明确按零处理；最终 `[0,1]` clamp 只处理浮点舍入偏差，不用于放宽 probe 通过门槛。

`calibration/probes.py` 的 `clinical_risk` 与局部 probe 已共同引用此函数，未增加另一个版本。**0.001/0.005 及其余阈值源码均未改变。** 已有旧 JS 风险数值和 probe 标签不能直接沿用，后续工作/最终标定必须用当前代码重新生成；本轮未执行真实标定。

测试覆盖相同分布、完全不相交且含零概率、共享零支持、极端 logits、float32/64、随机分布与直接概率公式、对称性/平移不变性、接近相同分布的稳定性、双输入梯度及 double gradcheck。probe 绝对门槛边界仍按既定单位执行。

### D11：D_id / A_sem 多余末端卷积

**设计：D_id §7.2；A_sem 编码段详列于 §7.5。** 原两者复用 `MaskedNet`，两 ResBlock 后多出一个 128→128 的 1×1 Conv。

修复位置：[representation/purifier.py](../src/ccrr/representation/purifier.py)。`MaskedEncoder` 仅包含 1×1 start 与两个原有 `ResBlock(128)`；`MaskedNet` 在此基础上添加 end，仅给需要末端投影的 Es/Ds 使用。

- D_id：64→128，2 个 ResBlock，有效区池化，原 128→2048 Linear，原 L2 normalize（epsilon=1e-6）。
- A_sem：768→128，2 个 ResBlock，原全局和 2×2 池化/640→256/对称配对头。
- Es 768→128→两块→64、Ds 64→128→两块→768、mask head、clinical head、统计 buffer、整数 Q 的范围与 `/7` 接口不变。ResBlock/LNch 的定义和 padding 清零规则未改。
- Qid/Qsem 仍为 Purifier 内部辅助特征，没有改成可传输 ZC；授权恢复继续使用认证整数与同一 INN 的逆。

这是**有意的非等价结构修正**：Purifier 删除 `identity.end.weight/bias`，参数数 2,318,726→2,302,214；SemanticAttacker 删除 `net.end.weight/bias`，936,193→919,681。各减少 16,512 个参数；没有新增参数键，其余键形状不变。初始化随机数消耗会变化，不能声称剩余参数初始化字节都不变。

见 [D11_structure.json](../verification/correctness_batch1/D11_structure.json)。将同一权重分别加载到修复前后 Es/Ds 后，两分支前向逐位一致；该局部检查不等于整个 Purifier 等价。逐层结构、实际 B×768×14×14 前向、2048 身份向量、所有 Purifier 参数梯度、源 mask 和双向对抗梯度隔离均通过。

### D12：单等级越界不能终止全部候选

**设计：§9.3、§9.4、§17。** 原 `SourceCodec.quantize` 用一个普通异常同时表达非有限和符号越界，`CCRR.payload` 没有候选隔离，一档越界立即退出。

修复位置：[codec/entropy.py](../src/ccrr/codec/entropy.py) `QuantizationRangeError,SourceCodec.quantize,encode`；[runtime/system.py](../src/ccrr/runtime/system.py) `CCRR.payload`。

- 先检查 shape、输入有限性与合法 level；量化结果再次检查有限性。只有形状/数值合法但整数范围超出原边界的情况抛专用 `QuantizationRangeError`。
- payload 只在 quantize 调用处捕获该专用类型，记录 `level/reason=symbol_range/detail` 并继续；不捕获其他 CCRRError、ValueError 或 encode/AEAD 异常。
- NaN/Inf、有限输入除步长后发生非有限结果返回数值错误，不伪装为容量候选拒绝；错误形状、协议等级、无效 CDF 继续报各自错误。
- 所有候选不可用/不适配时仍返回 **Insufficient Carrier Capacity**，列出各候选拒绝原因。不裁剪或回绕整数，不改步长/编码边界。
- 直接 `encode` 在索引 CDF 前检查当前合法等级，避免 level=0 误用负索引或未登记等级进入编码。
- 继续按已有 `quality_order`（训练显式 level_order 保持原作用）依次检查真实 rANS 明文长度 + 88 字节认证封装、M 完整预留、记录向上取整和单流上限；没有按估计熵或倍率数值替代质量顺序。

测试用稀疏真实整数载荷复现第一档失败/第二档成功、前两档失败/第三档成功、全部失败、正负半格边界、真实编码认证后的整数往返、乱序质量优先级、恰好容量与少一条记录、真实 CDF 导致单流超长后继续下档。异常注入只检查传播边界；成功案例均运行实际量化/rANS/AEAD，没有 Mock 成功。

## 3. 修复前复现与逐项测试结果

每项在改相应源码前先运行失败复现，再执行修复后相关测试。日志位于 [verification/correctness_batch1](../verification/correctness_batch1/verification_report.json)。

| 项目 | 修复前实测 | 修复后当步相关测试 | 核心新增测试 |
|---|---|---|---|
| D01 | 26 failed、10 passed、2 deselected | 51 passed、1 skipped、2 xfailed（D03/D04 尚未修） | `test_correctness_records.py`：所有模式/两流归属/实际认证隔离；最终另补同区域前缀两例 |
| D03 | 原 strict xfail 以 `--runxfail` 执行：1 failed | 29 passed、1 xfailed（D04） | `test_correctness_losses.py`：22 cases，损失/参数梯度/多视图/实际训练累积路径 |
| D04 | 原 strict xfail 以 `--runxfail` 执行：1 failed | 18 passed、1 skipped | `test_correctness_js.py`：5 cases，范围/定义/梯度/probe 门槛 |
| D11 | 3 failed、1 passed | 14 passed | `test_correctness_purifier.py`：4 cases，逐层/梯度/隔离/checkpoint |
| D12 | 10 failed、10 passed | 50 passed、1 skipped | `test_correctness_rate_matching.py`：最终21 cases，真实候选与错误传播 |

逐项测试的普通 skip 均为当步未指定可选原生参照。最终完整回归显式使用现有原生参照并核验 SHA，故没有 skip。既有 `test_known_design_gaps.py` 的三个测试均保留并转换为普通必须通过测试；D03 测试按正确的完整 batch 上下文传分母，没有删除断言或放宽标准。

## 4. 全部回归与适用性

本机 Python 3.11.15、PyTorch 2.6.0、macOS arm64，CPU；完整结果为 **133 passed / 0 failed / 0 errors / 0 skipped / 0 xfailed，9.49 秒**。相对原44项增加89项，原3个xfail现在通过。

| 验证 | 实际结果 |
|---|---|
| pytest 单元/集成、全包导入 | 133 全部通过；详见 [tests.log](../verification/correctness_batch1/tests.log)、[pytest.xml](../verification/correctness_batch1/pytest.xml) |
| pip check、ruff F、compileall | 全部 exit 0 |
| 安装 CLI、模块 CLI help、inspect | 全部 exit 0 |
| 缺资源 preflight | exit 2，符合预期；未开始训练，缺 CUDA、权重及官方源码配置 |
| 512 INN | 非零耦合和混合参数下正逆门槛 max≤1e-4、RMSE≤1e-5；参数梯度、冻结 INN 逆的公共输入梯度通过 |
| 三档量化、真实熵流、固定字节 | 原 golden vectors、边界整数及本机既有 CompressAI1.2.8 原生双向互通通过；无下载 |
| PAD-M1、C/M AEAD、独立凭证 | 精确往返、错钥、篡改、流级故障隔离、nonce 并发/重启通过；协议字节未变 |
| Writer/Decoder | 形状、区域/幅度限制、共享 Carrier/Writer 梯度与消息损失检查通过 |
| checkpoint/调度 | 底层实际 AdamW + RNG 下一步一致，修正模型保存/恢复、A–J 分派/失败依赖通过；D09 的完整阶段续跑仍未实现 |
| export/receive、512 smoke | 随机真实 bundle/PNG/Decoder 链路执行；C/M 盲提取仍为 Extraction Failure，不冒充成功 |

独立512 smoke连续往返 max=`1.1920928955078125e-07`、RMSE=`2.2153120227130785e-08`；合成零 latent 的完整 C=884 bytes、M=408 bytes、82条记录。Writer 梯度范数约0.968；正确记录 oracle 成功，真实盲 Decoder 两流失败。随机消息损失不用于跨运行等价性或医学指标比较。

复现命令（已有环境，项目目录中）：

```bash
python scripts/verify_refactor.py --out verification/correctness_batch1 \
  --reference-ans /path/to/existing/ans.so \
  --reference-sha256 <verified-sha256>
# 无本机可选原生参照时省略后两项；该项应如实跳过。
```

本次原生文件 SHA：`03b4b8dba1555eccda2b6000d4a2fd69036d10ec5ca1813b4526e7c14da90387`。工程验证脚本使用临时目录，不执行 `train`，不生成虚假教师准入。测试中的轻量组件和调用桩仅用于梯度/控制流，不能作为真实教师或完整训练的替代证据。

## 5. 参数、checkpoint 与受保护协议

只有 **D11 的两个子网络参数结构发生变化**。`CCRR-CHECKPOINT-1` 容器及存储函数未变；带旧 `identity.end.*` / `net.end.*` 的 state_dict 将被 strict load 拒绝，不自动删键或继续错误结构。新结构 checkpoint 精确往返通过。旧 A-mfm checkpoint 如包含完整旧 Purifier 状态，同样不能直接严格载入新实例；不能把格式容器未改误称为全模型兼容。

本轮没有正式训练权重需要迁移。后续 Purifier/A_sem 应从正确结构初始化。旧 D/E/H/I 训练结果即使键可加载，也使用了错误损失；旧 JS 标定缓存的量纲不同，均不能因能够加载而称为修复后的正式证据。

配置、数据/配对代码、INN、通信网络、量化算式、rANS、metadata/AEAD、教师适配器、优化器/对抗调度、checkpoint 序列化、probe 阈值源码和 golden vectors 均与 `52665d5` 字节相同。另按每模型独立 seed=77 核验 CCRR 系统288条状态的键、形状、字节与历史基线一致，**明确排除 D11**。完整结果见 [preservation.json](../verification/correctness_batch1/preservation.json)。

## 6. 尚未解决的问题与依赖

以下继续按原报告登记，不因133项工程通过而关闭。只补充依赖说明，本轮没有修复这些问题。

| 编号 | 剩余问题 | 主要依赖阶段 |
|---|---|---|
| D02 | Q8 未采用最终稿指定的自定义硬前向 autograd；CPU 未观察到的误差不等于跨平台保证 | 正式量化通信/恢复数值验收 |
| D05 | B/C/D/E/H/I 选点与合法等级排序规则缺项 | B及以后模型选择，非仅加载 A 权重的阻塞 |
| D06 | probe 部署背景未保存上一策略真实完整 C/M 布局 | G/H/I |
| D07 | 标定背景漂移修正、全图复核和统计不全 | G及最终标定 |
| D08 | 热图验证/fallback、独立无分支控制、弱攻击者适应检查未接全 | B、D及后续隐私证据 |
| D09 | 多优化器/RNG/策略/选点等完整续跑状态缺失 | 中断恢复；底层checkpoint通过不能代替 |
| D10 | 单独 J 无法可靠恢复被选 H/I 和完整 Profile | 独立终评重启 |
| D13 | Profile 对冻结数值和正式512的强制约束不全 | 正式运行前必须核对配置，不能任意改值 |
| D14 | object_id 显式全局查重不足 | 密钥/nonce完整发布协议 |
| D15 | 教师完整分类区间、P2及来源重叠证据不完整，尚无真实能力结果 | A完整能力验收 |
| D16 | J 独立验收、置信区间、四凭证/有损通道/细分漏斗未接全 | 最终科学验收 |
| D17 | 部分攻击对照查询集合、阈值和属性学习率分组不一致 | 独立比较 |
| D18 | 五角色/原始metadata/配对清单总摘要强绑定和采样审计不足 | A输入证据及后续训练复现 |
| D19 | microbatch/batch_pairs/bootstrap 配置未完整生效；默认值符合原设定 | 配置审计；本轮仅修复当前累积数学 |
| D20 | 匿名公开文件名契约未完成 | 对外发布 |
| D21 | 直接 receive 张量8-bit网格、记录当前合法等级前置检查不足 | 接收API完整契约；D01仅修空洞 |
| D22 | F 选择端点、部分依赖门禁和 I 启动依据不足 | F/I及独立阶段执行 |
| D23 | preflight 仅资源可读性/摘要，且检查所有阶段权重 | 当前 A CLI 的资源门槛与合格证明边界 |
| D24 | 实物 PanDerm checkpoint 容器/键/shape 尚未核验 | 实际 MFM 加载；不能预先断言只填路径或必需改网络 |

## 7. H100 真实权重加载与 A 阶段准入

**本轮仍不能宣布 A 阶段可以直接正式开跑或能力已具备。可以进入下一步真实权重加载核验的准备工作，不必等待所有 G–J 问题解决。** 下游 D05–D10 等并非单纯加载教师权重的统一前置条件。

当前可证实的直接阻塞和必要条件：

1. **资源与环境缺失。** 本次机器无 CUDA/H100；默认 `weights` 路径/摘要为空，PanDerm source/source_sha256 也为空。现有 preflight 返回拒绝。H100 上需要已合法取得的官方 EfficientNet-B0、ResNet-50、PanDerm-Base 文件及摘要、冻结官方源码/依赖、外部数据/输出路径、FP32/禁TF32环境验证。本轮不下载、不连接启动训练。
2. **当前入口的额外资源依赖（D23）。** `stage_a → preflight` 遍历所有 weights，连 ConvNeXt/Swin 终评基座也要求配置。它们并非教师优化的科学依赖，但在未另行修复分阶段 preflight 前，是现有 CLI 的真实门禁；不能绕过检查假称就绪。
3. **真实加载适配核验（D24）。** 先严格记录三基座的文件摘要、missing/unexpected/shape；PanDerm 按既定 norm 后196×768 patch接口、224几何和规定标准差核查。若文件容器需要适配，应单列最小格式适配修复，不能静默漏载主干。做 RGB/灰度前向、冻结参数/非零输入梯度和有效 token 检查。随机适配测试不替代这一证据。
4. **输入协议与来源证据（D18/D15）。** 在服务器重新核验冻结五角色、512预处理、标签、真实患者配对、摘要与预训练重叠情况，保留未知项；旧本机像素审计不等于 H100 实物核验。当前缺乏自动总manifest强绑定，不能只凭文件存在认定协议已证实。
5. **完整能力判定仍缺报告链路（D15）。** A 有训练与部分门禁，但完整患者聚类区间、逐类分母与缺陷、ReID P2单列等证据尚未接全。需要按最终稿补齐后，才能称“A阶段完整能力验证通过”；只运行现有 A 并保存 checkpoint 不够。

资源和加载核验通过后，A 的既定执行顺序仍是 Clinical 混合 RGB/灰度及两个同预算单域对照 → 冻结 PanDerm patch线性probe与统计 → ReID Teacher。Clinical 两域 AUROC≥0.75、相对域内对照下降≤0.03，逐类缺陷/区间完整报告；MFM 两域 patch probe≥0.75；ReID P1 三方向 AUROC≥0.70 且患者95%区间下界>0.5，P2单列并保留其余规定指标。阈值、训练预算和数据角色不变。

D09 不阻止从头做一次加载或从头运行 A，但现有 `--stage` 不是可靠的任意步续跑；在恢复机制补齐前不能承诺中断后精确继续。资源就绪、加载正确、能力达标是三个独立结论。本轮只提供上述五项修复的工程证据，未取得后两者的真实教师实验结果。
