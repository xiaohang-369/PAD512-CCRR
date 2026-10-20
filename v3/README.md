# V3：ICCV 式 PAD 前置监督开发实现

当前生成接入使用第9节R5入口：直接batch16，无梯度累积。旧pilot/start属于历史短跑；R5没有自动启动训练的start动作。

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

2026-10-20审查修订：`support_policy=bounded_codes_v1`。标准化后的输入在`coverage=0`处置零；量化Q在这些位置置零，再送Mask和解码器。Q_sem/Q_link保持互补，Mask在无效位置为0；各解码入口再次屏蔽未计费位置。预热与联合训练共用`complete/encode/decode`路径。`coverage>0`控制二值支持域，分数coverage只用于原有损失／池化权重，不衰减部分覆盖特征。解码器内部卷积仍可经过支持域外，但输入只能来自受约束的码；不宣称逐层空间隔离、病灶保护或padding token不含信息。A0磁盘缓存和医学probe不变。

GMM使用对称erfc尾差，保留原概率／尺度下界及触发统计；不是更换熵模型或真实通信预算。额外攻击器仅设置CPU默认生成器并恢复CPU状态，迁移设备不重新设置CUDA种子。缺少支持域版本的旧separator拒绝在新版加载，应使用其原release审计；旧probe/targets不因此失效。

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
| `link-audit` | `--features --separator --targets --output` | 无更新，直接比较教师、D_link输出和固定fit均值的P2与单位化重建误差；targets必须是该separator绑定的教师 |
| `compare` | `--before <audit> --after <audit> --output` | 匹配临床及关联差值区间，仍不自动颁发通过 |

训练所需的`reference-review`见样例：需绑定已审阅targets的artifact.json摘要，三方向都明确记录，决定为`allow_development_reference`并写出依据。它只是允许该候选进入开发比较的记录，不是正式教师通过证书；不得无视初始化／困难负例／有效性证据直接批量生成。旧V2已训练ReID不能通过`--reference`静默导入。

`train`仅加载fit/val_tune特征；`probe`及默认`audit`另用val_select开发报告。`audit --evaluation-role val_tune`只加载fit/val_tune，用于DEV2端点筛选。数据完整性核验可读取test角色元数据，但不读取test像素／特征或使用test性能。val_select已有开发历史，不能称新的无偏终测。

## 5. 评价解释与未闭合项

对照是独立训练的compression与separation，而不是同一模型中full与semantic的替代比较。附加对抗的作用单列。若要归因到学习Mask本身，还需实现并登记匹配随机／固定Mask对照；本版未实现该额外消融。

攻击方向仅在val_tune固定，在val_select不再反转或选点；同时报告双方向保守诊断，不能把低于0.5的AUROC视为安全。相同分数Rank-1与AP采用随机打破平局的期望，避免图像名排序制造成功。P3近1的NCC在1e-6范围合并为平局，以避免浮点误差使自身匹配虚高。

pair-AUROC按查询患者及每查询正负集合平衡；区间是**给定固定图库、已拟合攻击器时的查询患者聚类不确定性**，不包含重新生成图库或重训模型的方差。临床区间按患者聚类。缺类bootstrap报告无效比例；区间不足不得当通过。多seed、困难负例同查询子集报告、完整最强攻击联合准入、正式阈值仍需后续登记／完善，不能把当前compare.json当最终科学验收。

P3当前仅普通灰度NCC基线，未实现生成图的P3训练或局部特征攻击全集；其高匹配率不是不可行性证明。监督器灰度接口虽有输入梯度测试，真实PanDerm—生成像素链路尚待服务器检查。未提供任何“自动晋升合格监督包”功能。

P3主报告`p3_baseline.json`仅对完整公开画布居中／单位化；`p3_source_support_auxiliary.json`另外读取源V，明确标记已知有效区辅助条件。两者均为同一确定灰度变换的自来源匹配，不能用于不同的生成查询／图库。攻击者是否可获取或推断V需另登记，不从接收端契约推定。

`link-audit`仅评价val_tune/val_select患者，均值只从fit单位化教师embedding按域计算，再按同一epsilon单位化。报告三方向固定正向cosine P2、Rank-1/mAP/AUROC、val_select查询患者区间，以及逐域单位化距离和近零范数计数；零向量形成平局，不冒充合格表示。它不拟合新攻击器，不替代semantic独立攻击或医学验收，val_select仍是开发数据。结果不自动触发后续训练。

## 6. 记录及工程验证

每运行写`binding.json`、`events.jsonl`、`status.json`及带内容SHA的`artifact.json`。运行状态completed只表示完成预算，`scientific_acceptance=not_evaluated`不变。父资源／cache／模型的内容与来源绑定，禁止路径逃逸、错误阶段或不同特征来源。输出已存在即拒绝，不删除重跑。

separator.pt保存模型、头、fit统计、优化器和主要RNG状态供审计；当前**不支持恢复命令**，不能仅凭这些字段宣称精确续训。P2参照保存选中模型而非完整恢复状态。模型输出、配对明细和患者标识只留私有服务器，不提交Git。

本地检查使用合成数据，可验证硬量化、互补Mask、GMM概率、输入梯度、冻结教师、三组短循环、P2／P3合法性、配对统计、hash防篡改、配置登记和输出隔离。合成更新是工程测试，不是PAD训练或科学结论。既有V2测试同时回归；实际结果以提交时实验记录为准。

无需pytest的数值检查：设置本release的`PYTHONPATH=v3:v2:src`后，使用既有解释器运行`python -B -m ccrr_v3.numerical_check --device cpu`。服务器仅暴露指定物理卡3的UUID后，用`--device cuda:0`同时检查CPU与该卡：支持域扰动、尾概率／梯度、攻击器RNG及特征级冻结监督。只构造合成小网络、做反向，无optimizer.step、不加载真实数据或权重、不保存模型。更完整回归可设置`CCRR_TEST_DEVICE=cuda:0`运行`v3/tests/test_revision_regressions.py`，要求pytest已存在，不自动安装依赖。真实PanDerm像素链和科研准入仍待独立验证。

## 7. DEV2有限训练与checkpoint联合筛选

用户2026-10-20选择保留当前2500步基线，另开同预算compression/separation，最多10000步，不预设严格医学门槛、不自动选冠军。配置可选字段`checkpoint_steps`声明保存端点，`temperature_decay_steps`将温度退火长度与总预算分开。旧配置行为不变。DEV2沿用500预热及2000联合步退火至0.5，之后保持0.5；固定学习率与原损失不变，不把延长退火造成的早期轨迹变化误当作训练时长效应。

0—3000每250步、后续每500步保存，共26个端点／组。每步有放回采样16个图像位置，同一批分别计算RGB/灰度损失并平均，不能将图像位置累计量称为严格epoch或独立图像数。保存模型、头、统计、优化器及RNG供审计，仍不支持续训。训练不是自动早停：运行目录出现`STOP_AFTER_CHECKPOINT`文件时，在下一个登记端点保存并正常结束；日志标明实际步数和最大预算，停止的两组预算须对齐。不以冻结头的波动自动判定联合指标停滞。

`python -m ccrr_v3.dev2 register`从已登记DEV1验证父文件并建立新的DEV2计划，不训练。随后显式`train compression`或`train separation`一次运行一组。默认固定当前2500步基线为可用实验候选；本轮优化不覆盖旧release或产物，也不自动进入生成训练。

对各端点使用`python -m ccrr_v3.dev2 audit <variant> --separator <run> --step <saved-step>`。每次从fit重新拟合Linear/MLP各2000步和global/spatial各3000步，val_tune选评价器和攻击方向，再在val_tune报告筛选指标。它是带选点乐观性的开发评价，不是独立泛化终测。完整26×2个端点评价包含52轮审计，评价成本可能高于训练。一次只执行显式指定的端点，不静默跳过、复用不适配的攻击器或自动批量发起。

`python -m ccrr_v3.selection --audits <matching-val_tune-audit-runs...> --output <new-V3-run>`验证同代码、训练／评价预算、同队列／图库、每组单一训练run及配对端点。44个目标为4项医学AUROC、4项macro-F1、24项逐类召回（均越高越好），6项保守双向攻击AUROC和6项Rank1（均越低越好）。保留各组及合并的全部经验Pareto集合，不宣称统计显著支配。默认每组展示最多2个覆盖权衡范围的代表，确定性规则为按步数的首个非支配点起始，然后对44维按候选范围归一化后做最远点选择；这只是展示抽样，不是加权得分或质量排序，完整前沿不丢弃。

输出`candidates.json`、`metrics.csv`和可离线打开的`tradeoffs.html`，后者投影展示最弱医学AUROC与最强攻击AUROC并按步数连线，不代替全部F1／召回指标。固定有限候选后才单独安排val_select复核；当前包装入口不自动使用val_select，更不读取test表现。没有阈值时不写`passed`或自动决定正式监督器。真实曲线只能在服务器完成训练和端点评价后生成，本地合成检查不构成结果。

## 8. DEV2候选接入P/E_C/D：8500主候选、7000对照

用户在审阅52个端点评价后明确选择8500步接入，7000步保留对照。
`grayscale.py`按《设计2.0》第5节实现公共P、私有E_C和接收端RGB Decoder：
512画布、8×32×32私有码、4/8-bit hard-forward half-up量化、零初始化公共残差。
不使用旧INN，恢复是学习式近似恢复，不声称解析可逆或无损。
Decoder只接收灰度、反量化整数码与精度编号；无发送端skip。

`PixelSupervisor`使用对应checkpoint原绑定的域内医学头与fit标准化，
不替换成独立评价读出器。PanDerm和分离器全程冻结且eval，但生成图保持输入梯度；
硬量化采用原STE，确定性Mask的反向沿用原softmax代理（默认温度1）。
源RGB／普通灰度的KD目标停止梯度。源缓存只用于校验实时提取一致性，不代替生成图特征。

显式命令`python -m ccrr_v3.grayscale_dev register`绑定已有8500/7000文件、
完成的独立评价和新代码，不修改旧实验。`verify --supervisor-step 8500`
在真实fit第0张512图上检查RGB/灰度实时PanDerm与A0缓存一致（atol/rtol均1e-4）、
像素输入梯度、P医学梯度、冻结参数/缓冲区和PNG＋整数码重读。
无optimizer更新。7000可用同一入口独立核验。

`start --supervisor-step 8500`先执行上述真实检查，成功后才在新目录启动有限B/C1 pilot；
也可用`pilot --verification <本版本同候选verify-run>`显式指定已完成检查。
每个候选从相同初始化重新开始，seed20261020，batch2有放回，Adam1e-4：
500次固定普通灰度B更新E/D，随后500周期C1各更新一次E/D和一次P，
共1500次optimizer更新、2000个抽样图像位置（C1同batch用于两子步）。
每阶段每100周期保存，保留全部10个checkpoint，不自动选点、早停、续训或重试。
每阶段末在全部169张val_tune上输出绑定监督头的医学诊断与恢复/灰度指标，
不是独立读出器或隐私验收；图像面板只存私有服务器。

恢复L1＋0.1SSIM＋0.1RGB医学，公共外观1、医学0.1、range0.001、KD0.5；
appearance遵照3×3十二邻接边、8×8池化对比和8/255亮度无惩罚区。
源/输出域保持匹配。第一C1周期记录加权目标在P上的梯度范数与夹角，
普通日志保留P/E/D梯度、码饱和/符号均值和灰度越界率。
高维对齐、区域/概念、rate与身份损失均为0，符合B/C1范围。
还未实现C2/C3直接像素P2/P3对抗、Writer、认证密钥或完整通信；本pilot不声称隐私通过。
本阶段仅训练8-bit，4-bit结构与量化可用但训练/率失真评价仍待后续D1。

新运行入口按指定GPU UUID绑定，不检查显卡占用、其他进程或空闲显存。
任何真实数值、资源绑定或运行错误仍失败停止，保留证据，不杀其他进程。


## 9. R5 直接batch16生成配置（准备完成，未启动服务器训练）

新入口为 `python -m ccrr_v3.grayscale_r5_cli`，配置文件为
`v3/configs/grayscale-r5-direct16.json`。8500主监督、7000对照，绑定DEV2
原checkpoint医学头；不以独立审计头成绩替代原头诊断，不重新训练分离器。

- batch_size=16、microbatch_size=16、accumulation_steps=1，FP32张量/参数/优化器；cuDNN允许TF32、矩阵乘禁用TF32，与已复现A0一致。
  单次完整16图像前向/反向，不实施梯度累积、不自动缩batch。
- AdamW：ED基准1e-4，P基准1e-5，betas=.9/.999、eps=1e-8，权重衰减1e-4，bias为0。
  每阶段100更新warmup后余弦至基准的0.1倍；B→C1保留ED矩状态，重启阶段日程。
- B2000 ED更新；C1最多10000个ED→P周期，同批16图像、P重新前向。
  合计ED12000/P10000更新、192000抽样位置；采样有放回，非严格epoch。
- 固定灰度对照ED同为12000更新，P不更新；计算量另报，不能称总算力相等。
- 初始及每250周期全val_tune冻结头AUROC/F1/逐类召回、重建和外观诊断；
  B8+C1 40端点。无独立图像医学/身份攻击结论，无自动冠军。
- 无效补边统一128/255，V仅发送端使用，接收D只读灰度与码；8-bit正文8192bytes。
  4-bit正文4096bytes可打包/解包，但本阶段不训练4-bit，正文不包含AEAD/profile/记录开销。

register只登记，verify仅反向检查不调用optimizer.step；二者都不训练。
verify还保存全量val_tune原绑定头诊断、真实batch16无更新ED/P耗时与峰值。
该耗时不是训练吞吐，模型快照大小也是估计。实际训练周期耗时、峰值和快照字节
由未来显式train记录。plan必须基于对应监督器的verify结果以及显式墙钟/快照存储上限。
train需要已登记budget；resume需要同一budget和上一个run的latest完整周期checkpoint。
没有start命令、自动重试、自动扩大预算或自动7000/8500第二轮训练。

完整周期保存模型、两个优化器、阶段日程位置、全部RNG、采样/seen、父哈希和累计成本。
新run续训，不覆盖旧目录。对同一个run使用锁以防重复续训，**不检查GPU占用**。
捕获的中途异常保留成本，舍弃的工作也计入原预算；因此恢复后可能先用完预算而少于目标步数。
SIGKILL/掉电留下未完成成本日志时拒绝自动续训，需核对实际成本后另行登记，不能默认为零。
CPU连续/恢复等价测试不等于跨硬件、库版本或所有CUDA算子的逐bit保证。

墙钟在完整周期/诊断边界检查，允许正在进行的一个周期或诊断结束后停止，保留其实际耗时；
不宣称硬实时中断。快照上限统计本续训链保存的pt文件累计字节，不删除旧快照以腾额度；
临时序列化需额外单快照空间，诊断/日志和旧实验占用另列，不是整个磁盘配额。
`STOP_AFTER_CHECKPOINT`只作用于当前run的下一个端点，仍须手动显式续训。

本实现范围为R5的B/C1准备，不包括C2/C3像素P2/P3训练、独立生成图评价、D1四位训练、
源profile/认证流、Writer或完整授权。所有科学验收仍为not_evaluated。


### R5 A0数值口径修正（2026-10-20）

服务器诊断已证明：首16个fit样本在batch16、cuDNN TF32允许时RGB/灰度均与A0缓存逐元素相同；
单图比较及batch16禁用cuDNN TF32都不满足原容差。源码、模型哈希与主要依赖版本一致。
旧f46e634先用batch1检查缓存、随后Engine又关闭cuDNN TF32，这两处不一致由新版本同时修正。
验证改为对首16个样本完整比较并在相同batch执行输入梯度检查，保留atol=rtol=1e-4；
PNG与单图接收测试单列，不再混入缓存一致性条件。训练、目标、验证与诊断统一使用
float32张量、cuDNN TF32允许、matmul TF32禁用，无autocast，无低精度参数或优化器状态。
本次保留实际A0数值参照，不重新生成缓存或重训监督器。TF32是允许使用的后端模式，
并不声称每个卷积一定用了TF32，也不承诺跨batch输出逐bit一致。
新计划目录为plans/grayscale-r5-direct16-a0-20261020；旧登记、失败run与旧release全部保留。
R5仍为直接batch16，无累积、不启动训练；新版本真实完整verify必须重新回传后才能算通过。

### R5 按步数完成训练（用户修订）

新登记目录为 `plans/grayscale-r5-direct16-steps-20261020`。
`plan --verification <run> --steps-only` 明确登记 `resource_policy=steps_only`，
`wall_seconds=null` 和 `checkpoint_bytes=null`：不因墙钟或检查点累计空间停止。
不是用极大数值伪装无限制。正常从头训练按 B2000＋C1 10000 完成，
仍记录实际资源使用，每250保存/诊断，网络、batch16、学习率及损失不变。
真实I/O、CUDA等运行错误仍报告失败；人工停止仍有效。旧限额run不原地改预算。
8500主实验重启从相同登记初始化开始，不读取旧训练checkpoint；旧文件保留。
