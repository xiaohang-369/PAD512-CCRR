# V4 稳定版本、权重来源与历史对应

## 稳定范围

基点为 d398fc11ff3044f352c8c9647152013e4552d48f。模型、原损失、原训练/推理入口和权重加载源码与该基点逐字节相同。新增范围只有只读检查点收集器及其配置/测试、文档导航和Git忽略规则；不改变科学方法。本次不部署、训练或选择新模型。

X为RGB [-1,1]；E0/D0均为原方法CNN，E0输出单通道，D0恢复RGB。第二阶段 E1/D1 为从120轮 E0/D0 复制的可训练副本，E1仍直接接收RGB。联合优化原RGB恢复/VGG/TV/lightness/quantization，以及冻结医学/身份监督。医学8500仅主监督，7000仅额外评价；身份为冻结公共ResNet50 RGB/G1图库的均匀KL，不是新排序损失。

原重建使用连续灰度，新增监督使用原8bit前向+STE反向；部署/评价需要数字PNG一致性。E1/D1成对实验恢复与固定D0不适配诊断应分列；不能把后者下降称一般身份隐私成功。

## 已完成运行及模型来源

|阶段|真实代码提交|服务器run后缀|权重/末轮|
|---|---|---|---|
|E0/D0原方法|ba37500af51c483c74c94ac26d2c8cbf56e5121a|phase1/train.20261021-182445.3665556|epoch-120.pt，epoch120/step10800|
|E1/D1联合|7f575168107f38ef31b93f50c2af32a183cc78c9|g1-train.20261022-122453.3890568|g1-epoch-030.pt，epoch30/step2700|
|联合续训|d398fc11ff3044f352c8c9647152013e4552d48f|g1-resume-train.20261022-174510.3977016|g1-epoch-060.pt，epoch60/step5400|

根路径均为 `/data/zwc/zyh/PAD512-CCRR-V4/runs/`。对应checkpoint SHA256：

- epoch120：5d7f8c2bed98356166bbecebbc87a7fb4324e75126a347e6173788a7fcbef387
- epoch30：25ebc5e61e64c5249f61a23d51725a651e4534aae85cc2bad08602f16dbb0c74
- epoch60：685899c0c2b266a0ff6076b3d0b2ac33d67e42c9eb8d2b0559f5c37d9ff621a1

阶段1保留90+30轮和TF1Adam原日程；阶段2lr=2e-6、betas0.9/0.999、eps1e-8，global16、FP32。ContinuousBatches跨逻辑epoch拼满16；30轮2700步。续训加载原30轮模型、Adam m/v/updates及数据位置，从epoch31开始，不是重新训练60轮。旧checkpoint缺全局RNG，因此不声明与从未中断运行逐位等价。

包内旧 manifest 的“pending”是交付当时时点；完成状态由后续 binding/report/status 回传确认。PROVENANCE.json 记录三个原交付包外层SHA、全部SHA256SUMS校验、bundle中的提交，以及30/60轮真实回传绑定。原60条检查点清单保留在本地审计中，不上传检查点或逐患者评价数据。

## 医学与数据资产

历史服务器Python为 `/home/zwc/anaconda3/envs/ccrr_v2/bin/python`；PyTorch2.6.0 / torchvision0.21.0、512×512、FP32、CUDA/NCCL；90/30/60预算和单卡/双卡参数以各运行binding为准。GPU0/2是联合30轮历史配置，GPU1/2是60轮续训历史配置，不能当通用可用卡承诺。

- 原始数据根：`/data/zwc/zyh/data/PAD-UFES-20`；固定patient-disjoint roles与prepared绑定，不能混入val_tune优化。
- prepared：`/data/zwc/zyh/PAD512-CCRR-V4/data/phase1-pad512-b16/prepared.json`，SHA256 5025af1f633450c2eb7d7a32420dffd7eb87c1d0d4e482af9b00021339f9832e。
- 原作者VGG：resources/vgg19-conv4_1.npz，SHA256 978ab8d420115421506279b38f5fc19a8a74fb02036d5f9c043604fa07fcf53b。
- PanDerm权重：ccrr-resources/weights/panderm_bb_data6_checkpoint-499.pth，SHA256 be1e0fb108b3bc58721cb5195f136c948160799438f222acf1fd142230ac1ff1；模型官方源码必须匹配原检查报告的source SHA。
- separator8500：PAD512-CCRR-V3/runs/train/dev2-separation.20261020-184228.3079814/separator-step-008500.pt，SHA256 954353662c276981ca5d493ce63e93686eb321f24d1220657d919a9b36c99dc5；7000对照SHA256 98516d3dab7fccbe410657ad5ee79ff0fb90de1eaddd730af438c7b0c2d8fb46。
- load_separator从同一checkpoint读取separator、heads、statistics；fit统计来源为历史A0-features/fit_statistics.pt，不可随机替换头或统计。
- 公共ResNet50：ccrr-resources/weights/resnet50-11ad3fa6.pth，SHA256 11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca。

资源前缀为 `/data/zwc/zyh/`。绝对路径来自历史回传，不是此次实时登录核验；本次未访问服务器、未改环境或权重。配置仍保留历史路径与哈希门禁。git稳定提交号不同于历史checkpoint的code_commit时，应按原binding字段加载/校验，不能篡改checkpoint迁就当前HEAD。

## 历史代码与独立实验

旧main 5e086984aa26cbd00af24326d2b985afa4b3a282 为当前V4基点的祖先，二者距离40个提交。旧main归档为 archive/pre-v4-main-20261102；本地main仍可能停在更早fb68b6b，不能错把本地main当远程旧main。

experiment/e1d1-rankloss-20261102 的提交7787c62b50d8208901cfff20ea203eda6dc471f7仅为独立排序损失预检。稳定分支从d398fc1独立建立，没有合并/cherry-pick该实验提交，不含rank_identity.py或rank_preflight目录。

V4原有residual_probe/transport与设计文档保留于历史树。原63个未提交文件只纳入正式E1/D1只读证据工具3个，其他残差RD/编码/恢复/扫描实现和内部生成清单保持本地历史，不接入稳定执行链。

macOS原Git存储存在V4与v4/stable的大小写/目录冲突，因此stable使用独立bare Git存储加worktree；原7个worktree和分支不改名、不移动。备份bundle保留全部引用和历史，待处理文本另有完整tar和patch备份；不会为“清理”删除原工作。

## 验证口径

运行现有V4和兼容回归、CLI --help、JSON解析与schema检查、原权重来源/实现哈希检查；不运行正式实验。完整原环境SciPy1.15.3在本机出现Mach-O加载错误，重装到隔离目录的同版本wheel仍有该问题；兼容回归另用审计目录下SciPy1.14.1覆盖，不改原依赖声明、现有环境或服务器环境。V4回归可在原锁定环境运行。四个历史测试目录分别使用独立进程，保留其同名测试和裸模块fixture导入，不修改旧测试来强行合并收集。最终通过/失败/未测数量以随治理报告提供的实际日志为准，兼容依赖通过不等于SciPy1.15.3锁定环境全量通过。
