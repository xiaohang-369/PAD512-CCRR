# G1 联合训练从第30轮续至第60轮

用户授权将已完成的 E1＋D1 联合训练延长到总计60轮。只新增第31–60轮，共2700次更新，累计5400次；独立新目录，不修改旧30轮结果，也不触及 GPU1/3 上的 E1-only 对照。

外置配置 `phase2-g1-resume60.json` 保留512×512、全局batch16、物理GPU0/2、FP32、常数学习率2e-6，以及原损失、新增监督权重、数据和评价协议。仅增加 resume 父运行/校验值并把 epochs 改为60。此交付严格拒绝其他配置差异，设备选择仍通过配置表达。

父运行：`/data/zwc/zyh/PAD512-CCRR-V4/runs/g1-train.20261022-122453.3890568`。
父模型：`g1-epoch-030.pt`，SHA256 `25ebc5e61e64c5249f61a23d51725a651e4534aae85cc2bad08602f16dbb0c74`。

恢复全部 E1/D1 参数和 TF1Adam 的 m/v、updates=2700、lr。依据旧连续批次实现，从 position=2700×16=43200 接续数据顺序。旧检查点没有保存全局 RNG，不能宣称服务器数值逐位等价于从未中断的60轮运行；当前 E/D 无 dropout，批次顺序由独立确定性种子生成。CPU连续/恢复更新等价测试并不替代H100验证。

医学监督必须保留最初 G0 的概率目标：复用原预检 `fit-bank.pt` 中固定的 target、RGB特征、标签和裁剪位置，只用第30轮 E1 刷新 G1 图库。不能重新把第30轮 G1 作为医学教师目标。

新预检加载第30轮模型、优化器与数据位置，执行真实batch16反向但零更新，并重新评价第30轮。报告应显示 start_epoch=30、start_step=2700、optimizer_state_updates=2700、initial_G1_equals_G0=false。随后训练第31–60轮；新 metrics.jsonl 只包含这30轮，与旧文件按epoch拼接分析。

安装：`bash install-code.sh`。启动：`bash start-training.sh`，新预检成功才会后台续训。安装不修改环境，不启动训练。默认使用已有 ccrr_v2 环境。启动脚本用 flock 避免同一入口重复并发；不要绕过脚本重复派发。预检错误时不启动训练。

输出 `runs/g1-resume-preflight.*`、`runs/g1-resume-train.*`。完成后 report.json 应为 training_completed=true、optimizer_updates=5400、optimizer_updates_this_run=2700、final_checkpoint.epoch=60、next_phase=stop。第60轮停止，不进入嵌入或独立ReID训练。

当前身份指标仍为固定公开ResNet50、同病灶跨照片、val_tune开发评价；不称独立医学ReID攻击验收。医学只报告变化，不设通过阈值。
