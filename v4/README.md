# V4：基于 Invertible Grayscale 原方法的优化与扩展

状态：2026-10-21，第一阶段原方法训练入口已实现，本地工程验证通过；服务器运行与PAD训练尚未执行。

最新：已选定学习公开灰度与加密恢复差值路线，新增真实PNG双路密文/可逆基底恢复的协议诊断；未完成同域训练、监督改进或自适应Writer。

用户明确优先顺序：原方法训练G0 → 加监督训练G1并评价 → 差值加密、自适应嵌入得Gpub并评价 → W/R恢复G1和Zc、重组G0、原D0恢复RGB。当前只执行512×512、全局batch16的原方法基线阶段，暂不继续协议诊断。见[workflow-v4.json](configs/workflow-v4.json)。

旧9696b59的256/batch8配置已撤回；当前512/batch16保留完整90＋30epoch，旧包只作历史证据。

## 当前第一阶段入口

- [论文/源码核对、精确损失与训练日程](PHASE1_AUDIT.md)
- [服务器逐步运行说明](PHASE1_SERVER.md)
- [独立训练配置](configs/phase1-train.json) 与 [资源配置](configs/phase1-resources.json)
- `scripts/run-phase1.py`：prepare / preflight / train，支持单卡/DDP，120epoch结束即停止。

## 历史工程入口

- [V4设计与修改边界](DESIGN.md)
- [当前已选方案与运行命令](RESIDUAL_PLAN.md)
- [新增步骤、接收接口与训练职责推导](INTERFACE_DESIGN.md)
- [逐层来源对应及差异](SOURCE_MAP.md)
- [独立配置目录](configs/README.md)
- `upstream/InvertibleGrayscale/`：完整作者仓库的固定版本内容、原README和许可证；不修改、不直接运行原main.py。
- `ccrr_v4/original.py`：可核对的PyTorch迁移，默认只有原E→单通道→原D。
- `scripts/run-verify.sh`：CPU/指定GPU及已有监督器的无更新验证。

本地测试：`PYTHONPATH=v4:v3:v2:src python -m pytest v4/tests -q`。需要已有torch/numpy/Pillow/pytest；不通过脚本安装依赖。

旧验证入口说明（当前第一阶段请用上方PHASE1_SERVER）：服务器先安装独立离线包，然后用包内README执行CPU验证。GPU用`--gpus`显式指定一张或多张卡；多个设备逐卡验证，不宣称已经实现DDP训练。原监督器路径通过`--resources`读取配置，只有已选的fit首批像素参与接入检查，test不评价。

原作者代码与论文存在差异；本次源码基线采用固定作者实现。源码算子/结构检查不是TensorFlow运行时逐位复现，随机初始化检查也不是训练结果。当前实际使用的9个VGG卷积层已与牛津官方Caffe逐元素核验，完整原损失的合成图CPU梯度检查通过；不等于服务器训练成功。

上游：Xia, Liu, Wong, *Invertible Grayscale*, ACM TOG 37(6), 2018. https://doi.org/10.1145/3272127.3275080

保留作者版权与许可证：[LICENSE.txt](upstream/InvertibleGrayscale/LICENSE.txt)。派生实现随同该许可与引用说明分发。
