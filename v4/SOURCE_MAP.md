# 作者源码与V4迁移对应

作者仓库 https://github.com/MenghanXia/InvertibleGrayscale ，固定commit `5dc2a0f7c6b692ca7914941745bd29b227ff4dbb`。
完整7文件保留，逐文件SHA-256在`upstream/provenance.json`。本地完整Git clone在研究目录`参考源码/InvertibleGrayscale`。

未来新增职责及其具体接入位置见[接口推导第5节](INTERFACE_DESIGN.md#5-每项新增职责怎样对应作者代码)；该表不表示这些模块已经实现。

当前新增的`residual_transport.py`位于原数字灰度输出与原Decoder输入之间：计算两张学习灰度的int16差值，真实编码/加密/嵌入/提取后还原原uint8编码图。它是我们新增的协议实现，作者源码中没有对应模块；并未修改`encode`、`decode`或增加公共头。诊断载体与未来自适应Writer区别见[当前方案](RESIDUAL_PLAN.md)。

| 原文件位置 | V4对应 | 核对点 |
|---|---|---|
| model.py:11–19 Conv2d | original.Conv / same_pad | 无bias，std=.02，TF SAME stride2偶数尺寸前0后1 |
| model.py:122–179 encode | original.Encoder | 64/128/256、2+4+2残差块、最近邻上采样、内部加法skip |
| model.py:177 latent | Encoder.latent | 64→1，tanh；无额外公共头/私有分支 |
| model.py:182–198 decode | original.Decoder | 1→64，8残差块，64→256→3，tanh；无发送端skip |
| main.py:47–48 | OriginalGrayscale.forward | 训练连续E输出直接接D，不默认STE |
| util.py:26–38 | original.quantize8 / verify PNG | [-1,1]→round至uint8→PNG重读→[-1,1] |
| main.py:63–82 | losses.source_terms / author_loss | 灰度系数、TV全batch均值之差、70/127死区、MSE、网格距离 |
| main.py:57–58 | losses.legacy_resize_bilinear | TF1非half-pixel坐标，不能直接用常见PyTorch interpolate替换 |
| main.py:85–96 | configs/train-author-reference.json | 90+30、polynomial .9、共享step、两个Adam各自状态；训练器未实现 |
| model.py:249–258 | 配置中的VGG接口约束 | 代码conv4_1后return（带ReLU），论文写conv4_4；权重加载/验证待完成 |

不改变原E/D宽度、残差块数量、卷积偏置设置和激活位置。输入扩展到4的倍数、NCHW布局及PyTorch运行时属于迁移；数据与随机数序列不同，不承诺复现原论文数值。

损失差异：论文VGG项L1，代码MSE；论文beta=.5，代码第二阶段=.1；代码恢复项系数3且第二阶段外观项不是统一乘论文omega1=.5。严格记录，不混合成所谓原设置。

局部变化在源256×256上使用`TV/256**2`；迁移对其他尺寸采用`TV/(H*W)`，是显式分辨率适配。完整VGG特征缺失时报错，不以其他特征或0代替。量化网格最小距离用相邻两级计算并检查函数值/梯度，不建立256倍大张量。

验证层级：
1. 原文件哈希；
2. 实际执行原model.py的encode/decode，底层算子由独立NumPy实现，使用同一组权重，与迁移的整链输出比较；
3. TF padding、resize、量化、参数映射的独立数值例；
4. PyTorch前后向、参数未更新、实际PNG整数重读；
5. 服务器真实fit/A0一致与冻结监督输入梯度（待回传）。

第2项没有执行TensorFlow库，不能写成已通过TF1.6运行时数值对照；也没有载入作者预训练E/D权重或完成VOC/PAD复现。VGG权重来源、许可、哈希及完整感知损失验证仍是后续工作。
