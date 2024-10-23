**Q1 What dose alfabet / alfabet-lite come from?**
St. John et al., (2020)开发的基于快速准确预测有机分子的键解离焓 (BDE)的目的的 graph neural network工具
alfabet-lite是他们开发的仅用于预测C、H、N 和 O 原子键解离能的简化版本

"This tool predicts BDEs for single, noncyclic bonds in neutral organic molecules consisting of C, H, O, and N atoms. Mean absolute errors are typically less than 1 kcal/mol for most compounds. "
(St. John et al., 2020)

**Q2 What is alfabet / alfabet-lite?**
alfabet(Fast, Accurate Bond dissociation Enthalpy Tool): 一个训练完成的训练的图神经网络模型，在基于Tensorflow (2.x)和neural fingerprint library (0.1.x)已处理的特征fingerprints,用于以接近化学精度和亚秒级计算成本来预测均解 BDE
alfabet-lite：完整模型框架的轻量级版本用于预测C、H、N 和 O 原子键解离能的简化版本.


**Q3 What’s the difference between alfabet-lite and  alfabet?**
This Lite version of the model is a lightweigted version of the full model framework
1.Enhanced compatibility with modern Python and Tensorflow versions（可以用python3.9跑以及和TensorFlow 2.x适配）
2.Dropped dependency on nfp (不需要只能从neural fingerprint library导入)，有自己的TensorFlow库了

**Q4 Where we use alfabet-lite？**
alfabet-lite：准确地预测类药物分子代谢中氢提取的主要位点(预测分子中最弱的键) **还没会用**
alfabet： 加上烟灰形成过程中的主要分子碎片途径(预测燃料分子燃烧过程中形成的主要自由基)

**Q5 How to download it?**
alfabet-lite：载入GitHub库，新建alfabet env, 同时安装alfabet-lite[tensorflow]
或者先装install tensorflow version 2.15，然后载入alfabet-lite存储库
要配置新的tf.compat.v1.Dimension

**Q6 How to use it?**
exmsple test

pps BDE: 气相反应焓变 Homolytic bond dissociation enthalpies (BDEs) are defined by the enthalpy change for the gas-phase reaction at 298 K: