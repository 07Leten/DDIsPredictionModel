# 基于深度学习的药物-疾病相互作用预测 (Drug-Disease Interaction Prediction)

本项目基于论文https://doi.org/10.1002/advs.202409130，原项目地址：https://github.com/WBY20/DrugComb_Disease_Prediction。本项目为拥有者在清华大学新雅小课《基于人工智能的中医药推理模型与探索》课程基于论文代码对单一药物模型构建与训练进行实践复现时所作。本项目作为拥有者的代码记录仓库使用。  
本项目包含一个深度学习模型及相关脚本，用于预测潜在的药物-疾病相互作用 (DDIs)。该模型利用基于网络特征的药物嵌入（利用SMILES表示法并基于基因靶点）和疾病嵌入（基于 MeSH 描述符）来预测相互作用的概率。

## 项目概述

识别潜在的药物-疾病相互作用对于药物重定位（Drug Repositioning）和安全性评估至关重要。本项目实现了一个全连接神经网络 (FCN)，它接收高维特征向量（19,292 维）作为输入。这些特征来源于：

*   **药物嵌入 (Drug Embeddings):** 通过在基因相互作用网络上进行随机游走生成，捕捉药物对生物通路的影响。
*   **疾病嵌入 (Disease Embeddings):** 基于 MeSH (Medical Subject Headings) 描述符生成，捕捉疾病的表型和关联信息。

模型使用比较毒理基因组学数据库 (CTD) 的数据进行训练，并结合 SMOTE 技术处理类别不平衡问题，在验证集上达到了约 95% 的准确率。

## 仓库结构

*   `predict_interaction.py`: 主脚本，用于加载训练好的模型并执行预测。
*   `model_epoch_30.pt`: 训练好的 PyTorch 模型权重。
*   `scaler_SMOTE.pkl`: 用于数据归一化的 StandardScaler 对象。
*   `drug_list_all.pkl`: 包含所有药物名称的列表。
*   `use_gene_list_all.pkl`: 包含所有基因名称的列表。
*   `mesh_info.csv`: MeSH 描述符信息。
*   `save_walk_results_drug_general/`: 包含预先生成的药物随机游走结果（嵌入），已拆分为多个批处理文件以优化加载。
*   `README.md`: 本文档。
*   `Avaliable_Drug_and_Disease_Name/`:
    *   `available_diseases.txt`: 模型支持的所有疾病名称 (MeSH IDs) 列表。
    *   `available_drugs.txt`: 模型支持的所有药物名称列表。
*   `Practice_paper/`:
    *   `Practice_paper.pdf`: 相关的实践论文。

## 环境要求

要运行预测脚本，你需要安装以下 Python 包：

*   Python 3.8+
*   PyTorch
*   NumPy
*   Pandas
*   Scikit-learn

你可以使用 pip 安装这些依赖：

```bash
pip install torch numpy pandas scikit-learn
```

## 使用方法

### 1. 交互式预测

你可以直接在终端运行预测脚本来进行交互式预测。

```bash
python3 predict_interaction.py
```

脚本会加载必要的资源（这可能需要几秒钟），然后提示你输入药物名称和疾病名称。

**示例会话：**

```text
--- Prediction System Ready ---
Enter a drug name and a disease name (MESH ID) to predict interaction probability.
Type 'exit' to quit.

Drug Name (e.g., '10-nitro-oleic acid'): 10-nitro-oleic acid
Disease Name (e.g., 'Piper'): Piper
Interaction Probability: 0.0076
Prediction: No Interaction
```

**注意：** 输入的名称必须与 `available_drugs.txt` 和 `available_diseases.txt` 文件中的条目严格匹配。请参考这些文件以获取正确的拼写和格式。

### 2. 作为 Python 模块使用

你也可以将 `DDI_Predictor` 类导入到你自己的 Python 脚本中，以便将模型集成到你的工作流中。

```python
from predict_interaction import DDI_Predictor

# 初始化预测器 (只需加载一次模型和数据)
predictor = DDI_Predictor()

# 执行预测
drug_name = "Imatinib"
disease_name = "Leukemia, Myelogenous, Chronic, BCR-ABL Positive"

probability = predictor.predict(drug_name, disease_name)

print(f"{drug_name} 和 {disease_name} 之间的相互作用概率: {probability}")
```

## 数据来源

*   **药物-疾病关联:** Comparative Toxicogenomics Database (CTD).
*   **药物靶点与网络:** 源自基因相互作用网络（PPI）和随机游走算法。
*   **疾病信息:** Medical Subject Headings (MeSH).

## 模型细节

*   **架构:** 5 层全连接神经网络 (MLP)。
*   **输入维度:** 19,292。
*   **隐藏层:** 4096 -> 1024 -> 256 -> 64。
*   **激活函数:** Leaky ReLU。
*   **正则化:** Dropout (0.5), Batch Normalization。
*   **训练:** 使用 Adam 优化器和 BCEWithLogitsLoss 损失函数训练了 30 个 Epoch。

## 许可证

[MIT License]

