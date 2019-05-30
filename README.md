# 【毕设】药物关系抽取

### 数据集来源

===>[SemEval 2013 任务9.2](https://www.cs.york.ac.uk/semeval-2013/task9/)

### 毕设创新
- 提出双向GRU+CNN模型（one stage)
- 针对数据分布不均匀，通过集成学习方式过滤负例。简单快速，数据预处理仅需使用字典序，无需抽取特征。(two stage)
- 尝试使用多任务学习的方式，性能优于前两者(multi task)

### 数据预处理参考
> Huang D., Jiang Z., Zou L., Li L.
Drug–drug interaction extraction from biomedical literature using support vector machine and long short term memory networks
Information Sciences, Volumes 415-416, 2017

