# 中文命名实体识别

## 文件结构
```
    named_entity_recognition
    |____ ckpts                   #训练好的模型                          
    |____ datas                   #数据集
    |____ models                  #模型构建代码
    |____ data.py                 #数据集预处理
    |____ evaluate.py             #训练并评估模型的函数
    |____ evaluating.py           #用于评价模型，计算每个标签的精确率，召回率，F1分数和混淆矩阵等
    |____ utils.py                 #一些工具类
    |____ main.py                 #训练并评估模型
    |____ predict.py              #预测代码，使用训练好的模型识别一句话中的命名实体
    |____ test.py                 #加载并评估已有模型
```
## 环境    
numpy==1.20.2

sklearn_crfsuite==0.3.6

torch==1.8.1+cu102
       
## 数据集
MSRA微软亚洲研究院数据集

该数据集位于项目目录下的`datas`文件夹里。

## 快速开始
```
#安装依赖项：
pip install -r requirements.txt

#训练并评估：
python main.py --model hmm/crf/bilstm/bilstm_crf(四个参数选择一个)

#加载已有模型并评估:
python test.py --model hmm/crf/bilstm/bilstm_crf(四个参数选择一个)

#使用已有模型预测：
python predict.py --model hmm/crf/bilstm/bilstm_crf(四个参数选择一个)
```

## 运行示例
```
>python main.py --model hmm

读取数据...
正在训练评估HMM模型...
           precision    recall  f1-score   support
    B-PER     0.8356    0.7187    0.7728      1973
    I-PER     0.7903    0.8076    0.7989      3851
    B-LOC     0.7564    0.6378    0.6921      2877
    I-LOC     0.6669    0.5330    0.5925      4394
        O     0.9720    0.9747    0.9734    152505
    I-ORG     0.5433    0.6547    0.5938      5670
    B-ORG     0.4814    0.4568    0.4688      1331
avg/total     0.9371    0.9367    0.9364    172601

Confusion Matrix:
          B-PER   I-PER   B-LOC   I-LOC       O   I-ORG   B-ORG
  B-PER    1418     200      60       8     270      11       6
  I-PER      23    3110      10      67     616      23       2
  B-LOC      91      18    1835     114     485     114     220
  I-LOC       8     257      91    2342     964     730       2
      O     147     325     258     675  148652    2139     309
  I-ORG       3      24      18     292    1505    3712     116
  B-ORG       7       1     154      14     444     103     608

>python predict.py --model hmm

请输入要识别的句子：陈明亮在北京怀柔区的国科大读书

预测标签: ['B-PER', 'I-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
人名: ['陈明亮']
地名: ['北京']
组织机构名: []
```
## 运行结果
可通过修改test.py中第17行的REMOVE_O参数来控制评价模型性能时是否考虑O标签。
为True代表不考虑O标签，为False代表考虑O标签。

使用main.py训练并评估模型时，默认考虑O标签。

下面是四个模型预在考虑O标签的情况下的准确率：

|      | HMM    | CRF    | BiLSTM | BiLSTM+CRF | 
| ---- | ------ | ------ | ------ | ---------- |     
| 准确率  | 93.45% | 97.23% | 96.79% | 97.40%     |
| 召回率  | 93.45% | 97.33% | 96.86% | 97.48%     | 
| F1分数 | 93.41% | 97.24% | 96.81% | 97.41%     |

下面是四个模型预在不考虑O标签的情况下的准确率：

|      | HMM    | CRF    | BiLSTM | BiLSTM+CRF | 
| ---- | ------ | ------ | ------ | ---------- |   
| 准确率  | 81.24% | 92.68% | 91.81% | 94.00%     |
| 召回率  | 63.06% | 80.24% | 77.00% | 83.22%     |   
| F1分数 | 70.74% | 85.90% | 83.70% | 88.25%     |



















