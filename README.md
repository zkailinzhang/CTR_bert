# adviertisement_recommendation 

#### 教育课程APP介绍
- APP教育K12课程推荐，外部广告及内课程推荐

#### 模型
- base: bGRU,deepFM
- atteGRU
- BERT,dien

#### 数据信息
- 用户信息，年龄城市等，app，onehot 
- 用户点击历史   attention gru-attention  self-attention  
- 广告信息  广告位 品牌 rnn lstm gru blstm bgru
- 上下文 时间 
- 广告特征注意与广告区别使用

#### 特征处理
- 简单的特征 嵌入  稀疏 -》 密集    batch*128 
- 用户点击历史  是个 广告id list  (广告ID嵌入矩阵 ID_num *128)  ->  batch * len * 128  -> reduce_sum  
- 广告特征，label  value 三维嵌入矩阵，经过一些复杂计算，最后batch *128  
- 广告特征 三维嵌入矩阵， 引入 gru，
- 用户点击历史 引入注意力   输入候选广告id，历史广告，计算scores ，加权求和，，（din）， 最后直接拼接
- 用户点击历史 引入注意力和次序，gru，attention，gruatte，             （dien）
- 用户点击历史 引入自注意力                                           dsin）

最后  tf.concat( 广告，用户点击历史，广告*用户点击历史（特征组合），


广告信息，这里又包括不同的域 这里又包括不同的域 可以继续onehot
这里是 类似multihot， 不同标签域下 是onehot， 不同的域对应一个嵌入矩阵，
这几个label   两层嵌入矩阵，， 不同label索引，label下不用值得索引

#### 训练方式：
- 数据都存放在hbase上，在线训练方式，第一天的数据作为训练数据，第二天的数据作为测试数据，然后第二天的数据作为训练
，保留测试结果，依次迭代下去。
- 因为每天的数据500万级别，且是cpu训练，非常耗时，，所以训练时不是所有的数据，可以支取100万的数据，测试可以全部的数据

#### 超参数：
- 学习率：0.001，0.001
- 迭代次数：1000，10000 
- 下降率：0.5，0.8 0.6  0.7  0.9  0.99
- 还有一个用tfapi指数自动调节学习率 ，单有几个参数需要设置，总训练次数 = （alldatasize / batch） * epoches（一个数据集跑完算一个epoch），两次循环，
- 天数固定20天，从六月20开始，每天的数据集 训练完 大概50000*128 ，部分训练，只设置20000，
  
#### 确定好哪个模型之后，再调优
- 逐层改变学习率
- 初始化 ，高斯，凯明，随机
- 梯度裁剪

#### 模型结构：
- base：one-hot embedding mlp softmax
- din  dien  dsin  p'n'n
- dropout，batchnorm，


#### 指标可视化
- csv excel，一个学习率不同方法 一张图，
- tf.summary train test tensorboard，   test 第二天所有轮的均值 train也可以第一天所有轮的均值  ，仅仅test 没法看欠过拟合
