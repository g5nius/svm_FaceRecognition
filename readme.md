### 人脸识别

###### 任务概述：

人脸识别，是基于人的脸部特征信息进行身份识别的一种生物识别技术。用摄像机或摄像头采集含有人脸的图像或视频流，并自动在图像中检测和跟踪人脸，进而对检测到的人脸进行脸部识别的一系列相关技术，通常也叫做人像识别、面部识别。人脸识别系统的研究始于 20 世纪 60 年代， 80 年代后随着计算机技术和光学成像技术的发展得到提高，而真正进入初级的应用阶段则在 90 年后期，并且以美国、德国和日本的技术实现为主；人脸识别系统成功的关键在于是否拥有尖端的核心算法，并使识别结果具有实用化的识别率和识别速度；“人脸识别系统”集成了人工智能、机器识别、机器学习、模型理论、专家系统、视频图像处理等多种专业技术，同时需结合中间值处理的理论与实现，是生物特征识别的最新应用，其核心技术的实现，展现了弱人工智能向强人工智能的转化。  

###### 数据集描述：

哥伦比亚大学公众人物脸部数据库， PubFig Dataset 是一个大型人脸数据集，主要用于人脸识别和身份鉴定，不同于大多数现有面部数据集，这些图像是在主体完全不受控制的情况下拍摄的，因此不同图像中姿势、光照、表情、场景、相机、成像条件和参数存在较大差异，该数据集类似于 UMass-Amherst 创建的 LFW 数据集。该数据集由哥伦比亚大学于 2009 年发布，相关论文有《Attribute and Simile Classifiers for Face Verification》。 该数据集  
包含从网络上收集的 13000 多张人脸图像。每张脸上都标有照片中人物的名字。 1680 名照片中的人在数据集中有两张或多张不同的照片。  
此数据集可从以下网址获取：  
http://vis-www.cs.umass.edu/lfw/

###### 总体思路和方法：

本次主要针对机器学习课程设计，要求设计过程中使用传统的机器学习方法而非深度的神经网络，我们将人脸识别任务看成了一种分类问题：给定一张人脸，判断它属于那个类别，即哪个人。

数据处理阶段，经过统计分析，我们发现1680个类别中有两张以上图片，其余4070个类别中仅有一张图片，于是我们使用数据增强和删减的方法将所有类别的图片增加或减少到5张，我们使用的数据增强方法有旋转、反转、饱和度亮度改变这几种方法，这样的处理有极大的弊端稍后将会讨论到。除了数据增强，我们还对图片中的人脸特征进行了提取，用了pca、dlit人脸特征点检测和insightface框架中的神经网络方法，获得了不同的效果。

模型训练和预测阶段，对于svm仅调用sklearn中的方法即可，

###### 过程和原理：

![支持向量机（SVM）——原理篇](https://picx.zhimg.com/v2-197913c461c1953c30b804b4a7eddfcc_720w.jpg?source=172ae18b)

支持向量机（support vector machines, SVM）是一种二分类模型，它的基本模型是定义在特征空间上的**间隔最大的线性分类器**，间隔最大使它有别于感知机；SVM还包括**核技巧**，这使它成为实质上的非线性分类器。SVM的的学习策略就是间隔最大化，可形式化为一个求解凸二次规划的问题，也等价于正则化的合页损失函数的最小化问题。SVM的的学习算法就是求解凸二次规划的最优化算法。

核函数

- 线性核函数   
  κ(x,xi)=x⋅xi  
  线性核，主要用于线性可分的情况，我们可以看到特征空间到输入空间的维度是一样的，其参数少速度快，对于线性可分数据，其分类效果很理想

- 多项式核函数  
  κ(x,xi)=((x⋅xi)+1)d  
  多项式核函数可以实现将低维的输入空间映射到高纬的特征空间，但是多项式核函数的参数多，当多项式的阶数比较高的时候，核矩阵的元素值将趋于无穷大或者无穷小，计算复杂度会大到无法计算。

- 高斯核函数  
  κ(x,xi)=exp(−||x−xi||2δ2)  
  高斯径向基函数是一种局部性强的核函数，其可以将一个样本映射到一个更高维的空间内，该核函数是应用最广的一个，无论大样本还是小样本都有比较好的性能，而且其相对于多项式核函数参数要少，因此大多数情况下在不知道用什么核函数的时候，优先使用高斯核函数。

如果特征的数量大到和样本数量差不多，则选用LR或者线性核的SVM；如果特征的数量小，样本的数量正常，则选用SVM+高斯核函数；如果特征的数量小，而样本的数量很大，则需要手工添加一些特征从而变成第一种情况。

多分类方法

- a.一对多法（one-versus-rest,简称1-v-r SVMs）。训练时依次把某个类别的样本归为一类,其他剩余的样本归为另一类，这样k个类别的样本就构造出了k个SVM。分类时将未知样本分类为具有最大分类函数值的那类。

- b.一对一法（one-versus-one,简称1-v-1 SVMs）。其做法是在任意两类样本之间设计一个SVM，因此k个类别的样本就需要设计k(k-1)/2个SVM。当对一个未知样本进行分类时，最后得票最多的类别即为该未知样本的类别。Libsvm中的多类分类就是根据这个方法实现的。

PCA:

Insightface:

![](https://pic3.zhimg.com/v2-243b0fbf86e2dad7557fd286328ead7e_r.jpg)

以减少提取出特征的类内间距和增大类间距离为目的，设计出了aceface作为loss函数

dlib特征点检测:

![](https://img-blog.csdnimg.cn/20200605170409305.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ0NDMxNjkw,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200606121038636.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ0NDMxNjkw,size_16,color_FFFFFF,t_70)

###### 总结和有待改进的地方：

结果中屡次出现了在测试集上acc为1的情况，经过讨论分析我们认为在问题出在数据处理阶段，我们在数据集中增加了大量的非常相似的图片，这会导致训练出的模型过拟合严重。我们认为这样的数据处理非常不合理。insightface的训练给了我们一个非常棒的思路，他的训练每次输入两张图片，标签是这两张图片中的人是不是同一个，这样就不用要求每个类别都有多张图片，具体的实现方式一是还没想到，但这绝对是一个好方法。

经过查阅详细资料，现实中的人脸检测流程如下，

首先使用如上方法训练出一个提取人脸特征的模型，利用这个模型建立人脸特征数据库，具体过程是每次给定一张人脸图片，首先使用mtcnn或者retinaface检测出人脸，经过相似性变换得到对齐的人脸图像大小固定，然后使用模型提取得到人脸特征。真实场景中拍摄一张图片，经过上述步骤得到人脸特征，在和数据库中的进行比对，从而达到识别人脸的目的。
