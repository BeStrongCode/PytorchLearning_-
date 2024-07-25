### torch.max 

#### 形式一

```
torch.max(input) 
```

功能：输出数组的最大值

#### 形式二

```
torch.max(input,dim,keepdim=False,*,out=None)
```

功能：按指定维度判断，返回数组的最大值以及最大值处的索引

参数：

input：输入数组

dim: 给定的维度

keepdim: 如果指定为True，则输出的张良数组维数和输入一直，并且除了dim维度是1，其余的维度大小和输入数组维度大小一致。如果改为False，则相当于将True的结果压缩了。两者的差别就在于是否保留dim维度



```
import torch
a = torch.arange(10).reshape(2,5)
b = torch.max(a) 
b = torch.max(a,0)
print(a)
print(b)
print(c)


输出：
# 原数组
tensor([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9]])
# 不输入dim时，返回数组的最大值
tensor(9)
# 输入dim时，返回指定维度的最大值及索引
torch.return_types.max(
# 第一个是值，按dim=0比较得到的
values=tensor([5, 6, 7, 8, 9]),
# 第二个是值对应的索引，对应维度dim=0上的索引
indices=tensor([1, 1, 1, 1, 1]))

```

keepdim为true或false的区别

```
import torch
a=torch.arange(10).reshape(2,5)
b=torch.max(a,0,True)
c=torch.max(a,0)
print(a.shape)
# 这里定为0和1都一样，值与索引具有同样的形状
print(b[0].shape)
print(c[0].shape)

输出：
只有维度不同
torch.Size([2, 5])
torch.Size([1, 5])
torch.Size([5])
```

不同dim对结果的影响，这里以input的维数是3为例，维数更多的可以类推，首先定义好数组input

```
import torch
a=torch.arange(32).reshape(2,4,4)
print(a)

tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11],
         [12, 13, 14, 15]],

        [[16, 17, 18, 19],
         [20, 21, 22, 23],
         [24, 25, 26, 27],
         [28, 29, 30, 31]]])

```

dim为0时

![在这里插入图片描述](https://img-blog.csdnimg.cn/2289b20fe8c947d6bc3de88a4029fd7a.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6KeG6KeJ6JCM5paw44CB,size_15,color_FFFFFF,t_70,g_se,x_16#pic_center)

dim为1时

![image-20240302161709429](C:\Users\RQY\AppData\Roaming\Typora\typora-user-images\image-20240302161709429.png)

dim为2时

![image-20240302161721489](C:\Users\RQY\AppData\Roaming\Typora\typora-user-images\image-20240302161721489.png)

### torch.mean

torch.mean(input,dim=None,keepdim=False,*,out=None)

input:输入张量

dim：指定计算平均值的维度。如果不指定，则计算整个张量的平均值

keepdim:指定是否保持结果张量的维度和输入张量相同。默认为False

out：可选参数，输出张量

```
import torch

x = torch.tensor([[1,2,3],
				 [4,5,6],
				 [7,8,9]], dtype=torch.float32)
mean_all = torch.mean(x)
print(mean_all) #输出，tensor(5,)

```



### tensor

#### permute

用于重新排列张量维度顺序的函数

```
import torch

x = torch.tensor([[1,2,3],[4,5,6]])
y = x.permute(1,0)
print(y)

tensor([1,4],
	   [2,5],
	   [3,6])
```

### F.binary_cross_entropy_with_logits

![image-20240305202052782](C:\Users\RQY\AppData\Roaming\Typora\typora-user-images\image-20240305202052782.png)

计算二进制交叉熵

二进制交叉熵通常用于二分类问题，其中模型的输出是一个logits张量，表示正类和负类的分数。

```
import troch
import torch.nn.functional as F

# 创建logits张量和目标标签
logits = torch.tensor([0.5,-1.0,2.0])
targets = torch.tensor([1.0,0.0,1.0])

#使用binary_cross_entropy_with_logits计算二进制交叉熵损失
loss = F.binary_cross_entropy_with_logits(logits,targets)



```

![image-20240303185037682](C:\Users\RQY\AppData\Roaming\Typora\typora-user-images\image-20240303185037682.png)

![image-20240303185104632](C:\Users\RQY\AppData\Roaming\Typora\typora-user-images\image-20240303185104632.png)

### torch.autograd.Function

用于创建自定义autograd.Function的基类

```
Examples：
class EXP(function):
	@staticmethod
	def forward(ctx,i):
		result = i.exp()
		ctx.save_for_backward(result)
		return result
	@staticmethod
	def backward(ctx, grad_output):
		result, = ctx.saved_tensors
		return grad_output * result

# Use it by calling the apply mtehod:
output = Exp.apply(input)
```

### torch.nn.BatchNorm2d

用于二维批归一化的函数，有助于加速模型收敛，提高模型的泛化能力，并减轻梯度消失或梯度爆炸的问题

```
torch.nn.BatchNorm2d(num_featuers)
norm_features #输入数据的特征
```

```
input = torch.randn(16,3,32,32)
output = nn.BatchNorm2d(3)

```

### def fuseforward()

fuseforward用于在模型推理阶段进行融合推理，融合推理是一种优化技术，旨在减少推理过程中的计算量和内存访问，从而提高推理速度。通常在训练过程中，需要使用归一化操作来提高模型的收敛速度和稳定性。但在推理阶段，我们可以通过融合推理来加速模型的推理速度，因为推理阶段不需要进行梯度计算和参数更新。



### torch.nn.functional.linear

```
torch.nn.functional.linear(input, weight, bias=None)

```

* input 形状为（N,*,in_features）的张量， *代表任意维度
* weight 形状为（out_features,in_features）的权重矩阵
* bias为偏置向量

用于计算
$$
y=xA^T+b
$$
其中，A为权重矩阵，x为输入数据，b是偏置向量，是神经网络中常见的全连接层的数学表达式





### torch.flatten()

```
torch.flatten(input,start_dim,end_dim)
input:需要被压缩的输入张量
start_dim:开始压缩的维度索引，默认为0
end_dim:结束压缩的维度
```

Example 1：

input(2,3,4,5)

input.flatten()  -> input(2* 3 * 4 * 5)

input。flatten(1,2) -> input(2,3 * 4，5)
