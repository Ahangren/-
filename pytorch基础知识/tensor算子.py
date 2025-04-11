import torch
import numpy as np

tens=torch.tensor([1,2.1,3])
# 判断是否为tensor数
# print(torch.is_tensor(tens))
# # 判断是否是pytorch储存对象
# print(torch.is_storage(tens))
# # 判断是否为复杂数据类型
# print(torch.is_complex(tens))
#
# # 判断是否是浮点数据类型
# print(torch.is_floating_point(tens))
# # 设置默认的浮点数类型
# torch.set_default_dtype(torch.float32)
# 设置默认的device
# torch.set_default_device('cpu')
# 创建稀疏张量
indices = torch.tensor([[0, 1, 2],  # 第0维坐标
                        [2, 0, 1],
                        [2,1,0]])
values=torch.tensor([3,4,5])
sparse=torch.sparse_coo_tensor(indices,values)

# print(sparse)

indices = torch.tensor([[0, 1, 2],  # 第0维坐标
                        [2, 0, 1]]) # 第1维坐标
values = torch.tensor([3, 4, 5])    # 非零值
shape = (3, 3)                      # 张量形状

sparse_tensor = torch.sparse_coo_tensor(indices, values)
# print(sparse_tensor)

crow_indices=torch.tensor([0,2,3,6])
col_indices=torch.tensor([0,3,2,0,1,3])
values=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
shape=(3,4)

csr=torch.sparse_csr_tensor(values=values,crow_indices=crow_indices,col_indices=col_indices,size=shape)
# print(csr)
arr1=np.array([1,2,3,4])
arr=torch.asarray(arr1)
# print(type(arr))
tens2=torch.tensor([1,2,3,4])
tens1=torch.as_tensor(tens2)

tens1[1]=13
# print(tens2)
# print(tens1)

x=torch.tensor([[1,2,3],[4,5,6]])
xt=torch.as_strided(x,size=(3,2),stride=(1,3))
# print(xt)

# 高效读取文件中的内容
data=np.random.rand(100).astype(np.float32)
data.tofile('tensor_data.bin')
x1=torch.from_file('tensor_data.bin',dtype=torch.float32,size=100)
# print(x1[:5])

#将来自外部库的张量转换为 .torch.Tensor,共享数据的
arr2=np.arange(10,dtype=np.float32).reshape(2,5)
torch_tensor=torch.from_dlpack(arr2.__dlpack__())

arr2[1]=100
# print(arr2)
# print(torch_tensor)

print(torch.full((2,3),5))
print(torch.full_like(torch.tensor([2,3,4,5]),10))

