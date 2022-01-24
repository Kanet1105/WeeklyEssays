# PyTorch - Types and Basic Operations

**Dependencies required to replicate the results for code written in the post :**

 1. Python 3.9.6 (64 bit)
 2. NumPy 1.22.1
 3. PyTorch 1.10.1

**Note : All references are cited within relevant paragraphs to make it more readily accessible for viewers.**

## 1. Difference Between torch.tensor vs. torch.Tensor

공식 문서의 내용을 Paraphrase 하자면 다음과 같다.

 - a torch.tensor is a function which returns a Tensor object 
 - a torch.Tensor is a base class for tensor object which all tensor objects inherit from

torch.tensor의 경우 data 인자를 반드시 필요로 하는 데 비해 torch.Tensor의 경우 빈 텐서 오브젝트를 생성할 수 있다.

#### torch.Tensor()

    import numpy as np
    import torch

	# Tensor 는 dtype 을 따로 지정해주지 않는다면 torch.float32 의 Tensor object를 반환
    tensor1 = torch.Tensor()
    print(tensor1, tensor1.dtype)
    
    # torch.float32 로 자동 캐스팅
    tensor2 = torch.Tensor([1, 2, 3])
    print(tensor2, tensor2.dtype)
    
#### results

    tensor([]) torch.float32
    tensor([1., 2., 3.]) torch.float32
    
> PyTorch 의 데이터 타입
> https://pytorch.org/docs/stable/tensors.html?highlight=torch%20tensor#torch.Tensor

---

#### torch.tensor()

    # torch.tensor 는 data 인자를 넣어야 함
    tensor = torch.tensor([1, 2, 3])
    print(tensor, tensor.dtype)

#### results

    tensor([1, 2, 3]) torch.int64

## 2. Caveats

 - torch.tensor() 는 데이터를 항상 복사한다. 
 - 만약 NumPy ndarray 를 복사가 아닌 참조 형태로 텐서로 만들고 싶다면 torch.as_tensor() 함수를 사용해야 한다.

#### copy data

    def get_address(variable):
	    print(variable, hex(id(variable)))

    array = np.array([1, 2, 3, 4], dtype=np.float32)  
    tensor = torch.tensor(array)  
      
    array[0] = 5.0  
    print(array)  
    print(tensor)

#### results

    [5. 2. 3. 4.]
    tensor([1., 2., 3., 4.])

array[0] 의 값을 5.0 으로 바꿔도 tensor[0] 의 값이 바뀌지 않는다. 같은 주소를 참조하고 있지만 신기하게도 아래 코드를 실행시켜보면 주소값이 다르게 나오는 것을 확인할 수 있다.

#### memory address

    def get_address(variable):  
        return hex(id(variable))  
      
      
    array = np.array([1, 2, 3, 4], dtype=np.float32)  
    tensor = torch.as_tensor(array)  
      
    address_a0 = get_address(array[0])  
    address_t0 = get_address(tensor[0])  
    print(address_a0) 
    print(address_t0)

#### results

    0x25060b42b70
    0x250607de720

개인적인 생각으로는 연산 효율을 높이기 위해 contiguous 한 메모리 구조를 만들기 위해서 array 변수의 각 인덱스가 가리키는 주소값을 연속적으로 저장한 주소의 값을 value 자체가 가진 주소 대신 반환하기 때문에 그런 것 같다. 아래와 같이

- &tensor[0] = &array[0] 을 저장하는 포인터의 주소
- &tensor[1] = &array[1] 을 저장하는 포인터의 주소
- ... 
- &tensor[-1] = &array[-1] 을 저장하는 포인터의 주소
 
들을 저장하는 torch.Tensor 오브젝트의 주소는 새로 contiguous 하게 할당되고 *get_address()* 함수는 array[0] 의 주소와 tensor[0] 의 주소를 반환하기 때문에 둘은 차이가 있지만 *print(tensor[0])* 또는 *print([0])* 을 실행할 경우 이런 indirection 을 전부 dereference 해서 값만 출력하기 때문에 차이가 없는 것 같다. Tensor 오브젝트는 포인터 타입 벡터가 아닐까 추측해본다.

---

#### torch.as_tensor()

    array = np.array([1, 2, 3, 4], dtype=np.float32)  
    tensor = torch.as_tensor(array)  
      
    array[0] = 5.0  
    print(array)  
    print(tensor)

#### results

    [5. 2. 3. 4.]
    tensor([5., 2., 3., 4.])

같은 메모리 주소를 참조한다.

---

#### copy a tensor as an input to another tensor

    array = np.array([1, 2, 3, 4], dtype=np.float32)  
    tensor1 = torch.tensor(array)  
    tensor2 = tensor1.clone()  
      
    tensor1[0] = 5.0  
    print(tensor1, tensor2)

#### results

    tensor([5., 2., 3., 4.]) tensor([1., 2., 3., 4.])

---

#### shares the same memory space between two tensors

    array = np.array([1, 2, 3, 4], dtype=np.float32)  
    tensor1 = torch.tensor(array)  
    tensor2 = tensor1.detach()  
      
    tensor1[0] = 5.0  
    print(tensor1, tensor2)

#### results

    tensor([5., 2., 3., 4.]) tensor([5., 2., 3., 4.])

*clone() 과 detach() 는 leaf variable 과 연관이 있으므로 이에 대해서 추후에 다루도록 하자.*

> torch.tensor() 문서
> https://pytorch.org/docs/stable/generated/torch.tensor.html?highlight=torch%20tensor#torch.tensor
> 
> leaf variable 에 대해서
> https://discuss.pytorch.org/t/leaf-variable-was-used-in-an-inplace-operation/308
> 
> View 와 Reshape
> https://inmoonlight.github.io/2021/03/03/PyTorch-view-transpose-reshape/


## 3. Basic Operation

기본 사칙연산 함수들은 다음과 같으며 모두 element-wise operation 이다.

 - torch.add()
 - torch.sub()
 - torch.mul()
 - torch.div()

#### element-wise operations

    A = torch.Tensor([3])  
    B = torch.Tensor([7])  
    C = torch.Tensor([2])  
    D = torch.Tensor([5])  
    E = torch.Tensor([10])  
      
    output = (A + B) * 2 - 5 / 10  
    print(output)  
      
    addition = torch.add(A, B)  
    multiplication = torch.mul(addition, C)  
    division = torch.div(D, E)  
    subtraction = torch.sub(multiplication, division)  
    print(subtraction)

#### results

    tensor([19.5000])
    tensor([19.5000])

사실상 연산자를 이용한 계산과 차이는 없고 오히려 계산 프로세스가 functional 해지고 코드가 길어졌다. 아마도 연산자를 사용해서 값을 계산하다 보면 사용자가 고려하지 못한 이슈들에 대해서 런타임에서 문제가 발생할 여지가 있는데 이를 functional 하게 처리함으로서 코드 길이와 안전성의 trade-off 가 생긴 것 같다.
