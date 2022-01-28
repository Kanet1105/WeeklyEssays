# Pytorch - view() vs reshape()

**Dependencies required to replicate the results for code written in the post :**

 1. Python 3.9.6 (64 bit)
 2. NumPy 1.22.1
 3. PyTorch 1.10.1

**Note : All references are cited within relevant paragraphs to make it more readily accessible for viewers.**

## torch.view()

view 는 원본 텐서의 모양을 변경시켜주는 함수이다. 하지만 view를 쓰기 위해선 먼저 메모리가 contiguous 하다는 개념이 중요하다. Tensor 객체는 Contiguous 한 메모리를 갖는데 PyTorch 에서의 contiguous 한 메모리는 반드시 **행 메모리** 이다. 

만약 `A = torch.arange(1, 10)` 를 실행하면 다음과 같은 Tensor 객체를 생성하게 된다.

![enter image description here](https://github.com/Kanet1105/WeeklyEssays/blob/main/images/contiguity_01.png)

위와 같이 1번 배열에서 2번 배열까지의 메모리 상 거리를 stride 라고 하는데 시작 주소 [i] 에서 [i + 1] 까지, [i + 1] 에서 [i + 2] 까지의 stride 가 항상 일정해야만 contiguous 한 배열이라고 할 수 있다. Tensor 객체를 생성하게 되면 항상 이렇게 contiguous 한 배열 객체를 생성해서 반환한다. Tensor 객체가 contiguous 한지 확인하기 위해 is_contiguous() 함수를 쓸 수 있다.

```python
A.is_contiguous()
```

```
---------------------------------------------------------------------------
True
```

이렇게 생성된 Tensor 를 (3, 3) 모양으로 변화시키고 B 라는 변수에 할당하면

```python
B = A.view(3, 3)
print(B)
```

```
---------------------------------------------------------------------------
tensor([[1, 2, 3], 
	[4, 5, 6], 
	[7, 8, 9]])
```
           

위와 같은 결과를 얻을 수 있고 B 텐서 객체는 A 를 참초하므로 아래의 코드와 같이 A 와 B 에서 어떤 값을 수정해도 같이 변하게 된다. 

```python
A = torch.arange(1, 10)
B = A.view(3, 3)

A[0].fill_(55)
B[2][2].fill_(33)

print(A)
print(B)

print(A.is_contiguous())
print(B.is_contiguous())
```

```
---------------------------------------------------------------------------
tensor([55,  2,  3,  4,  5,  6,  7,  8, 33])
tensor([[55,  2,  3],
    [ 4,  5,  6],
    [ 7,  8, 33]])
True
True
```
    

하지만 PyTorch 는 메모리의 위치를 바꾸게 되는 (non-contiguous) 연산을 하게 되는 경우가 있다. torch.swapdim 의 자매품인 torch.transpose() 가 그런 경우인데 다음과 같은 경우는 view() 를 사용할 수 없게 된다.

```python
A = torch.arange(1, 10)
B = A.view(3, 3)
C = B.t()    # torch.transpose()
D = C.view(9, 1)
```

```
---------------------------------------------------------------------------    
RuntimeError                              Traceback (most recent call last)
Input In [31], in <module>
2 B = A.view(3, 3)
3 C = B.t()
----> 4 D = C.view(9,  1)
    
    RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```

위와 같이 런타임에러가 발생하며 reshape() 을 쓰라고 권한다. view 는 반환하는 참조 텐서에 대해서 메모리 구조를 contiguous 하게 유지하고 모양만 바꿔주기 때문에

```python
B = A.view(3, 3)
```

를 하더라도 아래와 같이 A 를 참조하는 B Tensor 객체의 메모리는 contiguous 하게 유지된다.

![enter image description here](https://github.com/Kanet1105/WeeklyEssays/blob/main/images/contiguity_02.png)

반면에 A 를 참조하는 B 객체에 대해 transpose() 같은 non-contiguous 연산을 적용하게 되면 C Tensor 는 다음과 같이 non-contiguous 해지기 때문에 C 에 대해서 view() 함수를 사용할 수 없게 된다.

```python
C = B.t()    # torch.transpose()
```

![enter image description here](https://github.com/Kanet1105/WeeklyEssays/blob/main/images/contiguity_03.png)

```python
A = torch.arange(1, 10)
B = A.view(3, 3)
C = B.t()    # torch.transpose()

print(A.is_contiguous())
print(B.is_contiguous())
print(C.is_contiguous())
```

```
---------------------------------------------------------------------------
True
True
False
```

C = 는 contiguous 하진 않지만 A, B, C 모두 같은 값을 참조하기 때문에 어떻게 결과를 바꿔도 A, B, C 모두에게 적용되는 것을 볼 수 있다.

```python
A = torch.arange(1, 10)
B = A.view(3, 3)
C = B.t()    # torch.transpose()

A[0].fill_(11)
B[1, 0].fill_(22)
C[0, 2].fill_(33)

print(A)
print(B)
print(C)
```

```
---------------------------------------------------------------------------
tensor([11,  2,  3, 22,  5,  6, 33,  8,  9])
tensor([[11,  2,  3],
	[22,  5,  6],
	[33,  8,  9]])
tensor([[11, 22, 33],
	[ 2,  5,  8],
	[ 3,  6,  9]])
```

## torch.reshape()

view 함수가 반드시 contiguous 한 텐서 객체에 대해서만 적용되는 반면 reshape() 은 그렇지 않다. 위에서 view() 를 적용했을 때 에러가 나는 부분을 인터프리터의 조언에 따라 reshape() 으로 바꾸면 잘 실행되는 것을 알 수 있다. 하지만 reshape() 은 더이상 참조를 하지 않고 원래 배열을 복사해서 D 에 할당하기 때문에 A, B, C 와 D 는 서로 다른 곳을 참조한다. 

```python
A = torch.arange(1, 10)
B = A.view(3, 3)
C = B.t()    # torch.transpose()
D = C.reshape(1, 9)

A[0].fill_(11)
B[1, 0].fill_(22)
C[0, 2].fill_(33)
D[0, -1].fill_(99)

print(A)
print(B)
print(C)
print(D)
```

```
---------------------------------------------------------------------------
tensor([11,  2,  3, 22,  5,  6, 33,  8,  9])
tensor([[11,  2,  3],
	[22,  5,  6],
	[33,  8,  9]])
tensor([[11, 22, 33],
	[ 2,  5,  8],
	[ 3,  6,  9]])
tensor([[ 1,  4,  7,  2,  5,  8,  3,  6, 99]])
```

이렇게 D Tensor 는 A, B, C 와는 별개의 Tensor 객체가 됐다. Pytorch 의 기본적인 연산 단위는 Tensor 객체이고 Tensor 의 계산을 할 때에 프레임워크에서 제공하는 모듈들을 사용하게 되는데 이럴 때 가장 중요한 것이 바로 연산의 결과가 contiguous 함을 보장하느냐인 것을 알 수 있었다. 

## 그래서 어떤 것을 써야할까
Trade-off 가 있다고 생각한다. view() 를 사용해서 메모리를 절약하고 runtime 에서 발생하는 error 처리를 따로 해주느냐 아니면 메모리 누수를 최소화하는 코드를 짤 자신이 있다면 reshape() 을 사용하는 것이 바람직하지 않을까 싶다. 개인적으로 정말 바람직한 방법이라고 생각하는 것이 있다면 PyTorch 의 non-contiguous 한 연산 결과를 반환하는 함수들을 다 정리한 다음 상황에 맞춰서 view() 와 reshape() 을 번갈아 가면서 쓰는 것이 가장 좋다고 본다.

> 공식 문서
> https://pytorch.org/docs/stable/generated/torch.Tensor.view.html?highlight=torch%20view#torch.Tensor.view
> 
> contiguous array 에 대한 설명
> https://stackoverflow.com/questions/26998223/what-is-the-difference-between-contiguous-and-non-contiguous-arrays/26999092#26999092
