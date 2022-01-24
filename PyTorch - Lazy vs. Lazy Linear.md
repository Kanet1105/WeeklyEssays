# Pytorch - Linear vs. Lazy Linear

Dependencies required to replicate the results for code written in the post :
Python 3.9.6 (64 bit)
NumPy 1.22.1
PyTorch 1.10.1
Note : All references are cited within relevant paragraphs to make it more readily accessible for viewers.

## Linear vs. Lazy Linear
Linear 모듈의 경우 input 과 output 의 사이즈를 미리 알고 있는 상태에서 weight 값들을 초기화시키는 반면 LazyLinear 모듈은 그렇지 않다. 다음 코드를 실행시켜 LazyLinear 의 동작에 대해 이해할 수 있다.

#### nn.LazyLinear()

    import torch  
    from torch import nn  
      
      
    X = torch.Tensor([[1, 2], [3, 4]])  
      
    lazy_linear = nn.LazyLinear(10)  
    print(lazy_linear.weight)  
    print(lazy_linear.bias)  
    print("=" * 50)  
      
    output1 = lazy_linear(X)  
    print(output1, output1.size())  
    print("=" * 50)  
      
    print(lazy_linear.weight, lazy_linear.weight.size())  
    print(lazy_linear.bias, lazy_linear.bias.size())

#### results

    <UninitializedParameter>
    <UninitializedParameter>
    ==================================================
    tensor([[-1.2074, -1.1112,  1.5974,  0.3237, -0.7451,  0.3171,  1.0228,  0.6705,
             -0.6540, -0.1020],
            [-1.9263, -2.6292,  3.1938,  0.3032, -0.9119,  0.5492,  2.3039,  2.3955,
             -2.4175, -1.3748]], grad_fn=<AddmmBackward0>) torch.Size([2, 10])
    ==================================================
    Parameter containing:
    tensor([[-0.0976, -0.2618],
            [-0.3858, -0.3732],
            [ 0.3823,  0.4159],
            [ 0.1103, -0.1206],
            [ 0.3450, -0.4284],
            [-0.2520,  0.3680],
            [ 0.3265,  0.3141],
            [ 0.6247,  0.2378],
            [-0.6019, -0.2799],
            [-0.6742,  0.0378]], requires_grad=True) torch.Size([10, 2])
    Parameter containing:
    tensor([-0.5861,  0.0210,  0.3833,  0.4546, -0.2333, -0.1670,  0.0682, -0.4299,
             0.5077,  0.4966], requires_grad=True) torch.Size([10])

LazyLinear 레이어의 weight, bias 값들은 입력값이 없는 상태에서는 UninitializedParameter 오브젝트로 존재하다가 입력 텐서의 사이즈에 맞춰서 weight 과 bias 값들을 초기화한다. 위 코드에서 입력 tensor 의 크기는 (2, 2) 이고 LazyLinear 는 output feature size 에 대해 하나의 정수 (10) 값만 가지고 있다가 입력 텐서에 맞춰 (2, 10) 크기의 weight 을 동적으로 초기화한다.

한번 초기화된 LazyLinear 레이어는 고정된 weight size 를 가지며 따라서 다음 코드는 런타임 에러를 발생시킨다.


    Y = torch.Tensor([[1, 2, 3], [4, 5, 6]])  
      
    output2 = lazy_linear(Y)  
    print(output2)

#### results
    RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x3 and 2x10)

이렇듯 LazyLinear 는 weight 과 bias 를 런타임에 동적으로 초기화시키는 모듈이며 한번 초기화되면 이후 고정된 weight size 를 가지는 것을 알 수 있다.

> nn.LazyLinear
> https://pytorch.org/docs/stable/generated/torch.nn.modules.lazy.LazyModuleMixin.html
> 
> nn.Linear
> https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
