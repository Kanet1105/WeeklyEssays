# PyTorch - Dataset, DataLoader and Multiprocessing
**Dependencies required to replicate the results for code written in the post :**

1.  Python 3.9.6 (64 bit)
2.  NumPy 1.22.1
3.  PyTorch 1.10.1

**Note : All references are cited within relevant paragraphs to make it more readily accessible for viewers.**

## Dataset

PyTorch 는 아래 2가지 타입의 데이터셋을 지원한다.

 - map-style :  `__getitem__()` 과 `__len__()` 함수를 구현하는 Dataset
 - iterable-style : `__iter__()` 를 구현하며 dataset 의 sample 에 대한 iterator 를 반환하는 Dataset (서버나 db로부터 데이터를 byte-stream 으로 받아올 때 사용)

> 공식 문서
> 
> https://pytorch.org/docs/stable/data.html?highlight=torch%20utils%20data#map-style-datasets
> https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset


본문에서 다룰 내용은 map-style Dataset 이며 다음과 같은 포맷으로 사용한다.

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, *args):
        """
        data 를 불러오거나 class attribute 들을 정의하는 부분
        file read 나 
        """

    def __len__(self):
        """
        data 의 요소 수를 반환
        """
    
    def __getitem__(self, index):
        """
        data 에서 index 번째 요소를 반환
        """
```

위와 같이 기본 포맷에서 다루고자 하는 데이터셋에 대해서 유연하게 작성하는 것이 가장 중요하고 Dataset 의 경우
`__getitem__` 을 구현하는 Iterable 객체이므로 다음과 같이 for() 나 map() 안에서 사용할 수 있다.

> Iterator, Iterable, Iteration 에 관해서
https://stackoverflow.com/questions/9884132/what-exactly-are-iterator-iterable-and-iteration

```python
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        return self.data[index]


A = CustomDataset([1, 2, 3, 4, 5])
it = iter(A)
print(next(it))
print(next(it))

try_list_comprehension = [i for i in A]
print(try_list_comprehension)
    
try_map = map(lambda x: x + 1, A)
for i in try_map:
    print(i)
```

```
---------------------------------------------------------------------------
1
2
[1, 2, 3, 4, 5]
2
3
4
5
6
```
하지만 Dataset 에 대해서 직접 custom iterator 클래스를 만들어 쓰거나 하는 방식을 사용하는 대신 `__iter__` 를 구현하는 DataLoader 클래스를 통해 데이터를 가져온다. 

## DataLoader

DataLoader 는 내부적으로 multiprocessing context 를 가지고 있어 CPU 를 utilize 하는 Python 의 멀티프로세싱을 사용할 것인지 CUDA 를 통한 GPU 멀티프로세싱을 할 것인지를 정할 수 있고 num_worker 인자를 통해 몇 개의 프로세스를 사용할 것인지 정해줄 수 있다. `torch.utils.data.dataloader.py` 모듈을 살펴보면

```python
# torch.utils.data.dataloader.py
def __iter__(self) -> '_BaseDataLoaderIter':  
    # When using a single worker the returned iterator should be  
    # created everytime to avoid reseting its state 
    # However, in the case of a multiple workers iterator 
    # the iterator is only created once in the lifetime of the 
    # DataLoader object so that workers can be reused  
    if self.persistent_workers and self.num_workers > 0:  
        if self._iterator is None:  
            self._iterator = self._get_iterator()  
        else:  
            self._iterator._reset(self)  
        return self._iterator  
    else:  
        return self._get_iterator()

def _get_iterator(self) -> '_BaseDataLoaderIter':  
    if self.num_workers == 0:  
        return _SingleProcessDataLoaderIter(self)  
    else:  
        self.check_worker_number_rationality()  
        return _MultiProcessingDataLoaderIter(self)
```

위와 같은 iterator 를 반환하는 부분에 대해서 알 수 있는데 단일 프로세스를 사용할 때, 즉 num_worker 인자에 아무것도 전달하지 않는다면 기본적으로 0의 값을 가지며 이 때는 상태를 저장할 필요가 없기 때문에 Dataset 객체에서 `__getitem__` 이 호출될 때마다 매번 새로운 Iterator 객체를 반환한다.

반면 num_worker 에 **1 이상의 값**을 전달하게 되면 무조건 하나 이상의 프로세스를 spawn 하기 때문에 멀티프로세싱을 위한 Iterator 객체를 생성해서 DataLoader 에 저장하고 이 객체를 반환하게 된다.

 - `_SingleProcessDataLoaderIter(self)`
 - `_MultiProcessingDataLoaderIter(self)`

Iterator 클래스 모두 다 호출하는 DataLoader 클래스 인스턴스를 인자로 넘겨서 초기화하게 된다. 

## Data Flow in Multiprocessing Context

multiprocessing 환경에서의 데이터 흐름은 다음 다이어그램처럼 표현된다.

![enter image description here](https://github.com/Kanet1105/WeeklyEssays/blob/main/images/dataflow.png)

num_worker 인자에 들어가는 process 갯수만큼 Iterator 클래스에서 초기화되고 여기서 인덱스를 큐에 넣어서 각 프로세스에 전달한다. 그리고 데이터 큐를 통해 나온 값들을 받고 리턴해준다. 이제 다시 메인 프로세스로 데이터가 모이게 되고 여기서 다음 단계인 모델에 data 를 feed 하게 된다.

#### Queue 를 쓰는 이유

Index 만 넘겨주게 되는 이유는 Python 에서의 multiprocessing 의 경우 다른 언어의 multithread 와 다르게 데몬 프로세스를 생성하기 때문에 상태를 공유할 수 없다. Index 값을 queue 를 통해 넘겨주지 않고 하위 프로세스에서 정하게 되면 프로세스들끼리 서로 어떤 인덱스를 작업하는지 모르기 때문에 병렬적으로 처리되는 이점이 있다 해도 똑같은 인덱스에 대해 작업을 하는 경우가 생길 수 있다. 하지만 queue 를 통해 index 를 넘겨준다면 생성된 프로세스들은 공유하는 queue 객체에 대해 blocking call 을 하며 들어오는 인덱스 값에 해당하는 data 에 대해서만 처리하므로 이중 작업을 하지 않는다. Queue 는 일종의 동기화 역할을 해준다.

#### collate_fn 에 들어가는 Callable 객체

만약 collate_fn 에 들어가는 기능들을 구현하기 위해 클래스를 써서 클래스 method 를 collate_fn 의 인자로 넘겨주게 되면 queue 객체는 클래스 인스턴스를 pickle 로 serialize 하지 못하기 때문에 오류가 생긴다. 그래서 collate_fn 에 인자로 넘겨주는 함수는 반드시 클래스 method 가 아닌 메인 함수에서 정의된 함수를 넘겨주어야 문제가 발생하지 않는다.

#### Pseudo Code

구조를 이해하기 위해 비슷하게 pseudo coding 을 해봤다. 항상 시간적 여유가 있을 때 어떤 프레임워크에서 제공하는 기능을 사용하기 위해 그 기능을 흉내내는 코드를 짜서 이해해보면 괴롭고 힘들지만 뭔가 배운 것을 summarize 하는 느낌이 들어서 나름 나쁘지 않은 것 같다.

```python
class CustomDataset:
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return self.data[index]

    
class CustomDataLoader:
    def __init__(self, dataset, callback=None):
        self.dataset = dataset.data
        self.iterator = None
        self.callback = callback
        
    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.register_iterator()
        return self.iterator
    
    def register_iterator(self):
        return CustomIterator(self)
    

class CustomIterator:
    def __init__(self, dataloader):
        self.dataset = dataloader.dataset
        self.callback = dataloader.callback
        self.index = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration
        data = self.dataset[self.index]
        self.index += 1
        
        if self.callback:
            result = self.callback(data)
            return result
        return data

    
def custom_callback(data):
    print("{value} 입니다.".format(value=data))
    return data


my_dataset = CustomDataset([1, 2, 3, 4, 5])
my_dataloader = CustomDataLoader(my_dataset, custom_callback)
for i in my_dataloader:
    print(i)
```

```
---------------------------------------------------------------------------
1 입니다.
1
2 입니다.
2
3 입니다.
3
4 입니다.
4
5 입니다.
5
```
> Sending a class method over a multiprocessing.Queue
> 
> https://stackoverflow.com/questions/44185770/call-multiprocessing-in-class-method-python
