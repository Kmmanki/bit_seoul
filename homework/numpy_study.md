# numpy공부

## numpy 기초 함수
arr = numpy.array([1,2,3,4])
    list를 array 타입으로 변환
    arr.shape      => 형태와 크기를 알 수 있다.
    arr.dtype => 타입을 알 수 있다.

    numpy.array([1,2,3,4,5,'a']) 를 넣게되면 모두 문자열로 변환

    arr4 = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    arr4.shape => (4,3) 3크기의 배열이 총 4개 

    numpy.zeros(10) => numpy.array([0,0,0,0,0,0,0,0,0,0])
    numpy.zeros(3*2) => numpy.array([[0,0],
                                        [0,0],
                                        [0,0]]) => zros 말고 ones도 있다

    numpy.arange(1,10) => numpy.array(list(range(1,10)))

    array 연산 array는 각 같은 위치에서 연산을 한다 행열연산x
    즉 a[0] + b[0]

    크기가 다른 array간 연산을 하면 작은 array가 확장된다.
    즉 
    [
        [1,2,3],
        [4,5,6]
    ] 
     +
     [10,10,10]을 하면 
     10array는 
     [
         [10,10,10],
         [10,10,10]
     ]
     이 되어 결과는
     [
         [11,12,13],
         [14,15,16]
     ]
     
     ps. 집 텐서플로우 설치
     
     ![집 텐서플로우 설치](https://user-images.githubusercontent.com/50133267/98552409-c602f280-22e1-11eb-9f4a-35767f8fa309.PNG)
