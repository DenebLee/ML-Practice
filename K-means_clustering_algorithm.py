# 주어진 데이터를 k개의 클러스터로 묶는 알고리즘
# 각 클러스터와 거리 차이의 분산을 최소화하는 방식으로 동작
# 평균값이 클러스터의 중심에 위치하기 떄문에 클러스터 중심 혹은 센트로이드라고 부른다
# 작동방식
# 1. 무작위로 k개의 클러스터 중심을 정함
# 2. 각 샘플에서 가장 가까운 클러스터 중심을 찾아 해당 클러스터의 샘플로 지정
# 3. 클러스터에 속한 샘플의 평균값으로 클러스터 중심을 변경
# 4. 클러스터 주심에 변화가 없을 때까지 2번으로 돌아가 반복

#%%
#KMeans 클래스
import numpy as np
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)
# 비지도학습이므로 fit()메서드에서 타깃 데이터를 사용x
from sklearn.cluster import KMeans
km =KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)
# 군집된 결과는 KMeans 클래스 객체의 labels_ 속성에 저장

#%%
# 출력
print(km.labels_)
# %%
# 샘플의 개수 출력
print(np.unique(km.labels_, return_counts=True))

# %%
# 각 클러스터가 어떤 이미지를 나타냈는지 그림으로 출력하기 위해 간단한 유틸리티 함수 draw_fruits()사용
import matplotlib.pyplot as plt
def draw_fruits(arr, ratio=1):
    n = len(arr) # 샘플의 개수
    # 한줄에 10개씩 이미지그리지
    rows = int(np.ceil(n/10))
    # 형이 1개이면 열의 개수는 샘플의 개수
    cols = n if rows < 2 else 10
    fig , axs = plt.subplots(rows, cols, figsize=(cols*ratio, rows*ratio), squeeze=False)
    
    for i in range(rows):
        for j in range(cols):
            if i*10 + j <n: # n개 까지만 그리기
                axs[ i , j ].imshow(arr[i*10 + j ],cmap='gray_r')
            axs[i , j].axis('off')
    plt.show()
# 넘파이는 이런 불리언 배열을 사용해 원소를 선택할 수 있는데 이를 불리언 인덱싱이라고한다
# %%
draw_fruits(fruits[km.labels_==0])
# %%
draw_fruits(fruits[km.labels_==1])
# %% 
draw_fruits(fruits[km.labels_==2])

# %%
# Kmeans 클래스가 최종적으로 찾은 클러스터 중심은 cluster_centers_속성에 저장되어있다 클러스터 중심이기 때문에 이미지로 출력할려면 2차원 배열로 수정
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)
# %%
print(km.transform(fruits_2d[100:101]))
# %%
print(km.predict(fruits_2d[100:101]))
# %%
draw_fruits(fruits[100:101])
# %%
print(km.n_iter_)
# %%
# 최적의 k 찾기
# k- 평균 알로지므의 단점 중 하나는 클러스터 개수를 사전에 지정해야한다는점
# 적절한 클러스터 계수를 찾기 위한 대표적인 방법 엘보우 
# 클러스터 중심과 클러스터에 속한 샘플 사이의 거리 제곱합을 이너셔라고한다
# 이너셔는 클러스터에 속한 샘플이 얼마나 가깝게 모여 있는지를 나타내는 값
# 일반적으로 클러스터 개수가 늘어나면 클러스터 개개의 크기는 줄어들기 때문에 이너셔도 줄어듬

inertia = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)
plt.plot(range(2, 7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()

# %%
