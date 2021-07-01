#  차원 축소 : 원본 데이터의 특성을 적은 수의 새로운 특성으로 변환하는 비지도 학습의 한 종류 -> 저장공간 줄이고 시각화 쉬워짐 알고리즘 성능향상
# 주성분 분석 : 차원 축소 알고리즘의 하나로 데이터에서 가장 분산이 큰 방향을 찾는방법
# 설명된 분산 : 주성분 분석에서 주성분이 얼마나 원본 데이터의 분산을 잘 나타내는지 기록한 것
#%%
import numpy as np
import matplotlib.pyplot as plt
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1,100*100)
# %%
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit(fruits_2d)
print(pca.components_.shape)
#n_components=50 으로 지정했기 때문에 pca.components_ 배열의 첫 번째 차원이 50이다
# %%
# 앞장에서 했던 draw_fruits 함수 가져오기
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

draw_fruits(pca.components_.reshape(-1, 100, 100))
# 원본 데이터에서 가장 분산이 큰 방향을 순서대로 나타낸것
# %%
print(fruits_2d.shape)

# %%
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)
# fruits_2d는 (300,10000) 크기의 배열  
# pca를 통해 (300,50) 크기의 배열로 변환


# %%
# 원본데이터 재구성

fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape)

# %%
fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
for start in [0, 100, 200]:
    draw_fruits(fruits_reconstruct[start:start+100])
    print("\n")
# 특성으 줄였다가 다시 원본데이터를 재구성해도 과거의 특성을 잘 가진다
# %%
# 주성분이 원본 데이터의 부산을 얼마나 잘 나타내는지 기록한 값을 설명된 분산이라고한다
print(np.sum(pca.explained_variance_ratio_))
# 대략 92%의 분산을 유지

# %%
# 분산 그래프 출력
plt.plot(pca.explained_variance_ratio_)
plt.show()
# %%
# 다른 알고리즘과 함께 사용하기
# 로지스틱 회귀 모델
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
# %%
# 지도학습 모델을 학습하기 위해선 TARGET값이 필요하다
target =np.array([0]*100 + [1]*100 + [2]*100)
# %%
# 원본 데이터인 fruits_2d데이터를 활용하여 cross_validate()함수로 교차 검증을 수행
from sklearn.model_selection import cross_validate
scores = cross_validate(lr, fruits_2d, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))
# %%
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))
#pca사용으로 엄청난 시간의 단축과 50개뿐인 특성의 갯수로도 정확도가 100퍼센트가 나옴

# %%
pca = PCA(n_components=0.5)
pca.fit(fruits_2d)

# %%
print(pca.n_components_)
# 주성분의 갯수
# %%
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)
# %%
scores =cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

# %%
# 차원 축소된 데이터를 사용해 k -평균 알고리즘으로 클러스터 찾기
from sklearn.cluster import KMeans
km =KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca)
print(np.unique(km.labels_, return_counts=True))
# %%
for label in range(0,3 ):
    data = fruits_pca[km.labels_ == label]
    plt.scatter(data[ : , 0], data[: , 1])
plt.legend(['apple', 'banana', 'pineapple'])
plt.show()
# 데이터를 시각화하면 예상치 못한 통찰을 얻을수 있는데 그런 면에서 차원축소는 매우 유용한 도구 중 하나

