# 군집화는 비지도학습의 대표적인 기술로 x에 대한 레이블이 지정 되어있지 않은 데이터를 그룹핑하는 분석 알고리즘
# 데이터들의 특성을 고려해 데이터 집단(클러스트)을 정의하고 데이터집단의 대표할수 있는 중심점을 찾는것으로 데이터마이닝의 방법중 하나
# 클러스터란 비슷한 특성을 가진 데이터들의 집단이다

#%%
import numpy as np
import matplotlib.pyplot as plt

fruits = np.load('fruits_300.npy')
print(fruits.shape)
# 배열의 첫번째 차원은 샘플의 개수
# 두번째 차원은 이미니높이
# 세번째 차원은 이미지너비
#%%
print(fruits[0 ,0, :])
# %%
plt.imshow(fruits[0], cmap='gray')
plt.show()
# %%
# 사진 분석 및 사진에 대한 데이터를 뽑아서 처리를 할땐 
# 컴퓨터가 255에 해당하는 하얀색 배경에 집중못하게
# 객체에는 하얀색으로 배경엔 검은색 즉 값을 떨어뜨려 객체에만 집중하도록 설정한다

plt.imshow(fruits[0], cmap='gray_r')
plt.show()

#%%
fig, axs = plt.subplots(1,2)
axs[0].imshow(fruits[100], cmap='gray_r')
axs[1].imshow(fruits[200], cmap='gray_r')
plt.show()
# %%
# 맷플롯립 subplots 함수를 사용하면 여러 개의 그래프를 배열처럼 쌓을 수 있도록 도와준다
# 픽셀값 계산하기
apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)
print(apple.shape)
# %%
# 애플 파인애플 바나나 배열에 들어가있는 샘플의 픽셀 평균값을 계산 -> mean()메서드 사용
print(apple.mean(axis=1))
# %%
plt.hist(np.mean(apple, axis=1), alpha=0.8)
plt.hist(np.mean(pineapple, axis=1), alpha=0.8)
plt.hist(np.mean(banana, axis=1), alpha=0.8)
plt.legend(['apple', 'pineapple', 'banana'])
plt.show
# %%
# 픽셀의 평균계산
fig, axs= plt.subplots(1, 3, figsize=(20, 5))
axs[0].bar(range(10000), np.mean(apple, axis=0))
axs[1].bar(range(10000), np.mean(pineapple, axis=0))
axs[2].bar(range(10000), np.mean(banana, axis=0))
plt.show()
# %%
apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean =np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean =np.mean(banana, axis=0).reshape(100, 100)
fig , axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')
plt.show()

# %%
# 평균값과 가까운 사진 고르기
abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis=(1,2))
print(abs_mean.shape)

# %%
# 값이 가장 작은 순서대로 100개 고르기
apple_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
    for j in range(10):
            axs[i , j].imshow(fruits[apple_index[i*10 + j]], cmap='gray_r')
            axs[i , j].axis('off')
plt.show()

# %%
