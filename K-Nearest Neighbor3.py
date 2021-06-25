#%%
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
import numpy as np
np.column_stack(([1,2,3], [4,5,6,]))
fish_data = np.column_stack((fish_length, fish_weight))
print(fish_data[:5])
# %%
print(np.ones(5))
# %%
fish_target = np.concatenate((np.ones(35), np.zeros(14)))
print(fish_target)
#파이썬 리스트로 하지않는이유 -> numpy는 저수준언어로 구성 리소스적게먹음
# %%
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)
print(train_input.shape, test_input.shape)
print(train_input.shape, test_target.shape)
print(test_target)
# %%
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)
print(test_target)
# %%
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)
print(kn.predict([[25,150]]))
# %%
import matplotlib.pyplot as plt
plt.scatter(train_input[ : , 0], train_input[ : , 1])
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# %%
distances, indexes = kn.kneighbors([[25,150]]) #거리계산 

plt.scatter(train_input[ : , 0 ], train_input[ : , 1 ])
plt.scatter(25,150, marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# 두 특성의 값이 놓인 범위가 다르다 이를 두 특성의 스케일이 다르다고 말한다
# 알고리즘이 거리기반일때 샘플간의 거리에 영향을 많이 받는데 제대로 사용하려면 특성값을 일정한 기준으로 맞춰주어야 하는데 이를 전처리 라고한다

# %%
print(train_input[indexes])
# %%
print(train_target[indexes])
# %%
print(distances)
# %%
plt.scatter(train_input[ : , 0], train_input[ : , 1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1],marker='D')
plt.xlim((0, 1000))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# %%
# 가장 널리 사용하는 전처리 방법 중 하나는 표준점수 
# 각 특성값이 0에서 표준편차의 몇 배만큼 떨어져 있는지를 나타낸다.
# np.mean() 함수는 평균을 계산 np,std()는 표준편차를 계산
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
print(mean, std)
# %%
train_scaled = (train_input - mean ) / std
#train_input의 모든행에 mean에 있는 두 평균값을 빼주고 std에 있는 모든 표준편차를 다시 모든행에 적용 = 이런 numpy기능을 브로드캐스팅
plt.scatter(train_scaled[: , 0], train_scaled[ : , 1])
plt.scatter( 25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# %%
new = ([25, 150] - mean ) / std
plt.scatter(train_scaled [ : , 0], train_scaled[: , 1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# %%
kn.fit(train_scaled, train_target) # K-Nearest Neighbor 다시 훈련
test_scaled = (test_input -mean) / std
kn.score(test_scaled, test_target) #학습시킨 학습모델 점수
# %%
print(kn.predict(([new])))
# %%
distances, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[ : , 0], train_scaled[ : , 1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes, 0 ], train_scaled[indexes,1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show() #학습된 모델을 통해 도미로 판별
# %%
