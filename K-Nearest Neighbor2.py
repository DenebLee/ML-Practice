# Data = input Answer(정답) = target 
# Data + Answer = Trainning Data
# input으로 사용된것을 특성(feature)
# 지도학습은 정답이 있으니 정답을 맞히는 것을 학습한다. 반면 비지도 학습 알고리즘은 타깃없이 입력데이터만 사용
#지도학습에는 분류(classification) 회귀(regression)
#분류 = Y/N로 클래스를 판별 (이중 분류와 다중분류)
#회귀 = 연속된 값을 예측하는것 (앞으로의 부동산 전망, 연봉상승추이등등) 단 연속적인 숫자가 있어야됨
#비지도학습은 방대한 데이터를 전처리하기위해 사용하며 지도학습은 설정된 학습모델로 해당 전처리된 데이터를 처리하기위해 사용
#%%
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

fish_data = [[l,w] for l, w in zip(fish_length, fish_weight)] 
fish_target = [1]*35 + [0]*14

#%%
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
print(fish_data[4])
# %%
print(fish_data[0:5])
# %%
print(fish_data[:5])
# %%
print(fish_data[44:])
# %%
train_input = fish_data[:35]
train_target = fish_target[:35]
test_input = fish_data[35:]
test_target = fish_target[35:]
kn = kn.fit(train_input, train_target)
kn.score(test_input, test_target)

# %%
import numpy as np
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)
print(input_arr)
# %%
print(input_arr.shape) #샘플 수, 특성 수
# %%
np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)
print(index)
# %%
train_input = input_arr[index[:35]]
train_target =target_arr[index[:35]]
print(input_arr[13], train_input[0])
# %%
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]
# %%
import matplotlib.pyplot as plt
plt.scatter(train_input[ : , 0 ], train_input[:,1])
plt.scatter(test_input[ : , 0 ], test_input[:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# %%
kn = kn.fit(train_input, train_target)
kn.score(test_input, test_target)
# %%
kn.predict(test_input)
test_target
# %%
