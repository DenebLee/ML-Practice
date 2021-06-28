#%%
#데이터 준비하기 
from numpy.random import logistic
import pandas as pd 
fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()
# %%
print(pd.unique(fish['Species']))
# %%
#데이터프레임에서 열을 선택하는 방법은 간단 데이터프레임에서 원하는 열을 리스트로 나열하면됨
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
print(fish_input[:5])
#데이터프레임에서 여러 열을 선택하면 새로운 데이터프레임이 반환된다
#%%
fish_target = fish['Species'].to_numpy()
# %%
#사이킷런의 StandardScaler 클래스를 사용해 훈련 세트와 테스트 세트를 표준화 전처리
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state = 42)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
# %%
#사이킷런의 KNeightborsClassifier 클래스 객체를 만들고 훈련 세트로 모델을 훈련한 다음 훈련 세트와 테스트 세트의 점수를 확인
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))
# %%
# 로지스틱 회귀는 선형 방정식을 사용한 분류 알고리즘 
# 다중 분류는 타깃 클래스가 2개 이상인 분류 문제 
# 시그모이드 함수는 선형 방정식의 출력을 0~ 1사이의 값으로 압축하여 이진 분류를 위해 사용
# 소프트맥스 함수는 다중 분류에서 여러 선형 방정식의 출력 결과를 정규화 하여 합이 1이 되도록 만듬
# %%
# 타깃 데이터에 2개 이상의 클래스가 포함된 문제를 다중 분류
print(kn.classes_)
# %%
# 테스트 세트에 있는 처음 5개 샘플의 타깃값을 예측
print(kn.predict(test_scaled[ :5 ]))

# %%
import numpy as np
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))

# %%
distance, indexes = kn.kneighbors(test_scaled[3 : 4])
print(train_target[indexes])
# %%
# 로지스틱 회귀는 이름은 회귀이지만 분류모델 선형회귀와 동일하게 선형 방정식을 학습
# z = a x (Weight) + b x (Lenght) + c x (Diagonal) + d x (Height) + e x (Width) + f

# a,b,c,d,e는 가중치 혹은 계수 
import numpy as np
import matplotlib.pyplot as plt
z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z , phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()
# %%
#numpy 배열은 True, False값을 전달하여 행을 선택할수 있다
# 이를 불리언 인덱싱(boolean indexcing)

char_arr = np.array(['A', 'B', 'C', 'D', 'E'])
print(char_arr[[True, False, True, False, False]])
# %%
bream_smelt_indexes = (train_target =='Bream') | (train_target =='Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]
# %%
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

# %%
print(lr.predict_proba(train_bream_smelt[:5]))

# %%
print(lr.predict(train_bream_smelt[:5]))
# %%
print(lr.classes_)
# %%
# 로지스틱 회귀가 학습한 계수 확인
print(lr.coef_, lr.intercept_)
# 여기서 사용된 로지스틱 회귀 모델이 학습한 방정식은
# z = -0.404 x (Weight) - 0.576 x (Lenght) - 0.663 x (Diagonal) - 1.013 x (Height) - 0.732 x (Width) - 2.161
#%%
# train_bream_smelt의 처음 5개 샘플의 z값 출력
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)
# %%
from scipy.special import expit
print(expit(decisions))
# %%
# LogisticRegression 클래스는 기본적으로 반복적인 알고리즘을 사용
# 또한 기본적으로 릿지 회귀와 같익 ㅖ수의 제곱을 규제 
# 이런 규제를 L2규제라고함
# LogisticRegression 클래스로 다중 분류 모델을 훈련하는 코드
lr = LogisticRegression(C=20, max_iter= 1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
# %%
# 테스트 세트의 처음 5개 샘플에 대한 예측 출력
print(lr.predict(test_scaled[:5]))

# %%
# 테스트 세트의 처음 5개 샘플에 대한 예측 확률 출력
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))
# %%
print(lr.classes_)
# %%
print(lr.coef_.shape, lr.intercept_.shape)
# 다중 분류는 클래스마다 z값을 하나씩 계산
# %%
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))
# %%
from scipy.special import softmax
proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))

# %%
# k-최근접 이웃 모델이 확률을 출력할 수 있지만 이웃한 샘플의 클래스 비율이므로 항상 정해진 확률만 출력 
# 로지스틱 회귀는 이진 분류에서는 하나의 선형 방정식을 훈련
# 다중 분류일 경우에는 클래스 개수만큼 방정식을 훈련