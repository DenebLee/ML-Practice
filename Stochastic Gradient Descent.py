# 훈련 세트에서 샘플 하나씩 꺼내 손실 함수의 경사를 따라 최적의 모델을 찾는 알고리즘 
# 샘플을 하나씩 사용하지 않고 여러 개를 사용하면 미니배치 경사 하강법이 된다 
# 한번에 전체 샘플을 사용하면 배치 경사 하강법

# 점진적 학습 : 훈련된 모델을 버리지않고 새로운 데이터에 대해서만 조금씩 훈련시키는 방법 
# 대표적인 점진적 학습 알고리즘 = 확률적 경사 하강법

# 확률적 경사 하강법엣 훈련 세트를 한번에 모두 사용하는 과정을 에포크(eporch)
#여러개의 샘플을 선택할 경우 미니배치 경사하강법이라고한다 
# 전체 샘플을 사용할경우 배치 경사 하강법 -> 가장 안정적이지만 자원소모가 심함
# 확률적 경사 하강법을 꼭 사용하는 알고리즘 = 신경망 알고리즘

#손실함수란 어떤 문제에서 머신러닝 알고리즘이 얼마나 엉터리인지를 측정하는 기준
# 대부분의 문제에 잘 맞는 손실 함수가 이미 정의되어 있다 . 이진분류에는 로지스틱 회귀( 또는 이진 크로스엔트로피) 손실 함수 사용
# 다중 분류에는 크로스엔트로피 손실 함수 사용 
# 회귀 문제에는 평균제곱 오차 손실 함수 사용 
#%%
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')
# %%
fish_input = fish[['Weight', 'Length','Diagonal', 'Height', 'Width' ]].to_numpy()
fish_target = fish['Species'].to_numpy()
# %%
#사이킷런의 train_test_split() 함수를 사용해 데이터를 훈련, 테스트 세트로 나눔
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state = 42)
# %%
# 훈련세트와 테스트 세트의 특성을 표준화 전처리 *꼭 훈련세트에서 학습한 통계 값으로 테스트세트도 변환해야함
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
# %%
# SGDClassifier의 객체를 만들때 2개의 매개변수 지정
# loss는 손실 함수의 종류를 지정
# loss= 'log'로 지정하여 로지스틱 손실 함수 지정 
# max_iter는 수행할 에포크 횟수를 지정
from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss='log', max_iter=10, random_state=42)
sc.fit(train_scaled, train_target)
# %%
# 훈련세트와 테스트 세트의 정확도 점수 출력
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
# 출력된 훈련과 테스트 세트의 정확도가 낮음 횟수 10번은 부족
# %%
sc.partial_fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
#에포크를 한번더 실행하니 정확도가 향상 # %%
# 훈련세트 점수는 에포크가 진행될수록 꾸준히 증가하지만 테스트 세트 점수는 어느 순간 감소하기 시작
# 바로 이 지점이 해당 모델의 과대적합되기 시작하는곳
# 과대적합이 시작하기 전에 훈련을 멈추는 것을 조기 종료
import numpy as np
sc = SGDClassifier(loss='log', random_state=42)
train_score = []
test_score = []
classes = np.unique(train_target)
# 300번의 에포크 동안 훈련을 반복하여 진행 
# 반복마다 훈련세트와 테스트 세트의 점수 계산하여 train과 test score 리스트에 추가
#%%
for _ in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes = classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))
# %%
import matplotlib.pyplot as plt
plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
# 파란색( 훈련세트 그래프) 주황색(테스트 세트 그래프)
# %%
sc = SGDClassifier(loss='log', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled,test_target))
# SGDClassfier는 일정 에포크 동안 성능이 향상되지 않으면 더 훈련하지 않고 자동으로 멈춘다
# tol매개변수에서 향상될 최솟값을 지정
# %%
# SGDClassifier의 loss 매개변수의 기본값은 'hinge'이다. 힌지 손실은 서포트 벡터 머신이라 불리는 또 다른 머신러닝 알고리즘 을 위한 손실함수
sc = SGDClassifier(loss='hinge', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
# %%
