# 테스트 세트로 일반화 성능을 올바르게 예측하려면 가능한 한 테스트 세트를 사용하지 말아야한다 
# 테스트 세트를 사용하지 않으면 모델이 과대적합인지 과소적합인지 모른다 
# 테스트 세트를 사용하지 않고 이를 측정하는 간단한 방법은 훈련세트를 또 나누는것 = 검증세트
#%%
from numpy.core.numeric import cross
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')
# class열을 타깃으로 사용하고 나머지 열은 특성 배열에 저장
data= wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# 훈련 세트의 입력 데이터와 타깃 데이터를 train_input과 train_target배열에 저장
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split( data, target, test_size=0.2, random_state=42)
# train_input과 train_target을 다시 train_test_split()함수에 넣어 훈련세트 sub_input, sub_target과 검증세트 val_input_target을 생성
sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)
# 훈련세트와 검증세트의 크기 비교
print(sub_input.shape, val_input.shape)
# %%
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)
print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))

# %%
# 보통 많은 데이터를 훈련에 사용할수록 좋은 모델이 생성
# 그렇다고 검증세트를 너무 조금 떼어놓으면 검증 점수가 들쭉날쭉되고 불안정할것 
# 이럴때 교차 검증을 이용하면 안정적이고 검증 점수를 얻고 훈련에 더 많은 데이터를 사용가능
# 훈련세트를 세 부분으로 나눠서 교차 검증을 수행하는 것을 3- 폴드 교차검증이라고한다 통칭  k-폴드 교차검증
# 사이킷런에는 cross_validate() 교차 검증 함수 
from sklearn.model_selection import cross_validate
scores = cross_validate(dt, train_input, train_target)
print(scores)
# %%
import numpy as np
print(np.mean(scores['test_score']))
# 이름은 test_score이지만 검증 폴드의 점수 

# %%
from sklearn.model_selection import StratifiedKFold
scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
print(np.mean(scores['test_score']))
# %%
# 모델이 학습할 수 없어서 사용자가 지정해야만 하는 파라미터를 하이퍼파라미터라고한다
from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease' : [0.0001, 0.0002, 0.003, 0.0004, 0.0005]}
# 탐색 대상 모델과 params변수를 전달하여 그리드 서치 객체를 만듬
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs= -1)
gs.fit(train_input, train_target)
dt =gs.best_estimator_
print(dt.score(train_input, train_target))
#%%
print(gs.best_params_)

#%%
print(gs.cv_results_['mean_test_score'])

#%%
best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index])
# 과정정리
# 1. 먼저 탐색할 매개변수 지정
# 2. 그다음 훈련 세트에서 그리드 서치를 수행하여 최상의 평균 검증 점수가 나오는데 매개변수 조합을 찾는다. 이 조합은 그리드 서치 객체에 저장된다
# 3. 그리드 서치는 최상의 매개변수에서 전체 훈련세트를 사용하여 최종 모델을 훈련 이모델도 그리드 서치 객체에 저장

#%%
params = {'min_impurity_decrease' : np.arange(0.0001, 0.001, 0.0001),'max_depth': range(5, 20, 1),
'min_samples_split' : range(2, 100, 10)}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs= -1)
gs.fit(train_input, train_target)
# %%
# 최상의 매개변수 조합 확인
print(gs.best_params_)
# %%
# 랜덤서치 매개변수의 값이 수치일때 값의 범위나 간격을 미리 정하기 어려울 수 있다. 또 너무많은 매개변수 조건이 있어 그리드 서치 수행시간이 오래걸릴 수있다 이러때 랜덤서치를 사용
from scipy.stats import uniform, randint
rgen = randint(0,10)
rgen.rvs(10)

#%%
np.unique(rgen.rvs(1000), return_counts=True)

#%%
ugen = uniform(0,1)
ugen.rvs(10)

#%%
# 탐색할 매개변수 범위 
params = {'min_impurity_decrease' : uniform(0.0001, 0.001), 'max_depth' : randint(20, 50), 'min_samples_split' : randint(2, 25), 'min_samples_leaf' : randint(1, 25),} 
#%%
from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs= -1, random_state=42)
gs.fit(train_input, train_target)

#%% 
print(gs.best_params_)

#%%
print(np.max(gs.cv_results_['mean_test_score']))

#%%
# 최적의 모델은 이미 전체훈련세트로 훈련되어 best_estimator_속에 저장
# 이모델을 최종 모델로 결정후 테스트 세트의 성능 확인
dt = gs.best_estimator_
print(dt.score(test_input, test_target))
# %%
