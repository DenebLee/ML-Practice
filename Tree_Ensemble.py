# 정형데이터 = 데이터베이스의 정해진 규첵이 맞게 데이터를 들어간 데이터중에 수치만으로 의미 파악이 쉬운 데이터들
# 비정형 데이터 = 정해진 규칙이 없어서 값의 의미를 쉽게 파악하기 힘든경우 ( 텍스트, 음성, 영상과 같은 데이터
# 정형 데이터를 다루는데 가장 뛰어난 성과를 내는 알고리즘이 앙상블 학습
# 반대로 비정형 데이터를 다루는데 가장 뛰어난 성과를 내는 알고리즘은 신경망 알고리즘
# 랜덤포레스트는 앙상블 학습의 대표 주자중 하나로 안정적인 성능 덕분에 널리 사용되고있다 
# 랜덤 포레스트는 선택한 샘플과 특성을 사용하기 때문에 훈련 세트에 과대적합되는 것을 막아주고 검증세트와 테스트 세트에서 안정적인 성능을 얻을 수 있다
#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
train_input, test_input, train_target,test_target = train_test_split( data, target, test_size=0.2, random_state=42)

#%%
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs= -1, random_state=42)
scores = cross_validate(rf, train_input, train_target,return_train_score=True, n_jobs= -1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))                                                                                                                
 
#%%
#결정트리의 큰 장점중 하나인 특성중요도를 계산
rf.fit(train_input, train_target)
print(rf.feature_importances_)
# 랜덤포레스트는 하나의 특성에 과도하게 집중하지 않고 좀더 많은 특성이 훈련에 기여할 기회를 얻는다
#%%
rf = RandomForestClassifier(oob_score=True, n_jobs= -1, random_state=42)
rf.fit(train_input, train_target)
print(rf.oob_score_)

#%%
# 엑스트라 트리는 랜덤 포레스트와 매우 비슷하게 동작 
# 랜덤 포레스트와의 차이점은 부트스트랩 샘플을 사용하지 않는다는점 
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_jobs= -1, random_state=42)
scores = cross_validate(et, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 랜덤 포레스트와 비슷한 결과를 얻음(특성이 많지않아 차이가 별로없음)
# 엑스트라 트리는 무작위성이 좀 더 크기 떄문에 랜덤 포레스트보다 더 많은 결정 트리를 훈련
# 하지만 랜덤하게 노드 분활하기에 속도가 더빠름

#%%
et.fit(train_input, train_target)
print(et.feature_importances_)
# 엑스트라 트리의 회귀 버전은 ExtraTreesRegressor 클래스 

#%%
# 그레이디언트 부스팅은 깊이가 얕은 결정 트리를 사용하여 이전 트리의 오차를 보완하는 방식으로 앙상블 하는 방법
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 그레이디언트 부스팅은 결정 트리의 개수를 늘려도 과대적합에 매우강하다 
# 학습률을 증사키시고 트리의 개수를 늘리면 성능이 향상됨

#%%
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, random_state=42)
scores = cross_validate(gb, train_input,train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 결정 트리 개수를 500개로 5배나 늘렸지만 과대적합을 잘억제하고있다
# 그레이디언트 부스팅도 특성 중요도를 제공
# %%
gb.fit(train_input, train_target)
print(gb.feature_importances_)
# 일반적으로 그레이디언트 부스팅이 랜덤 포레스트보다 조금 더 높은 성능을 얻을수 있다. 하지만 순서대로 트리를 추가하기 때문에 훈련속도는 느리다
# 그레이디언트 부스팅의 속도와 성능을 더욱 개선한 것이 다음에 살펴볼 히스토그램 기반 그레이디언트 부스팅

#%% 
# 히스토그램 기반 그레이디언트 부스팅
# 정형 데이터를 다루는 머신러닝 알고리즘 중에 가장 인기가 높은 알고리즘
# 입력 특성을 256개의 구간으로 누눈다. 따라서 노드를 분할 할때 최적의 분할을 매우 빠르게 찾을 수 있다. 
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
hgb = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(gb, train_input,train_target, return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 과대적합을 잘 억제하면서 그레이디언트 부스팅보다 조금 더 높은 성능을 제공한다. 

#%%
from sklearn.inspection import permutation_importance
hgb.fit(train_input, train_target)
result =permutation_importance(hgb, train_input, train_target, n_repeats=10, random_state=42, n_jobs= -1)
print(result.importances_mean)
# permutation_importance() 함수가 반환하는 객체는 반복하여 얻은 특성 중요도, 평균, 표준 편차를 담고있다

#%%
result = permutation_importance(hgb, test_input, test_target, n_repeats=10, random_state=42, n_jobs= -1)
print(result.importances_mean)
# 분석을 통해 모델을 실전에 투입했을때 어떤 특성에 관심을 둘지 예상가능

#%%
hgb.score(test_input, test_target)
# 최종 확인
# %%
from xgboost import XGBClassifier
xgb= XGBClassifier(tree_method='hist', random_state=42)
scores= cross_validate(xgb, train_input, train_target, return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# %%
from lightgbm import LGBMClassifier
lgb = LGBMClassifier(random_state=42)
scores= cross_validate(lgb, train_input, train_target, return_train_score=True, n_jobs= -1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# %%
