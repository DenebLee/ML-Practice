#%%
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')
# %%
wine.head()
#info 메서드 = 데이터프레임의 각 열의 데이터타입과 누락된 데이터가 있는지 확인하는데 유용
# %%
wine.info()
# %%
# describe()메서드 = 열에 대한 간략한 통계를 출력해준다 
wine.describe()
# 평균mean 표준편차std 최소min 최대 max 
# 사분위수 = 데이터를 순서대로 4등분 한 값
# %%
# StandardScaler 클래스를 사용해 특성을 표준화
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
# %%
# wine 데이터프레임에서 처음 3개의 열을 넘파이 배열로 바꿔서 data배열에 저장하고 마지막 calss열을 넘파이 배열로 바꿔 target배열에 저장 
# 훈련세트와 테스트 세트로 나누기
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

# %%
# 만들어진 훈련세트와 테스트 세트의 크기 확인
print(train_input.shape, test_input.shape)
# %%
# 훈련세트 전처리 
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
# %%
# 표준점수로 변환된 train_scaled와 test_scaled를 사용해 로지스틱 회귀모델 훈련
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
# %%
# 로지스틱 회귀가 학습한 계수와 절편 출력
print(lr.coef_, lr.intercept_)
# %%
# 결정 트리 모델을 만들때 random_state를 지정하는 이유 
# 결정 트리 알고리즘은 노드에서 최적의 분활을 찾기 전에 특성의 순서를 섞는다 따라서 약간의 무작위성이 주입되는데 실행할 때마다 점수가 조금씩 달라질수 있기때문
# 실전에선 불필요
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target)) # 훈련 세트
print(dt.score(test_scaled, test_target)) # 테스트 세트

# %%
# plot_tree() 함수로 트리 그림으로 출력
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()
# 맨위의 노드를 루트노드 
# 맨아래의 노드를 리프 노드
# 노드는 결정트리를 구성하는 핵심 요소 훈련 데이터의 특성에 대한 테스트를 표현
# %%
# 트리의 깊이를 제한해서 출력 
plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar','pH'])
plt.show()
# filed= true로 지정하면 클래스마다 색깔을 부여하고 어떤 클래스의 비율이 높아지면 점점 진한 색으로 표시한다
# 결정 트리에서 예측하는 방법은 간단 
# 리프 노드에서 가장 많은 클래스가 예측 클래스가된다
# KNN과 비슷
# %%
# gini는 지니 불손도를 의미한다 
# DecisionTreeClassfier 클래스의 criterion 매개변수의 기본값이 gini다 
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))
# %%
# 훈련세트의 성능은 낮아졌지만 테스트 세트의 성능은 거의 그대로
# 함수로 그려보기
plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol','sugar', 'pH'])
plt.show()
# 특성값의 스케일은 결정 트리 아록리즘에 아무런 영향을 미치지 않는다. 따라서 표준화 전처리를 할 필요가없다 
# 전처리 하기전의 훈련세트와 테스트 세트로 결정트리 모델을 다시 훈련
#%%
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))
# %%
plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol','sugar', 'pH'])
plt.show()
# 결정 트리는 어떤 특성이 가장 유용한지 나타내는 특성 중요도를 계산해준다. 
# 특성 중요도는 결정 트리 모델의 feature_importances_ 속성에 저장

#%%
print(dt.feature_importances_)
