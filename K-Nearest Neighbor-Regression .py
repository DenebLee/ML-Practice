#%%
#k-최근접 알고리즘 회귀 
#개인적인 예시를 들자면 총 10번의 배달의 민족을 사용하면서 평균적으로 35분정도가 걸렸다. 그럼 11번째 배달을 주문을 했을때 앞서 10번의 경험에서 추론을 하여 35분정도 걸릴꺼라고 예상을 할수있다. 
from itertools import filterfalse
import numpy as np
import matplotlib.pyplot as plt
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])
plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# %%
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
test_array = np.array([1,2,3,4]) 
print(test_array.shape)# 특성을 1개만 사용하기 떄문에 수동으로 2차원 배열을 생성
# %%
test_array = test_array.reshape(2,2) #reshape메서드는 크기가 바뀐 새로운 배열을 반환할 때 지정한 크기가 원본 배열에 있는 원소의 개수와 다르면 에러가 발생 
print(test_array.shape)

# %%
#train_input과 test_input을 2차원 배열로 바꿈 
train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1, 1)
print(train_input.shape, test_input.shape)
# %%
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
print(knr.score(test_input, test_target))
#결정계수  
#회귀에서는 정확한 숫자를 맞힌다는 것은 거의 불가능 예측하는 값이나 타깃 모두 임의의 수치이기때문
#회귀의 경우에는 조금 다른 값으로 평가하는데 이 점수를 결정계수라고한다 R2
#R2 = 1 - (타깃 - 평균)제곱의 합 / (타깃-예측)제곱의 합
# %%
from sklearn.metrics import mean_absolute_error
#테스트 세트에 대한 예측 제작
test_prediction = knr.predict(test_input)
#테스트 세트에 대한 평균 절댓값 오차 계산
mae = mean_absolute_error(test_target, test_prediction)
print(mae)
# %%
#앞에서 훈련한 모델을 사용해 훈련 세트의 R2 점수확인
print(knr.score(train_input, train_target))
#만약 훈련세트에서 점수가 굉장히 좋았는데 테스트 세트에서는 점수가 굉장히 나쁘다면 모델이 훈련세트에 과대적합되었다고말한다 
#반대로 훈련세트에선 점수가 굉장이 낮았는데 테스트세트에선 점수가 좋았다면 과소적합이라고한다  -> 모델이 너무단순해서 
#훈련세트가 전체 데이터를 대표한다고 가정하기 때문에 훈련 세트를 잘 학습하는것이 중요
# %%
#이웃의 개수를 3으로 설정
knr.n_neighbors = 3
#모델을 다시 훈련
knr.fit(train_input, train_target)
#훈련세트의 점수 다시 확인
print(knr.score(train_input, train_target))
#테스트세트의 점수 확인
print(knr.score(test_input, test_target))
# %%
#선형회귀는 널리 사용되는 대표적인 회귀 알고리즘 간단하고 성능이 뛰어남
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
#선형회귀 모델 훈련
lr.fit(train_input, train_target)
print(lr.predict([[50]])) 
# %%
print(lr.coef_, lr.intercept_)
#coef_와 intercept는 머신러닝 알고리즘이 찾은 값이라는 의미로 모델 파라미터라고 부른다
# 머신러닝 알고리즘의 훈련과정은 최적의 모델 파라미터를 찾는것과 같다 이를 모델 기반 학습
#머신러닝에서 기울기를 종종 계수 혹은 가중치라고 부른다
# %%
#훈련세트의 산점도 
plt.scatter(train_input, train_target)
#15에서 50까지 1차 방정식 그래프 그리기
plt.plot([15,50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])
plt.scatter(50, 1241.8, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
#보이는 직선이 선형회귀 알고리즘이 데이터셋에서 찾은 최적의 직선
# %%
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))
print(train_poly.shape, test_poly.shape)
#타깃값은 그대로 사용해야됨
# %%
lr.fit(train_poly, train_target)
print(lr.predict([[50**2, 50]]))
# %%
print(lr.coef_, lr.intercept_)
#무게 = 1.01x 길이제곱 - 21.6 x 길이 + 116.05
#이런 방정식을 다항식이라 부르며 다항식을 사용한 선형회귀를 다항회귀라고한다 
# %%
#구간별 직선을 그리기 위해 15에서 49까지 정수 배열을 만듬
point = np.arange(15, 50 )
#훈련 세트의 산점도 
plt.scatter(train_input, train_target)
#15에서 49까지 2차 방정식 그래프를 그림
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)
plt.scatter(50, 1574, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# %%
#테스트필드 점수확인
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))

#선형회귀는 특성이 많을수록 효과증대가 크다 
#pandas는 데이터분석 라이브러리 데이터프레임은 판다스의 핵심데이터구조
# %%
import pandas as pd
df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
print(perch_full)
#타깃데이터
import numpy as np
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])
       #perch_full perch_weight를 훈련세트와 테스트세트로 나눔
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
       perch_full, perch_weight, random_state=42
)
#%%
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures()
poly.fit([[2, 3]]) # fit(훈련)을해야 변환이 가능하기에 무조건 해야됨
print(poly.transform([[2, 3]]))
# %%
poly = PolynomialFeatures(include_bias = False)
poly.fit([[2, 3]])
print(poly.transform([[2, 3]]))
#절편을 위한 항이 제거 특성의 제곱과 특성끼리 곱한 항만 추가
# %%
poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_poly.shape)
# %%
#9개의 특성이 각각 어떤 입력의 조합으로 만들어졌는지 알려줌
poly.get_feature_names() #<--- 
test_poly = poly.transform(test_input)

#항상 훈련세트를 기준으로 테스트세트를 변환하는 습관을 들이는것이 좋다 
# %%
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))
#테스트 세트에 대한 점수는 높아지지 않았지만 과소적합 문제는 더이상 나타나지 않음
# %%
poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape)
#55제곱까지 특성을 추가
# %%
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
# %%
print(lr.score(test_poly, test_target))
# %%
#규제 = 머신러닝 모델이 훈련세트를 너무 과도하게 학습하지 못하도록 훼방하는것을 말함 
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)
#StandardScaler 클래스의 객체 ss를 초기화한후 polynimialFeatures 클래스로 만든 train_poly를 사용해 객체를 훈련
#%%
#선형회귀 모델에 규제를 추가한 모델을 릿지와 라쏘라고 칭함
#두모델은 규제를 가하는 방법이 다른데 릿지는 계수를 제곱한 값을 기준으로 규제적용
#라쏘는 계수의 절댓값을 기준으로 규제적용 일반적으로 릿지를 조금더 선호 
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
# %%
print(ridge.score(test_scaled, test_target))
#릿지와 라쏘 모델을 사용할때 규제의 양을 임의로 조절할수 있다 
#모델 객체를 만들때 alpha매개변수로 규제의 강도를 조절 값이 크면 규제 강도가 쎄짐
# %%
import matplotlib.pyplot as plt
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
       #릿지 모델을 생성
       ridge = Ridge(alpha =alpha)
       #릿지 모델 훈련
       ridge.fit(train_scaled, train_target)
       #훈련 점수와 테스트 점수를 저장
       train_score.append(ridge.score(train_scaled, train_target))
       test_score.append(ridge.score(test_scaled, test_target))
#%%
#그래프 그리기
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
# %%
ridge =Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))
# %%
#라쏘 회귀
#라쏘 모델을 훈련하는것은 릿지와 매우 비슷하다 Ridge -> Lasso클래스로 변경하는거뿐
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
# %%
train_score = []
test_score =[]
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
       #라쏘 모델 생성
       lasso = Lasso(alpha=alpha, max_iter=10000)
       #라쏘 모델 훈련
       lasso.fit(train_scaled, train_target)
       #훈련 점수와 테스트 점수를 저장
       train_score.append(lasso.score(train_scaled, train_target))
       test_score.append(lasso.score(test_scaled, test_target))
# %%
#라쏘 그래프 생성
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
#왼쪽은 과대적합을 보여주고 오른쪽으로 갈수록 훈련 세트와 테스트 세트의 점수가 좁혀진다
# 이 지점은 분명 과소적홥되는 모델
# %%
#라쏘 모델에서 최적의 alpha값은 1, 즉 10이다 
lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
# %%
print(np.sum(lasso.coef_ ==0))
#정리 
#특성을 많이 추가하면 선형 회귀는 매우 강력한 성능을 낸다 하지만 특성이 너무 많으면 회귀 모델을 제약하기 위한 도구가 필요하다 

# %%
