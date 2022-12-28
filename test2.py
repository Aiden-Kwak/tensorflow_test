# dataset

'''
admit: 붙었나
gre: 영어성적
gpa: 학점
rank: 지원학교수준(1이 높음)

합격확률 계산하기
'''
# csv 전처리
# 빈 데이터는 어떻게할까? 평균을 넣거나 행을 날리거나
import pandas as pd

data = pd.read_csv('./dataset/gpascore.csv')

# 판다스 유용한 사용법들
'''
null = data.isnull().sum() # 빈칸 확인용
data.dropna() # 빈칸있는행 날리기
data.fillna(100) # 빈칸을 다 100으로
data['gpa'].min() # 최소값
data['gpa'].count() # 개수
'''
data = data.dropna()

y_data = data['admit'].values
x_data = []

# iterrows : dataframe을 가로 한줄씩 출력한다
for i, rows in data.iterrows():
    x_data.append([rows['gre'],rows['gpa'],rows['rank']])


import tensorflow as tf
import numpy as np

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    # 레이어와 노드수는 실험적결정이다. 마지막레이어는 1. 노드수는 2의 제곱수가 관행적이다.
    # 마지막레이어는 예측결과를 뱉는다.
    # 0~1사이로 확률을 얻고싶어서 시그모이드를 사용했다.
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# binary_crossentropy는 결과가 0,1사이의 분류/확률 문제에서 사용한다.


# python list 바로 못넣음. 넘파이 변환
model.fit(np.array(x_data), np.array(y_data), epochs=1000) # 학습. (정답예측에 필요한 인풋, 정답, 반복수)

# 예측
p_data = model.predict([[750, 3.70, 3], [400, 2.2, 1]])
print(p_data)