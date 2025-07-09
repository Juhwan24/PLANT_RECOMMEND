# 🌱 실내 식물 추천 시스템 (Indoor Plant Recommendation System)

## 📋 프로젝트 개요

본 프로젝트는 **20종의 실내 식물 데이터**와 **난이도 가중치 데이터**를 활용하여 사용자의 환경과 선호도에 맞는 최적의 실내 식물을 추천하는 **AI 기반 개인화 추천 시스템**입니다.

---

## 🗂️ 데이터셋 분석

### 1. Indoor_Plant_With_AirPurification_Label.csv
- **데이터 크기**: 1,002개 샘플, 18개 특성
- **식물 종류**: 20종의 실내 식물
- **주요 특성**:
  - `Plant_ID`: 식물 종류 (20종)
  - `Height_cm`: 식물 높이 (cm)
  - `Leaf_Count`: 잎 수
  - `New_Growth_Count`: 새 성장 개수
  - `Watering_Amount_ml`: 물주기 양 (ml)
  - `Watering_Frequency_days`: 물주기 빈도 (일)
  - `Sunlight_Exposure`: 햇빛 노출 조건
  - `Room_Temperature_C`: 실내 온도 (°C)
  - `Humidity_%`: 습도 (%)
  - `Health_Score`: 건강 점수 (1-5)
  - `Air_Purification_Effective`: 공기정화 효과 (0/1)

### 2. Difficulty_Weights.xlsx
- **데이터 크기**: 5개 햇빛 조건, 2개 특성
- **주요 특성**:
  - `Sunlight_Exposure`: 햇빛 노출 조건
  - `Difficulty_Score (0~20)`: 난이도 점수

---

## 🔬 데이터 분석 및 전처리 기법

### 1. Pandas 데이터 처리
```python
# 데이터 로드 및 병합
plants_df = pd.read_csv('Indoor_Plant_With_AirPurification_Label.csv')
weights_df = pd.read_excel('Difficulty_Weights.xlsx')

# 그룹별 집계 함수를 활용한 특성 추출
plant_features = plants_df.groupby('Plant_ID').agg({
    'Height_cm': 'mean',
    'Leaf_Count': 'mean', 
    'Watering_Frequency_days': 'mean',
    'Room_Temperature_C': 'mean',
    'Humidity_%': 'mean',
    'Health_Score': 'mean',
    'Air_Purification_Effective': 'mean'
}).reset_index()
```

### 2. 데이터 정규화 (StandardScaler)
```python
# 특성 정규화를 통한 스케일 통일
features_to_normalize = ['Height_cm', 'Leaf_Count', 'Watering_Frequency_days', 
                        'Room_Temperature_C', 'Humidity_%', 'Health_Score', 'Difficulty_Score']

scaler = StandardScaler()
plant_features_normalized[features_to_normalize] = scaler.fit_transform(
    plant_features[features_to_normalize]
)
```

### 3. 범주형 데이터 매핑
```python
# 햇빛 노출을 난이도 점수로 변환
sunlight_mapping = dict(zip(weights_df['Sunlight_Exposure'], 
                           weights_df['Difficulty_Score (0~20)']))

# 식물별 가장 흔한 햇빛 조건의 난이도 점수 추출
sunlight_difficulty = plants_df.groupby('Plant_ID')['Sunlight_Exposure'].agg(
    lambda x: sunlight_mapping.get(x.mode()[0] if len(x.mode()) > 0 else x.iloc[0], 10)
).reset_index()
```

---

## 🧮 머신러닝 알고리즘

### 1. 코사인 유사도 (Cosine Similarity)
```python
from sklearn.metrics.pairwise import cosine_similarity

# 사용자 프로필 벡터 생성
user_vector = np.array([
    user_profile['temperature'],      # 온도
    user_profile['humidity'],         # 습도
    user_profile['watering_frequency'], # 물주기 빈도
    user_profile['air_purification_priority'] == '높음' # 공기정화 우선순위
])

# 식물 특성 벡터
plant_vectors = filtered_plants[['Room_Temperature_C', 'Humidity_%', 
                                'Watering_Frequency_days', 'Air_Purification_Effective']].values

# 유사도 계산
similarities = cosine_similarity([user_vector], plant_vectors)[0]
```

### 2. 가중치 적용 알고리즘
```python
# 다중 가중치 시스템
for idx, (_, plant) in enumerate(filtered_plants.iterrows()):
    base_score = similarities[idx]
    
    # 1. 건강 점수 가중치 (20% 가중치)
    health_bonus = plant['Health_Score'] / 5.0 * 0.2
    
    # 2. 난이도 적합성 가중치 (30% 가중치)
    difficulty_center = sum(user_profile['difficulty_range']) / 2
    difficulty_penalty = 1 - abs(plant['Difficulty_Score'] - difficulty_center) / 10
    
    # 3. 최종 점수 계산
    final_score = base_score + health_bonus + difficulty_penalty * 0.3
    weighted_scores.append(final_score)
```

---

## 🎯 추천 시스템 알고리즘

### 1. 다단계 필터링 시스템
```python
def filter_plants_by_constraints(self, user_profile):
    filtered_plants = self.plant_features.copy()
    
    # 1단계: 난이도 범위 필터링 (유연한 범위 적용)
    min_diff, max_diff = user_profile['difficulty_range']
    min_diff = max(0, min_diff - 2)  # 2점 여유
    max_diff = min(20, max_diff + 2) # 2점 여유
    
    # 2단계: 크기 범위 필터링 (유연한 범위 적용)
    min_size, max_size = user_profile['size_range']
    min_size = max(0, min_size - 10)  # 10cm 여유
    max_size = min(100, max_size + 10) # 10cm 여유
    
    # 3단계: 공기정화 우선순위 필터링 (선택적)
    if user_profile['air_purification_priority'] == '높음':
        air_purification_plants = filtered_plants[
            filtered_plants['Air_Purification_Effective'] == 1
        ]
        if len(air_purification_plants) >= 3:
            filtered_plants = air_purification_plants
    
    return filtered_plants
```

### 2. 개인화 프로필 생성
```python
def get_user_profile(self, experience_level, sunlight, temperature, humidity, 
                    watering_frequency, air_purification_priority, size_preference):
    
    # 경험 수준별 난이도 범위 매핑
    difficulty_ranges = {
        '초보자': (0, 7),    # 난이도 0-7
        '중급자': (5, 12),   # 난이도 5-12
        '전문가': (10, 20)   # 난이도 10-20
    }
    
    # 크기 선호도별 높이 범위 매핑
    size_ranges = {
        '작은 식물': (0, 20),    # 0-20cm
        '중간 크기': (15, 35),   # 15-35cm
        '큰 식물': (30, 100)     # 30-100cm
    }
    
    return user_profile
```

---

## 📊 데이터 시각화

### 1. Plotly 레이더 차트
```python
import plotly.graph_objects as go

# 식물 특성 비교 레이더 차트
fig = go.Figure()

for _, plant in recommendations.iterrows():
    fig.add_trace(go.Scatterpolar(
        r=[plant['Health_Score'], plant['Air_Purification_Effective'], 
           plant['Height_cm']/50, plant['Leaf_Count']/50, 
           20-plant['Difficulty_Score']],
        theta=['건강점수', '공기정화', '크기', '잎수', '쉬운관리'],
        fill='toself',
        name=plant['Plant_ID']
    ))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
    showlegend=True,
    title="식물 특성 비교"
)
```

### 2. Streamlit 인터랙티브 대시보드
- **사이드바**: 사용자 입력 폼
- **메인 영역**: 추천 결과 및 시각화
- **확장 패널**: 상세 정보 표시
- **실시간 업데이트**: 조건 변경 시 즉시 반영

---

## 🔧 기술 스택

### 데이터 처리
- **Pandas**: 데이터 로드, 전처리, 집계
- **NumPy**: 수치 계산, 배열 연산
- **OpenPyXL**: Excel 파일 처리

### 머신러닝
- **Scikit-learn**: 
  - `StandardScaler`: 특성 정규화
  - `cosine_similarity`: 유사도 계산
- **커스텀 가중치 알고리즘**: 다중 기준 점수 계산

### 웹 애플리케이션
- **Streamlit**: 인터랙티브 웹 인터페이스
- **Plotly**: 동적 데이터 시각화

---

## 📈 데이터 분석 결과

### 1. 식물 분포 분석
- **총 20종의 실내 식물**
- **1,002개의 성장 데이터 샘플**
- **18개의 다양한 특성 분석**

### 2. 환경 조건별 분류
- **햇빛 노출**: 5단계 (Low light corner ~ 6h full sun)
- **난이도 점수**: 0-20점 (햇빛 조건 기반)
- **온도 범위**: 15-30°C
- **습도 범위**: 30-80%

### 3. 성능 지표
- **정확도**: 사용자 선호도 기반 매칭
- **다양성**: 20종의 다양한 식물 추천
- **개인화**: 7개 차원의 개인 프로필

---

## 🚀 시스템 특징

### 1. 지능형 필터링
- **다단계 제약 조건**: 난이도, 크기, 공기정화
- **유연한 범위 적용**: 여유 범위로 추천 다양성 확보
- **동적 필터링**: 조건에 따른 실시간 조정

### 2. 개인화 알고리즘
- **다차원 유사도**: 온도, 습도, 물주기, 공기정화
- **가중치 시스템**: 건강점수, 난이도 적합성
- **경험 수준 매칭**: 사용자 능력에 맞는 난이도

### 3. 시각적 피드백
- **레이더 차트**: 다차원 특성 비교
- **실시간 업데이트**: 조건 변경 시 즉시 반영
- **상세 정보**: 각 식물의 관리법 제공

---

## 📝 결론

본 시스템은 **데이터 기반 의사결정**과 **개인화된 추천 알고리즘**을 통해 사용자에게 최적의 실내 식물을 제공합니다. 

**핵심 성과**:
- ✅ **20종 식물 데이터베이스** 활용
- ✅ **AI 기반 유사도 매칭** 구현
- ✅ **다중 가중치 알고리즘** 적용
- ✅ **인터랙티브 웹 인터페이스** 제공
- ✅ **실시간 시각화** 구현

이를 통해 사용자는 자신의 환경과 선호도에 맞는 완벽한 실내 식물을 쉽게 찾을 수 있습니다. 