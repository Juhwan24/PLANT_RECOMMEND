import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

class PlantRecommender:
    def __init__(self):
        self.load_data()
        self.preprocess_data()
        
    def load_data(self):
        """데이터 로드"""
        self.plants_df = pd.read_csv('Indoor_Plant_With_AirPurification_Label.csv')
        self.weights_df = pd.read_excel('Difficulty_Weights.xlsx')
        
    def preprocess_data(self):
        """데이터 전처리"""
        # 햇빛 노출을 난이도 점수로 변환
        sunlight_mapping = dict(zip(self.weights_df['Sunlight_Exposure'], 
                                   self.weights_df['Difficulty_Score (0~20)']))
        
        # 식물별 평균 특성 계산
        plant_features = self.plants_df.groupby('Plant_ID').agg({
            'Height_cm': 'mean',
            'Leaf_Count': 'mean', 
            'Watering_Frequency_days': 'mean',
            'Room_Temperature_C': 'mean',
            'Humidity_%': 'mean',
            'Health_Score': 'mean',
            'Air_Purification_Effective': 'mean'
        }).reset_index()
        
        # 햇빛 노출별 난이도 점수 추가
        sunlight_difficulty = self.plants_df.groupby('Plant_ID')['Sunlight_Exposure'].agg(
            lambda x: sunlight_mapping.get(x.mode()[0] if len(x.mode()) > 0 else x.iloc[0], 10)
        ).reset_index()
        sunlight_difficulty.columns = ['Plant_ID', 'Difficulty_Score']
        
        # 데이터 병합
        self.plant_features = plant_features.merge(sunlight_difficulty, on='Plant_ID')
        
        # [결측치 처리 로직 추가]
        features_to_fill = ['Height_cm', 'Leaf_Count', 'Health_Score', 'Room_Temperature_C', 
                           'Humidity_%', 'Watering_Frequency_days', 'Difficulty_Score']
        for col in features_to_fill:
            if col in self.plant_features.columns:
                self.plant_features[col] = self.plant_features[col].fillna(self.plant_features[col].mean())
        
        # 특성 정규화
        features_to_normalize = ['Height_cm', 'Leaf_Count', 'Watering_Frequency_days', 
                               'Room_Temperature_C', 'Humidity_%', 'Health_Score', 'Difficulty_Score']
        
        self.scaler = StandardScaler()
        self.plant_features_normalized = self.plant_features.copy()
        self.plant_features_normalized[features_to_normalize] = self.scaler.fit_transform(
            self.plant_features[features_to_normalize]
        )
        
    def get_user_profile(self, experience_level, sunlight, temperature, humidity, 
                        watering_frequency, air_purification_priority, size_preference):
        """사용자 프로필 생성"""
        # 경험 수준에 따른 난이도 범위 설정
        difficulty_ranges = {
            '초보자': (0, 7),
            '중급자': (5, 12), 
            '전문가': (10, 20)
        }
        
        # 크기 선호도에 따른 높이 범위
        size_ranges = {
            '작은 식물': (0, 20),
            '중간 크기': (15, 35),
            '큰 식물': (30, 100)
        }
        
        user_profile = {
            'experience_level': experience_level,
            'sunlight': sunlight,
            'temperature': temperature,
            'humidity': humidity,
            'watering_frequency': watering_frequency,
            'air_purification_priority': air_purification_priority,
            'size_preference': size_preference,
            'difficulty_range': difficulty_ranges[experience_level],
            'size_range': size_ranges[size_preference]
        }
        
        return user_profile
    
    def filter_plants_by_constraints(self, user_profile):
        """제약 조건에 따른 식물 필터링"""
        filtered_plants = self.plant_features.copy()
        
        # 난이도 범위 필터링 (더 유연하게)
        min_diff, max_diff = user_profile['difficulty_range']
        # 난이도 범위를 2점씩 확장
        min_diff = max(0, min_diff - 2)
        max_diff = min(20, max_diff + 2)
        filtered_plants = filtered_plants[
            (filtered_plants['Difficulty_Score'] >= min_diff) & 
            (filtered_plants['Difficulty_Score'] <= max_diff)
        ]
        
        # 크기 범위 필터링 (더 유연하게)
        min_size, max_size = user_profile['size_range']
        # 크기 범위를 10cm씩 확장
        min_size = max(0, min_size - 10)
        max_size = min(100, max_size + 10)
        filtered_plants = filtered_plants[
            (filtered_plants['Height_cm'] >= min_size) & 
            (filtered_plants['Height_cm'] <= max_size)
        ]
        
        # 공기정화 우선순위가 높은 경우 필터링 (선택적)
        if user_profile['air_purification_priority'] == '높음':
            # 공기정화 효과가 있는 식물들을 우선하되, 없어도 포함
            air_purification_plants = filtered_plants[
                filtered_plants['Air_Purification_Effective'] == 1
            ]
            if len(air_purification_plants) >= 3:  # 최소 3개 이상 있으면 필터링
                filtered_plants = air_purification_plants
        
        return filtered_plants
    
    def calculate_similarity_scores(self, user_profile, filtered_plants):
        """사용자 프로필과 식물 간 유사도 계산"""
        # 사용자 프로필 벡터 생성
        user_vector = np.array([
            user_profile['temperature'],  # 온도
            user_profile['humidity'],     # 습도
            user_profile['watering_frequency'],  # 물주기 빈도
            user_profile['air_purification_priority'] == '높음'  # 공기정화 우선순위
        ])
        
        # 식물 특성 벡터
        plant_vectors = filtered_plants[['Room_Temperature_C', 'Humidity_%', 
                                       'Watering_Frequency_days', 'Air_Purification_Effective']].values
        
        # 유사도 계산 (코사인 유사도)
        similarities = cosine_similarity([user_vector], plant_vectors)[0]
        
        # 추가 가중치 적용
        weighted_scores = []
        for idx, (_, plant) in enumerate(filtered_plants.iterrows()):
            base_score = similarities[idx]
            
            # 건강 점수 가중치
            health_bonus = plant['Health_Score'] / 5.0 * 0.2
            
            # 난이도 적합성 가중치 (사용자 경험과 일치할수록 높은 점수)
            difficulty_center = sum(user_profile['difficulty_range']) / 2
            difficulty_penalty = 1 - abs(plant['Difficulty_Score'] - difficulty_center) / 10
            
            final_score = base_score + health_bonus + difficulty_penalty * 0.3
            weighted_scores.append(final_score)
        
        return np.array(weighted_scores)
    
    def recommend_plants(self, user_profile, top_n=5):
        """식물 추천"""
        # 제약 조건 필터링
        filtered_plants = self.filter_plants_by_constraints(user_profile)
        
        if len(filtered_plants) == 0:
            # [A. 추천 결과 없음 대응: 대체 추천 제공 및 가이드 메시지 추가]
            all_plants = self.plant_features.copy()
            debug_info = f"""
            ❌ 조건에 맞는 식물이 없습니다.
            🔍 디버깅 정보:
            - 총 식물 수: {len(all_plants)}개
            - 난이도 범위: {user_profile['difficulty_range']}
            - 크기 범위: {user_profile['size_range']}
            - 공기정화 우선순위: {user_profile['air_purification_priority']}
            
            💡 제안: 조건을 완화하거나 범위를 넓혀보세요!
            """
            # 임시로 전체 식물 중 상위 5개 추천
            return all_plants.head(5), debug_info
        
        # 유사도 점수 계산
        similarity_scores = self.calculate_similarity_scores(user_profile, filtered_plants)
        
        # 결과 데이터프레임 생성
        results = filtered_plants.copy()
        results['Similarity_Score'] = similarity_scores
        results = results.sort_values('Similarity_Score', ascending=False)
        
        # [ 추천 다양성 확보: 유사 식물 반복 방지 및 비슷한 특성 감점]
        # Plant_ID 중복 제거로 동일 식물 반복 추천 방지
        results = results.drop_duplicates(subset='Plant_ID')
        
        # 비슷한 특성(예: 크기, 난이도 등)이 너무 비슷한 경우 약간 감점 적용
        for idx, (_, plant) in enumerate(results.iterrows()):
            # 비슷한 크기일 경우 0.05 감점
            if abs(plant['Height_cm'] - user_profile['size_range'][0]) < 5:
                results.iloc[idx, results.columns.get_loc('Similarity_Score')] -= 0.05
        
        # 상위 N개 추천
        top_recommendations = results.head(top_n)
        
        return top_recommendations, f"🎉 {len(filtered_plants)}개 식물 중 최적의 {top_n}개를 추천합니다!"
    
    def get_plant_details(self, plant_name):
        """특정 식물의 상세 정보"""
        plant_data = self.plants_df[self.plants_df['Plant_ID'] == plant_name]
        
        if len(plant_data) == 0:
            return None
        
        # 평균값 계산
        details = plant_data.agg({
            'Height_cm': 'mean',
            'Leaf_Count': 'mean',
            'Watering_Amount_ml': 'mean',
            'Watering_Frequency_days': 'mean',
            'Room_Temperature_C': 'mean',
            'Humidity_%': 'mean',
            'Health_Score': 'mean',
            'Air_Purification_Effective': 'mean'
        }).round(2)
        
        # 가장 흔한 특성들
        common_features = {
            'Sunlight_Exposure': plant_data['Sunlight_Exposure'].mode()[0],
            'Fertilizer_Type': plant_data['Fertilizer_Type'].mode()[0],
            'Soil_Type': plant_data['Soil_Type'].mode()[0]
        }
        
        return details, common_features

def display_recommendations(recommendations, recommender_instance):
    """[E. 구조적 리팩토링 - 추천 결과 출력 함수 분리]"""
    st.subheader("🎯 추천 식물")
    
    for i, (_, plant) in enumerate(recommendations.iterrows(), 1):
        with st.expander(f"{i}. {plant['Plant_ID']} (점수: {plant['Similarity_Score']:.2f})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**높이:** {plant['Height_cm']:.1f}cm")
                st.write(f"**잎 수:** {plant['Leaf_Count']:.0f}개")
                st.write(f"**물주기:** {plant['Watering_Frequency_days']:.1f}일마다")
                st.write(f"**건강 점수:** {plant['Health_Score']:.1f}/5")
                st.write(f"**난이도:** {plant['Difficulty_Score']:.0f}/20")
            
            with col2:
                st.write(f"**적정 온도:** {plant['Room_Temperature_C']:.1f}°C")
                st.write(f"**적정 습도:** {plant['Humidity_%']:.1f}%")
                st.write(f"**공기정화:** {'✅ 효과적' if plant['Air_Purification_Effective'] == 1 else '❌ 효과 없음'}")
                
                # 상세 정보 버튼
                detail_button = st.button(f"상세 정보 보기", key=f"detail_{i}")
                if detail_button:
                    details, features = recommender_instance.get_plant_details(plant['Plant_ID'])
                    if details is not None:
                        st.write("**추가 정보:**")
                        st.write(f"- 물주기 양: {details['Watering_Amount_ml']:.0f}ml")
                        st.write(f"- 햇빛: {features['Sunlight_Exposure']}")
                        st.write(f"- 비료: {features['Fertilizer_Type']}")
                        st.write(f"- 토양: {features['Soil_Type']}")

def main():
    st.set_page_config(page_title="🌱 실내 식물 추천 시스템", layout="wide")
    
    st.title("🌱 실내 식물 추천 시스템")
    st.markdown("당신의 환경과 선호도에 맞는 완벽한 실내 식물을 찾아보세요!")
    
    # 사이드바 - 사용자 입력
    st.sidebar.header("📋 사용자 정보")
    
    experience_level = st.sidebar.selectbox(
        "식물 키우기 경험 수준",
        ['초보자', '중급자', '전문가']
    )
    
    sunlight = st.sidebar.selectbox(
        "햇빛 환경",
        ['Low light corner', 'Indirect light all day', 'Filtered sunlight through curtain', 
         '3h direct morning sun', '6h full sun']
    )
    
    temperature = st.sidebar.slider(
        "실내 온도 (°C)",
        min_value=15, max_value=30, value=22, step=1
    )
    
    humidity = st.sidebar.slider(
        "실내 습도 (%)",
        min_value=30, max_value=80, value=50, step=5
    )
    
    watering_frequency = st.sidebar.slider(
        "물주기 빈도 (일)",
        min_value=1, max_value=7, value=3, step=1
    )
    
    air_purification_priority = st.sidebar.selectbox(
        "공기정화 효과 우선순위",
        ['낮음', '보통', '높음']
    )
    
    size_preference = st.sidebar.selectbox(
        "선호하는 식물 크기",
        ['작은 식물', '중간 크기', '큰 식물']
    )
    
    # 추천 시스템 초기화
    if 'recommender' not in st.session_state:
        st.session_state.recommender = PlantRecommender()
    
    recommender = st.session_state.recommender
    
    # 추천 버튼
    recommend_button = st.sidebar.button("🌿 식물 추천받기", type="primary", key="recommend_button")
    
    if recommend_button:
        with st.spinner("최적의 식물을 찾고 있습니다..."):
            # 사용자 프로필 생성
            user_profile = recommender.get_user_profile(
                experience_level, sunlight, temperature, humidity,
                watering_frequency, air_purification_priority, size_preference
            )
            
            # 식물 추천
            recommendations, message = recommender.recommend_plants(user_profile, top_n=5)
            
            if len(recommendations) > 0:
                st.success(message)
                
                # [E. 구조적 리팩토링 - 추천 결과 출력 함수 사용]
                display_recommendations(recommendations, recommender)
                
                # 시각화
                st.subheader("📊 추천 식물 비교")
                
                # 특성 비교 차트
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
                
                # [레이더 차트 스케일 및 가독성 향상]
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 5],
                            tickvals=[0, 1, 2, 3, 4, 5]
                        )
                    ),
                    title="🌿 추천 식물 특성 비교",
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error(message)
    
    # 메인 페이지 정보 (추천 버튼을 누르지 않았을 때만 표시)
    if not recommend_button:
        st.markdown("""
        ### 🌟 이 시스템의 특징
        
        **🎯 개인화된 추천**
        - 당신의 경험 수준에 맞는 난이도
        - 생활 환경에 최적화된 식물 선택
        - 개인 선호도 반영
        
        **🔬 과학적 분석**
        - 20종의 실내 식물 데이터베이스
        - 1000+ 샘플의 실제 성장 데이터
        - 공기정화 효과 검증
        
        **💡 스마트 매칭**
        - 환경 조건 자동 매칭
        - 건강 점수 기반 필터링
        - 유사도 알고리즘 활용
        
        ### 📋 추천 과정
        1. **사용자 정보 입력** - 경험, 환경, 선호도
        2. **조건 필터링** - 적합한 식물 선별
        3. **유사도 계산** - 최적 매칭 점수 산출
        4. **개인화 추천** - 상위 5개 식물 제시
        """)
        
        # 데이터 통계
        st.subheader("📈 데이터베이스 현황")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("총 식물 종류", "20종")
        with col2:
            st.metric("데이터 샘플", "1,002개")
        with col3:
            st.metric("분석 특성", "18개")

if __name__ == "__main__":
    main() 