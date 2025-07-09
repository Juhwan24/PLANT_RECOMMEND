import pandas as pd

# Difficulty_Weights.xlsx 파일 확인
print("=== Difficulty_Weights.xlsx 파일 분석 ===")
try:
    df_weights = pd.read_excel('Difficulty_Weights.xlsx')
    print("데이터 미리보기:")
    print(df_weights.head())
    print("\n컬럼명:")
    print(df_weights.columns.tolist())
    print(f"\n데이터 형태: {df_weights.shape}")
    print("\n데이터 타입:")
    print(df_weights.dtypes)
except Exception as e:
    print(f"Error reading Difficulty_Weights.xlsx: {e}")

print("\n" + "="*50)

# Indoor_Plant_With_AirPurification_Label.csv 파일 확인
print("=== Indoor_Plant_With_AirPurification_Label.csv 파일 분석 ===")
try:
    df_plants = pd.read_csv('Indoor_Plant_With_AirPurification_Label.csv')
    print("데이터 미리보기:")
    print(df_plants.head())
    print("\n컬럼명:")
    print(df_plants.columns.tolist())
    print(f"\n데이터 형태: {df_plants.shape}")
    print("\n고유한 식물 종류:")
    print(df_plants['Plant_ID'].unique())
    print(f"총 {len(df_plants['Plant_ID'].unique())}종의 식물")
except Exception as e:
    print(f"Error reading Indoor_Plant_With_AirPurification_Label.csv: {e}") 