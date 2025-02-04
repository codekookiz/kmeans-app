import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os
import seaborn as sb
from pandas.api.types import is_integer_dtype, is_float_dtype, is_object_dtype
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from matplotlib import rc

@st.cache_data
def fontRegistered():
    font_dirs = [os.getcwd() + '/custom_fonts']
    font_files = fm.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    fm._load_fontmanager(try_read_cache=False)

plt.rcParams['axes.unicode_minus'] = False
system_os = platform.system()
if system_os == "Darwin":  # macOS
    font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
elif system_os == "Windows":  # Windows
    font_path = "C:/Windows/Fonts/malgun.ttf"
else:  # Linux
    rc('font', family='NanumGothic')

def main():
    fontRegistered()
    plt.rc('font', family='NanumGothic')

    st.title('🎨 K-Means 클러스터링 앱')
    st.markdown("---")
    
    st.info('사용자가 업로드한 CSV 파일을 분석하여 클러스터링을 진행합니다.')
    file = st.file_uploader('📂 CSV 파일 업로드', type=['csv'])
    
    if file is not None:
        df = pd.read_csv(file)
        st.subheader('📊 데이터 미리보기')
        st.dataframe(df.head())
        
        st.warning('NaN 데이터가 존재할 경우 해당 행을 삭제합니다.')
        st.dataframe(df.isna().sum())
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        st.markdown("---")
        st.subheader('🔍 클러스터링에 사용할 컬럼 선택')
        selected_columns = st.multiselect('컬럼 선택', df.columns)
        if not selected_columns:
            return
        
        df_new = pd.DataFrame()
        for col in selected_columns:
            if is_integer_dtype(df[col]) or is_float_dtype(df[col]):
                df_new[col] = df[col]
            elif is_object_dtype(df[col]):
                if df[col].nunique() >= 3:
                    ct = ColumnTransformer([('onehot', OneHotEncoder(), [0])], remainder='passthrough')
                    col_names = sorted(df[col].unique())
                    df_new[col_names] = ct.fit_transform(df[col].to_frame())
                else:
                    encoder = LabelEncoder()
                    df_new[col] = encoder.fit_transform(df[col])
            else:
                st.error(f'🚨 {col} 컬럼은 클러스터링에서 제외됩니다.')
                
        scaler = StandardScaler()
        df_new = pd.DataFrame(scaler.fit_transform(df_new), columns=df_new.columns)
        
        st.subheader('📌 클러스터링을 위한 데이터')
        st.dataframe(df_new)
        
        st.subheader('📈 최적의 K값 결정')
        max_k = min(10, df_new.shape[0])
        st.info(f'데이터 개수: {df_new.shape[0]} / 최대 클러스터 개수: {max_k}')
        
        wcss = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=4, n_init=10)
            kmeans.fit(df_new)
            wcss.append(kmeans.inertia_)
        
        fig1 = plt.figure()
        plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--')
        plt.title('엘보우 메소드')
        plt.xlabel('클러스터 개수')
        plt.ylabel('WCSS')
        st.pyplot(fig1)
        
        # 최적의 K 결정
        if max_k == 3:
            k = 2 if (wcss[0] - wcss[1]) / (wcss[1] - wcss[2]) >= 2 else (3 if wcss[0] / min(wcss) >= 2 else 1)
        elif max_k == 2:
            k = 2 if (wcss[0] - wcss[1]) / wcss[1] >= 1 else 1
        else:
            best = [a for a in range(2, max_k - 1) if (wcss[a - 1] - wcss[a + 1]) != 0 and (wcss[a - 2] - wcss[a]) / (wcss[a - 1] - wcss[a + 1]) >= 2]
            k = max(best) if best else max_k
        
        st.subheader(f'🎯 최적의 클러스터 개수: {k}개')
        
        kmeans = KMeans(n_clusters=k, random_state=4, n_init=10)
        df['Group'] = kmeans.fit_predict(df_new)
        st.success('✅ 그룹 정보가 저장되었습니다.')
        st.dataframe(df)
        
        st.subheader('🎨 클러스터 시각화')
        fig2 = plt.figure()
        palette = sns.color_palette("tab10", k)
        sb.scatterplot(x=df_new.iloc[:, 0], y=df_new.iloc[:, 1], hue=df['Group'], palette=palette, legend='full')
        plt.xlabel(selected_columns[0])
        plt.ylabel(selected_columns[1] if len(selected_columns) > 1 else 'Feature 2')
        plt.title('클러스터링 결과')
        st.pyplot(fig2)

if __name__ == '__main__':
    main()