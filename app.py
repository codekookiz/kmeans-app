import streamlit as st, pandas as pd, matplotlib.pyplot as plt, matplotlib.font_manager as fm, platform, os
from pandas.api.types import is_integer_dtype, is_float_dtype, is_object_dtype
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
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


def main() :
    fontRegistered()
    plt.rc('font', family='NanumGothic')

    st.title('K-Means 클러스터링 앱')
    st.subheader('')

    # 1. csv 파일 업로드
    file = st.file_uploader('CSV 파일 업로드', type=['csv'])
    if file is not None :
        # 2. 데이터 불러오기
        df = pd.read_csv(file)
        st.dataframe(df.head())
        st.info('NaN 데이터가 존재할 경우 해당 행을 삭제합니다.')
        st.dataframe(df.isna().sum())
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        st.subheader('')
        # 3. 유저가 컬럼 선택 가능하도록 함
        st.info('K-Means 클러스터링에 사용할 컬럼을 선택해주세요.')
        selected_columns = st.multiselect('컬럼 선택', df.columns)
        if len(selected_columns) == 0 :
            return
        df_new = pd.DataFrame()
        # 4. 문자인지 숫자인지 판별하는 코드
        for col in selected_columns :
            if is_integer_dtype(df[col]) :
                df_new[col] = df[col]
            elif is_float_dtype(df[col]) :
                df_new[col] = df[col]
            elif is_object_dtype(df[col]) :
                if df[col].nunique() >= 3 :
                    # 원핫 인코딩
                    ct = ColumnTransformer([('onehot', OneHotEncoder(), [0])], remainder='passthrough')
                    col_names = sorted(df[col].unique())
                    df_new[col_names] = ct.fit_transform(df[col].to_frame())
                else :
                    # 레이블 인코딩
                    encoder = LabelEncoder()
                    df_new[col] = encoder.fit_transform(df[col])
            else :
                st.text(f'{col} 컬럼은 K-Means 클러스터링이 불가능하기에 클러스터링 프로세스에서 제외합니다.')
        st.text('')

        st.info('아래는 K-Means 클러스터링 수행을 위한 데이터프레임입니다.')
        st.dataframe(df_new)
        st.subheader('')

        st.subheader('최적의 K값을 결정하기 위해 WCSS를 구합니다.')
        st.text('')

        # 데이터 개수가 클러스터링 개수 이상이어야하므로, 데이터 개수로 K값의 최댓값을 결정
        st.info(f'데이터의 개수는 {df_new.shape[0]}입니다.')
        if df_new.shape[0] < 10 :
            max_k = df_new.shape[0]
        else :
            max_k = 10
        st.text('')

        
        wcss = []
        for k in range(1, max_k + 1) :
            kmeans = KMeans(n_clusters=k, random_state=4)
            kmeans.fit(df_new)
            print(wcss.append(kmeans.inertia_))
        fig1 = plt.figure()
        plt.plot(range(1, max_k + 1), wcss)
        plt.title('엘보우 메소드')
        plt.xlabel('클러스터(그룹) 개수')
        plt.ylabel('WCSS')
        st.pyplot(fig1)
        st.subheader('')

        #st.info('원하는 클러스터 개수를 입력하세요.')
        #k = st.number_input('숫자 입력', min_value=2, max_value=max_k)
        #st.subheader('')

        if max_k == 3 :
            if (wcss[0] - wcss[1]) / (wcss[1] - wcss[2]) >= 2 :
                k = 2
            else :
                if wcss[0] / min(wcss) >= 2 :
                    k = 3
                else :
                    k = 1
        elif max_k == 2 :
            if (wcss[0] - wcss[1]) / wcss[1] >= 1 :
                k = 2
            else :
                k = 1
        else : 
            best = []
            cnt = 0
            for a in range(2, max_k - 1) :
                if wcss[a - 1] - wcss[a + 1] != 0 :
                    new_delta = (wcss[a - 2] - wcss[a]) / (wcss[a - 1] - wcss[a + 1])
                    if new_delta >= 2 :
                        best.append(a)
                else :
                    if cnt == 0 :
                        best.append(a)
                        cnt += 1
                    else :
                        continue
            if len(best) != 0 :
                k = max(best)
            else : 
                k = max_k

        st.text(f'최적의 클러스터(그룹) 개수는 {k}개입니다.')
        kmeans = KMeans(n_clusters=k, random_state=4)
        df['Group'] = kmeans.fit_predict(df_new)

        st.info('그룹 정보가 저장되었습니다.')
        st.dataframe(df)


if __name__ == '__main__' :
    main()