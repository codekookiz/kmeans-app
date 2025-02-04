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
if platform.system() == 'Linux':
    rc('font', family='NanumGothic')

font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
font_name = fm.FontProperties(fname = font_path).get_name()
plt.rc('font', family = font_name)


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
                print(f'이 컬럼의 유니크 개수 : {df[col].nunique()}') # >>==>> 임시 코드, 삭제 필요
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
            max_k = st.slider('K값 선택 (최대 그룹 개수)', min_value=2, max_value=df_new.shape[0])
        else :
            max_k = st.slider('K값 선택 (최대 그룹 개수)', min_value=2, max_value=10)
        st.text('')

        
        wcss = []
        for k in range(1, max_k + 1) :
            kmeans = KMeans(n_clusters=k, random_state=4)
            kmeans.fit(df_new)
            print(wcss.append(kmeans.inertia_))
        fig1 = plt.figure()
        plt.plot(range(1, max_k + 1), wcss)
        plt.title('엘보우 메소드')
        plt.xlabel('클러스터 개수')
        plt.ylabel('WCSS')
        st.pyplot(fig1)
        st.subheader('')

        st.info('원하는 클러스터 개수를 입력하세요.')
        k = st.number_input('숫자 입력', min_value=2, max_value=max_k)
        st.subheader('')

        
        

        kmeans = KMeans(n_clusters=k, random_state=4)
        df['Group'] = kmeans.fit_predict(df_new)

        st.info('그룹 정보가 저장되었습니다.')
        st.dataframe(df)


if __name__ == '__main__' :
    main()