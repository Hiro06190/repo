import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import folium
import geopandas as gpd
from scipy import integrate
import scipy.optimize
import math
import numpy as np
import warnings

warnings.simplefilter('ignore')

@st.cache
def fetch_data():
    df = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    #df.sort_index(ascending=True, inplace=True)
    return df
df = fetch_data()
region_list=list((df['location']).unique())
selected_region = st.sidebar.multiselect('Select a country', region_list, default='Afghanistan')


#options1={region_list}
df2 = df[(df['location'].isin(selected_region))]

options = {"Cumulative Cases": 'total_cases',
    "Daily Positive Tests": 'new_cases',
    "Cumluative Deaths": 'total_deaths',
    "Daily Deaths": 'new_deaths',
    "Reproduction Rate": 'reproduction_rate'}

st.title('COVID-19 Dashboard: World Data')
st.subheader('Source: https://ourworldindata.org/coronavirus')

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime(df.index.min()))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime(df.index.max()))

charts = st.sidebar.multiselect("Select individual charts to display:",
                options=list(options.keys()),
                default=list(options.keys())[0:1])

x = pd.date_range(datetime(2020,4,1), periods=(datetime(2021,1,31)-datetime(2020,4,1)).days, freq='d')

def func(x, a,b):
    return a*np.exp(-b/((x)))*(b+(x))/(2*math.pi*b*(x))# Levy

d1=datetime(2021,1,30)
d2=datetime(2020,4,1)
x = pd.date_range(d2, periods=(d1-d2).days, freq='d')
periods=(d1-d2).days
xx=np.linspace(1,periods,periods)

md = '<p style="font-family:Courier; color:Green; font-size: 20px;">By choosing Latvia, Angola, Bangladesh, Kenya, Malaysia on selecting box, the fitting function, which using stochastic model, of the cumulative cases is plotted</p>'

st.markdown(md, unsafe_allow_html=True)
fig = plt.figure(figsize=(8,6))
for region in selected_region:
    df_tmp = df2[df2['location'] == region]
    df_tmp1=df_tmp['total_cases'][(df_tmp.index >= d2) & (df_tmp.index < d1)]
    for chart in charts:
        df_tmp[options[chart]].plot(label=region + '_' + chart)
        if region == 'Latvia' or region == 'Angola' or region == 'Bangladesh' or region == 'Kenya' or region == 'Malaysia':
            if chart == 'Cumulative Cases':
                N=df_tmp1.values

                paramater_optimal, covariance = scipy.optimize.curve_fit(func, xx, N)
                y = func(xx,paramater_optimal[0],paramater_optimal[1])
                A=paramater_optimal[0]
                B=paramater_optimal[1]
                C=N[0]
                y2=C+A*np.exp(-B/((xx)))*(B+(xx))/(2*math.pi*B*(xx))
                plt.plot(x,y2,label='fitting_by_stochastic_model'+'_'+region + '_' + chart)
                plt.legend()
            
    plt.xlabel('Date')
    plt.legend(loc="upper left")
st.pyplot(fig)
#with col2:
m = folium.Map(location=[50, 0], zoom_start=1)

gdf = gpd.read_file('https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json')

df3=df[(df.index.date==end_date - timedelta(days=5))]

df3A=df3.query("reproduction_rate>=1")
df3B=df3.query("reproduction_rate<1")
# 地図に色を塗る
gdf['Group'] = ''
gdf.loc[gdf['name'].isin(df3A['location']), 'Group'] = 'Reproduction rate≥1'
gdf.loc[gdf['name'].isin(df3B['location']), 'Group'] = 'Reproduction rate<1'



m = folium.Map(location=[0, 0], zoom_start=2)

folium.GeoJson(gdf[gdf['Group']=='Reproduction rate≥1'],
               # レンダリングする表示の線や色の設定ができます
               style_function=lambda x: {'fillColor': 'red', 'color': 'red'},
               # マウスホバー時のツールチップの設定ができます
               tooltip=folium.features.GeoJsonTooltip(fields=['name', 'Group'],labels=True, sticky=True)
               ).add_to(m)
folium.GeoJson(gdf[gdf['Group']=='Reproduction rate<1'],
               style_function=lambda x: {'fillColor': 'blue', 'color': 'blue'},
               tooltip=folium.features.GeoJsonTooltip(fields=['name', 'Group'],labels=True, sticky=True)
               ).add_to(m)
m
