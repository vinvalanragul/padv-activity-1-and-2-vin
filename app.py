import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from pathlib import Path
st.set_page_config(page_title='Renewable Energy Intelligence',page_icon='🌱',layout='wide')
st.title('🌱 Renewable Energy Intelligence')
st.caption('Historical renewable energy analysis and next-year prediction | 1960–2015 source data')
@st.cache_data
def clean(data):
 from io import BytesIO
 d=pd.read_csv(BytesIO(data)); d['Value']=pd.to_numeric(d['Value'],errors='coerce');d['TIME']=pd.to_numeric(d['TIME'],errors='coerce')
 d=d.dropna(subset=['LOCATION','MEASURE','TIME','Value']);d=d[d.MEASURE.isin(['KTOE','PC_PRYENRGSUPPLY'])];d=d.groupby(['LOCATION','MEASURE','TIME'],as_index=False).Value.mean();d['TIME']=d.TIME.astype(int)
 return d
source=Path(__file__).with_name('renewable_energy.csv')
with st.sidebar:
 st.header('Data source')
 upload=st.file_uploader('Upload renewable_energy.csv',type='csv')
 data=upload.getvalue() if upload else (source.read_bytes() if source.exists() else None)
if data is None:
 st.info('Upload the dataset to begin. Add renewable_energy.csv next to app.py to load it automatically.');st.stop()
try:d=clean(data)
except Exception as e:st.error(f'Invalid dataset: {e}');st.stop()
with st.sidebar:
 locations=sorted(d.LOCATION.unique());location=st.selectbox('Location',locations,index=locations.index('AUS') if 'AUS' in locations else 0)
 measure=st.radio('Measure',['KTOE','PC_PRYENRGSUPPLY'],format_func=lambda x:'Renewable supply (KTOE)' if x=='KTOE' else 'Renewable share of primary supply (%)')
 lo,hi=int(d.TIME.min()),int(d.TIME.max());period=st.slider('Year range',lo,hi,(lo,hi))
 st.caption('LOCATION values are source codes; some codes may represent aggregates rather than countries.')
view=d[(d.MEASURE==measure)&d.TIME.between(*period)]
loc=view[view.LOCATION==location].sort_values('TIME')
unit='KTOE' if measure=='KTOE' else '%'
a,b,c,e=st.columns(4)
a.metric('Valid observations',f'{len(view):,}')
b.metric('Selected location',location)
c.metric('Latest observed value',f'{loc.Value.iloc[-1]:,.2f} {unit}' if len(loc) else 'N/A')
e.metric('Last observed year',str(loc.TIME.iloc[-1]) if len(loc) else 'N/A')
t1,t2,t3,t4=st.tabs(['📈 Trends','🌍 Comparison','🤖 Prediction','📋 Data'])
with t1:
 if len(loc):
  st.plotly_chart(px.line(loc,x='TIME',y='Value',markers=True,title=f'{location} — renewable energy trend ({unit})'),use_container_width=True)
  st.plotly_chart(px.bar(loc.tail(15),x='TIME',y='Value',title='Most recent 15 observations'),use_container_width=True)
 else:st.warning('No data for selected location and period.')
with t2:
 year=st.select_slider('Compare locations in year',options=sorted(view.TIME.unique())) if len(view) else None
 if year is not None:
  comparison=view[view.TIME==year].nlargest(15,'Value').sort_values('Value')
  st.plotly_chart(px.bar(comparison,x='Value',y='LOCATION',orientation='h',title=f'Top 15 location codes — {year} ({unit})'),use_container_width=True)
  st.caption('Aggregate regions and countries may both appear; this is not necessarily a country-only ranking.')
with t3:
 st.subheader('One-year-ahead prediction')
 st.caption('Uses 3 previous annual observations to predict the next year. Chronological holdout testing prevents future-to-past leakage. This is an exploratory forecast, not a verified future outcome.')
 @st.cache_data
 def lagged(data,metric):
  base=data[data.MEASURE==metric].sort_values(['LOCATION','TIME']).copy()
  for lag in (1,2,3):
   previous=base[['LOCATION','TIME','Value']].copy();previous.TIME+=lag;previous=previous.rename(columns={'Value':f'lag{lag}'})
   base=base.merge(previous,on=['LOCATION','TIME'],how='left')
  return base.dropna(subset=['lag1','lag2','lag3'])
 train=lagged(d,measure)
 if len(train)<100:st.warning('Insufficient continuous-year histories to train.');st.stop()
 years=sorted(train.TIME.unique());cut=years[max(1,int(len(years)*.8))-1]
 tr=train[train.TIME<=cut];te=train[train.TIME>cut]
 if len(te)==0:st.warning('Insufficient holdout years.');st.stop()
 features=['lag1','lag2','lag3']
 @st.cache_resource
 def fit(metric,raw):
  base=clean(raw)
  allrows=lagged(base,metric);years=sorted(allrows.TIME.unique());cut=years[max(1,int(len(years)*.8))-1]
  trainrows=allrows[allrows.TIME<=cut];testrows=allrows[allrows.TIME>cut]
  m=RandomForestRegressor(n_estimators=120,min_samples_leaf=3,random_state=42,n_jobs=-1)
  m.fit(trainrows[features],trainrows.Value)
  p=m.predict(testrows[features]);return m,cut,mean_absolute_error(testrows.Value,p),np.sqrt(mean_squared_error(testrows.Value,p)),r2_score(testrows.Value,p),len(testrows)
 model,cut,mae,rmse,r2,n=fit(measure,data)
 a,b,c=st.columns(3);a.metric('Test MAE',f'{mae:,.2f}');b.metric('Test RMSE',f'{rmse:,.2f}');c.metric('Test R²',f'{r2:.3f}')
 st.caption(f'Training through {cut}; testing after {cut} ({n:,} location-year records). Metrics are in {unit} except R².')
 whole=d[(d.LOCATION==location)&(d.MEASURE==measure)].sort_values('TIME')
 last=whole.tail(3)
 if len(last)==3 and list(np.diff(last.TIME))==[1,1]:
  prediction=float(model.predict(pd.DataFrame([{'lag1':last.Value.iloc[-1],'lag2':last.Value.iloc[-2],'lag3':last.Value.iloc[-3]}]))[0]);next_year=int(last.TIME.iloc[-1]+1)
  st.metric(f'Predicted {next_year} value for {location}',f'{prediction:,.2f} {unit}')
  st.caption('Model uses the latest three consecutive observed years for this location. It does not use future observations.')
 else:st.warning('This location has fewer than three consecutive observations at the end of its series; prediction unavailable.')
with t4:
 st.dataframe(view,use_container_width=True)
 st.download_button('Download filtered CSV',view.to_csv(index=False).encode(),'renewable_energy_filtered.csv','text/csv')
 st.caption('Missing Value entries are excluded; Flag Codes and constant metadata are not used for modeling.')
