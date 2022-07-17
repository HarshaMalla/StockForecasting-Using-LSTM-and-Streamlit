import os
from matplotlib import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import pandas_datareader as data
from keras.models import load_model
import streamlit as st


start= '2010-01-01'
end= '2022-6-30'

st.title("Stock Mate")
def header(url):
     st.header(f'<p style="color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)
st.markdown(" Stockmate implement machine learning approach to predict stock prices. Machine learning is effectively implemented in forecasting stock prices. The objective is to predict the stock prices in order to make more informed and accurate investment decisions. We propose a stock price prediction system that integrates mathematical functions, machine learning, and other external factors for the purpose of achieving better stock prediction accuracy and issuing profitable trades.")
input = st.selectbox(
     'Enter or select Stock Tiker ',
     ('AAPL','MSFT','GOOG','GOOGL','AMZN','TSLA','UNH','JNJ','META','NVDA'))

st.write('You selected:', input)


# input=st.text_input("enter stock ticker",'AAPL')
df=data.DataReader(input,'yahoo',start,end)

st.subheader("Data from 2010-2021")
st.write(df.describe())
with st.expander("See explanation"):
     st.write("""
        Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.
     """)

st.subheader("closing price vs time chart")
fig =pl.figure(figsize=(12,6))
pl.plot(df.Close)
st.pyplot(fig)
with st.expander("See explanation"):
     st.write("""
        Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.
     """)

st.subheader("closing price vs time chart with 100MA and 200MA")
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig =pl.figure(figsize=(12,6))
pl.plot(ma100, 'r')
pl.plot (ma200 ,'g')
pl.plot(df.Close, 'b')
st.pyplot(fig)
with st.expander("See explanation"):
     st.write("""
        Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.
     """)

data_training= pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing= pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler(feature_range=(0,1))

data_trainig_array = scaler.fit_transform(data_training)
x_train=[]
y_train=[]
for i in range(100,data_trainig_array.shape[0]):
    x_train.append(data_trainig_array[i-100:i])
    y_train.append(data_trainig_array[i,0])
    
x_train,y_train = np.array(x_train),np.array(y_train)

model=load_model('keras_model.h5')

past_100 = data_training.tail(100)
final_df= past_100.append(data_testing,ignore_index=True)

input_data= scaler.fit_transform(final_df)
x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test,y_test = np.array(x_test),np.array(y_test)
y_predicted=model.predict(x_test)

scale_factor=1/scaler.scale_[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor   


st.subheader('Predictions vs Original')
fig2=pl.figure(figsize=(12,6))
pl.plot(y_test,'b')
pl.plot(y_predicted)
st.pyplot(fig2)
with st.expander("See explanation"):
     st.write("""
        Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.
     """)








    
    
    
    
    
    