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
st.markdown("<p style='color:#FF0000'><tt>Stockmate implement machine learning approach to predict stock prices. Machine learning is effectively implemented in forecasting stock prices. The objective is to predict the stock prices in order to make more informed and accurate investment decisions. We propose a stock price prediction system that integrates mathematical functions, machine learning, and other external factors for the purpose of achieving better stock prediction accuracy and issuing profitable trades.</tt></p>", unsafe_allow_html=True)
input = st.selectbox(
     'Enter or select Stock Tiker ',
     ('AAPL','MSFT','GOOG','GOOGL','AMZN','TSLA','UNH','JNJ','META','NVDA'))

st.write('You selected:', input)


# input=st.text_input("enter stock ticker",'AAPL')
df=data.DataReader(input,'yahoo',start,end)

st.subheader("Data from 2010-2021")
st.write(df.describe())
with st.expander("See explanation"):
     st.write(
       "High - The high is the highest price at which a stock is traded during a period.\n")
     st.write(" Open -  Open means the price at which a stock started trading when the opening bell rang.\n")
     st.write(" Close - Close refers to the price of an individual stock when the stock exchange closed shop for the day.\n")

st.subheader("closing price vs time chart")
fig =pl.figure(figsize=(12,6))
pl.legend(["price VS time"])
pl.xlabel("Time(years)")
pl.ylabel("price")
pl.plot(df.Close)
st.pyplot(fig)
with st.expander("See explanation"):
     st.write(" CLOSING PRICE - The closing price is the raw price or cash value of the last transacted price in a security before the market officially closes for normal trading.\n")
     st.write(" TIME CHART - A time frame refers to the amount of time that a trend lasts for in a market, which can be identified and used by traders.\n")

st.subheader("closing price vs time chart with 100MA and 200MA")
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig =pl.figure(figsize=(12,6))
a=pl.plot(ma100, 'r')
b=pl.plot (ma200 ,'g')
c=pl.plot(df.Close, 'b')
pl.legend(["RED - MA100","GREEN - MA200","Blue - Closing price"])
pl.xlabel("Time(years)")
pl.ylabel("price")
st.pyplot(fig)
with st.expander("See explanation"):
     st.write("""
         MA - The moving average (MA) is a simple technical analysis tool that smooths out price data by creating a constantly updated average price. The average is taken over a specific period of time, like 10 days, 20 minutes, 30 weeks, or any time period the trader chooses.
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
pl.legend(["Blue - ORIGINAL","GREEN - Predicted"])
st.pyplot(fig2)
with st.expander("See explanation"):
     st.write("""
        The prediction is  shown by Green line and the actual trend is  shown by blue. The proximity  of  these  two  lines  tells,  how  efficient  the LSTM based model is. The prediction approximates real trend when a  considerable amount  of time  has passed . This project is not responsible for your loss in investment . This project is for planning of your investment .
     """)








    
    
    
    
    
    