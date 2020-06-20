import neurokit as nk
import h5py
import pandas as pd
import numpy as np
import streamlit as st
import plotly as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from predict import diagnose
from PIL import Image
import io

#Functions
def plot_r_systole(df):
    fig = make_subplots(rows=2, cols=1, subplot_titles=("ECG Signal (filtered) with R-waves Marked", "ECG Systole and Diastole Information"))
    fig.update_layout(title = 'View Uploaded ECG Signal', width=1000, height=800)
    
    df['R_Peaks'] = df.loc[:,'ECG_Raw']*df.loc[:,'ECG_R_Peaks']
    
    columns = ['ECG_Raw', 'ECG_Filtered', 'R_Peaks']
    for i in columns:
        trace = go.Scatter(
            x = df.index,
            y = df.loc[:,i],
            name = i,
            mode = ['lines' if i!='R_Peaks' else 'markers'][0]
        )
        fig.add_trace(trace, row=1, col=1)
    fig.update_yaxes(title_text="Amplitude (in Volts)", row=1, col=1)
    fig.update_xaxes(title_text="Time (in seconds)", row=1, col=1)
    
    columns2 = ['ECG_Systole', 'ECG_R_Peaks']
    for j in np.arange(1,-1,-1):
        trace = go.Scatter(
            x = df.index,
            y = df.loc[:,columns2[j]],
            name = ["1-Systole,0-Diastole", "R_Peaks"][j],
            mode = ['lines','markers'][j]
        )
        fig.add_trace(trace, row=2, col=1)
    fig.update_yaxes(title_text="Systole/Diastole", row=2, col=1)
    fig.update_xaxes(title_text="Time (in seconds)", row=2, col=1)

    return fig

def plot_cardiac_cycles(df):
    data = []
    for i in range(len(df.columns)):
        trace = go.Scatter(
            x = np.arange(len(df))/400,
            y = df.iloc[:,i],
            mode = 'lines'
        )
        data.append(trace)
    layout = go.Layout(title = "Compare and Match all Cardiac Cycles in the Sample", showlegend=False, 
                       yaxis=dict(title="Amplitude (in Volts)"), xaxis=dict(title="Time (in seconds)"))
    fig = go.Figure(data=data, layout=layout)
    return fig, len(df.columns)
    
def disp_predict(arr):
    Max = max(arr)
    ind = np.where(arr == Max)[0][0]
    cond = {0:"First Degree AV Block", 1: "Right Bundle Branch Block", 2:"Left Bundle Branch Block", 
            3:"Sinus Bradycardia", 4:"Atrial Fibrillaton", 5:"Sinus Tachycardia"}
    if Max < 1e-2:
        return "Prognosis is Normal. No anomalies detected."
    else:
        return "Prognosis is " + cond[ind] + ". Consult a doctor at the earliest."

@st.cache
def load():
    i = []
    m = []
    image = Image.open(r'C:\Users\Aditi\Downloads\automatic-ecg-diagnosis-master\outputs\figures\model.png')
    m.append(image)
    image = Image.open(r'C:\Users\Aditi\Downloads\automatic-ecg-diagnosis-master\outputs\figures\model1.png')
    m.append(image)
    image = Image.open(r'C:\Users\Aditi\Downloads\automatic-ecg-diagnosis-master\outputs\figures\model2.png')
    m.append(image)
    image = Image.open(r'C:\Users\Aditi\Downloads\automatic-ecg-diagnosis-master\outputs\figures\boxplot_bootstrap_Precision-1.jpg')
    i.append(image)
    image = Image.open(r'C:\Users\Aditi\Downloads\automatic-ecg-diagnosis-master\outputs\figures\boxplot_bootstrap_Recall-1.jpg')
    i.append(image)
    image = Image.open(r'C:\Users\Aditi\Downloads\automatic-ecg-diagnosis-master\outputs\figures\boxplot_bootstrap_Specificity-1.jpg')
    i.append(image)
    image = Image.open(r'C:\Users\Aditi\Downloads\automatic-ecg-diagnosis-master\outputs\figures\boxplot_bootstrap_F1 score-1.jpg')
    i.append(image)
    ab = [ [35759, (1.5), 28, (3.4)],[63528, (2.7), 34 ,(4.1)],[39842, (1.7), 30, (3.6)],[37949 ,(1.6), 16 ,(1.9)],[41862 ,(1.8), 13 ,(1.6)],
      [49872 ,(2.1), 36, (4.4)]]
    df = pd.DataFrame(ab, columns = ["Train  +Val(n = 2,322,513)", "%", "Test (n=827)", "%"])
    df["Diagnosis Classes"] = ["1dAVb","RBBB","LBBB","SB", "AF", "ST"]
    df = df.set_index("Diagnosis Classes")
    classes = pd.DataFrame(["1st degree AV block (1dAVb)", "Right Bundle Branch Block (RBBB)", 
                  "Left Bundle Branch Block (LBBB)", "Sinus Bradycardia (SB)","Atrial Fibrillation (AF)", 
                  "Sinus Tachycardia (ST)"], columns = ["Diagnosis Classes"])    
    return classes, df, m, i
    
def conf():
    import seaborn as sn
    d = [[795,   4],
           [  2,  26],
           [814,   0],
           [  3,  10],
           [797,   0],
           [  0,  30],
           [789,   4],
           [  0,  34],
           [808,   3],
           [  1,  15],
           [788,   2],
           [  1,  36]]
    df = pd.DataFrame(d)
    df.columns = ['True Negative','True Positive']
    df['index'] = ['1dAVb -', '1dAVb +', 'RBBB -', 'RBBB +', 'LBBB -', 'LBBB +', 'SB -', 'SB +', 'AF -', 'AF +', 'ST -', 'ST +']
    df = df.set_index('index')
    df_norm_col=(df-df.mean())/df.std()
    st.write(df)
    sn.heatmap(df)
    st.pyplot()
    st.write('Normalized Confusion Matrix')
    sn.heatmap(df_norm_col, cmap='Blues')
    st.pyplot()
    

page = st.sidebar.selectbox("Choose a page", ["Diagnostic View", "Model Summary"])
if page == "Diagnostic View":
    st.header("Project Title : MultiPurpose Medical Diagnostic System")
    st.header("This is a Heart-Condition Diagnosis System using MultiChannel ECG Signals")
    c, df, model, image = load()
    st.header("Dataset Summary")
    st.write("The Dataset consisted of ECG 'Heart Signals' with six different abnormalities with the following distributions:  ")
    st.write(c)
    st.write(df)
    st.write('Shape of one sample : (4096,12) where 12 represents each channel of the ECG')
    uploaded_file = st.text_input('Enter file path of test case(s) :')
    if uploaded_file is not None:
        with h5py.File(uploaded_file, "r") as f:
            test_data = np.array(f['tracings'])
        if len(test_data) > 1:
            test = st.selectbox("Select the sample number you want to analyze: ", np.arange(1,len(test_data)+1))
            raw_ecg = np.ravel(test_data[test-1].T[1])
        else:
            raw_ecg = np.ravel(test_data.T[1])

        processed_ecg = nk.ecg_process(raw_ecg, sampling_rate=400, quality_model = None, hrv_features=['time'])

        figure = plot_r_systole(processed_ecg['df'])
        st.write(figure)

        fig, cc = plot_cardiac_cycles(processed_ecg['ECG']['Cardiac_Cycles'])
        st.write(fig)

        hr = round(np.mean(processed_ecg['df']['Heart_Rate']))
        mrr = round(processed_ecg['ECG']['HRV']['meanNN']/400,2)

        prediction, model = diagnose()
        if len(test_data) > 1:
            pred = prediction[test-1]
        else:
            if len(pred)==1:
                pred = prediction[0]
        disp = disp_predict(pred)


        st.header("Diagnosis : ")
        st.write("Resting Heart Rate is Calculated as: "+ str(hr) + " beats per minute")
        st.write("Mean R-R Interval is: " + str(mrr) + " seconds")
        st.write("Number of Cardiac Cycles in the Sample = " + str(cc))
        st.write("The probabilities for each of the classes are: ")
        st.write(pd.DataFrame([["1dAVb","RBBB","LBBB","SB", "AF", "ST"],prediction[test-1]]).T)
        st.header(disp)

elif page == "Model Summary":
    c, df, model, image = load()
#     st.header("Dataset Summary")
#     st.write("The Dataset consisted of ECG 'Heart Signals' with six different abnormalities with the following distributions:  ")
#     st.write(c)
#     st.write(df)
#     st.write('Shape of one sample : (4096,12) where 12 represents each channel of the ECG')
    st.header("The Model used is a Residual Network")
    st.header("Model Summary is as follows: ")
    st.image(model, caption=["Model Architecture","", "Model Summary"], use_column_width=True)
    st.header("Confusion Matrix")
    conf()
    st.header("Displayed here are Precision, Specificty and F1 Recall scores of the training set")
    st.image(image, caption=["Precision BoxPlots", "Recall BoxPlots","Specificity BoxPlots","F1 Scores"], use_column_width=True)
    
    
    






