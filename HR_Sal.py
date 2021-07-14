import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

st.title("Employee Salary Prediction")
print('\n')
    st.subheader("This project predicts the salary of the employee based on the given parameters")
print('\n')
print('\n')
st.markdown('Below plot confirms the relationship between the Salary and other input parameters')

add_csv = pd.read_csv('Final_HR.csv')

b = add_csv.corr()['Salary'].sort_values()
st.bar_chart(b)

# Normalization
X = add_csv.drop(['Salary'], axis=1)
y = add_csv['Salary']
mn = MinMaxScaler()
X1 = mn.fit_transform(X)

# User input parameters

st.sidebar.header('Select input parameters')

def Select_Parameters():

    maritaldesc = st.sidebar.selectbox('Select current Marital Status ', ('Single', 'Married', 'Divorced', 'Separated', 'Widowed'))
    if maritaldesc == 'Single':
        maritalstatusid = 0
    elif maritaldesc == 'Married':
        maritalstatusid = 1
    elif maritaldesc == 'Divorced':
        maritalstatusid = 2
    elif maritaldesc == 'Separated':
        maritalstatusid = 3
    else:
        maritalstatusid = 4

    empstatusid = st.sidebar.selectbox('Select Employment Status ID', (1, 2, 3, 4, 5))

    deptid = st.sidebar.selectbox('Select Department ID', (1, 2, 3, 4, 5, 6))

    racedesc = st.sidebar.selectbox('Select RaceDesc', ('Two or more races', 'American Indian or Alaska Native', 'White', 'Asian', 'Black or African American', 'Hispanic'))
    if racedesc == 'Two or more races':
        val = 59998.181818
    elif racedesc == 'American Indian or Alaska Native':
        val = 65806.000000

    elif racedesc == 'White':
        val = 67287.545455

    elif racedesc == 'Asian':
        val = 68521.206897

    elif racedesc == 'Black or African American':
        val = 74431.025000

    else:
        val = 83667.000000

    performancescore = st.sidebar.selectbox('Select Performance Score', ('Exceeds', 'Fully Meets', 'Needs Improvement', 'PIP'))
    if performancescore == 'Exceeds':
        val1 = 77144.864865
    elif performancescore == 'Fully Meets':
        val1 = 68366.720165
    elif performancescore == 'Needs Improvement':
        val1 = 68407.555556
    else:
        val1 = 58971.076923

    lreview_date = st.sidebar.selectbox('LastPerformanceReview_Date', (2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020))

    sex = st.sidebar.slider('Select Sex (O = Female, 1 = Male)', 0, 1)
    jobfairid = st.sidebar.slider('Select Job Fair ID', 0, 1)
    positionid = st.sidebar.slider('Select Position ID', 1, 30)
    managerid = st.sidebar.slider('Select Manager ID', 1, 30)
    empsatisfaction = st.sidebar.slider('Select Employee Satisfaction Score', 1, 5)
    specialprojectscount = st.sidebar.slider('Select Special Projects Count', 0, 7)
    dayslatelast = st.sidebar.slider('Select Days late last30', 0, 7)

    data = {'MaritalStatusID': maritalstatusid,
            'EmploymentStatus': empstatusid,
            'DeptID': deptid,
            'RaceDesc': val,
            'LastPerformanceReview_Date': lreview_date,
            'PerformanceScore': val1,
            'GenderID': sex,
            'FromDiversityJobFairID': jobfairid,
            'PositionID': positionid,
            'ManagerID': managerid,
            'EmpSatisfaction': empsatisfaction,
            'SpecialProjectsCount': specialprojectscount,
            'DaysLateLast30': dayslatelast}

    features = pd.DataFrame(data, index=[0])
    return features

df = Select_Parameters()

# Model Build
rn = RandomForestRegressor(n_estimators=250, max_features='auto')
rn.fit(X1, y)
prediction = rn.predict(df)
st.subheader('Prediction')
st.write('Predicted monthly salary of the employee in Rs is:')
st.write(prediction)




