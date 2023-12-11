import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

D_clients = pd.read_csv('D_clients.csv')
D_close_loan = pd.read_csv('D_close_loan.csv')
D_job = pd.read_csv('D_job.csv')
D_last_credit = pd.read_csv('D_last_credit.csv')
D_loan = pd.read_csv('D_loan.csv')
D_pens = pd.read_csv('D_pens.csv')
D_salary = pd.read_csv('D_salary.csv')
D_target = pd.read_csv('D_target.csv')
D_work = pd.read_csv('D_work.csv')

D_clients=D_clients.rename(columns={'ID':'ID_CLIENT'})
D_salary.drop_duplicates(inplace=True)

D_all = D_clients.merge(D_target, on='ID_CLIENT', how='inner')
D_all = D_all.merge(D_salary, on='ID_CLIENT', how='inner')
D_all = D_all.merge(D_job, on='ID_CLIENT', how='inner')
D_all = D_all.merge(D_last_credit, on='ID_CLIENT', how='inner')

D_loan_all = D_loan.merge(D_close_loan, on='ID_LOAN')
loan = pd.DataFrame()
loan['ID_CLIENT'] = D_loan_all.groupby('ID_CLIENT').size().index
loan['LOAN_NUM_TOTAL'] = D_loan_all.groupby('ID_CLIENT').size().values
loan['LOAN_NUM_CLOSED'] = D_loan_all.groupby('ID_CLIENT')['CLOSED_FL'].sum().values

D_all = D_all.merge(loan, on='ID_CLIENT', how='inner')
df = D_all[['AGREEMENT_RK', 'TARGET', 'AGE', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL',
            'GENDER','CHILD_TOTAL', 'DEPENDANTS', 'PERSONAL_INCOME', 'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED']]
df['LOAN_NUM_OPEN'] = df['LOAN_NUM_TOTAL'] - df['LOAN_NUM_CLOSED']
df.replace({'TARGET':{1: 'отклик был зарегистрирован', 0: 'отклика не было'},
            'GENDER': {1: 'мужчина', 0: 'женщина'},
            'SOCSTATUS_WORK_FL': {1: 'работает', 0: 'не работает'},
            'SOCSTATUS_PENS_FL': {1: 'пенсионер', 0: 'не пенсионер'}}, inplace=True)

st.set_page_config(page_title='bank')
st.title('Разведочный анализ данных')
st.markdown('В данном приложении реализован разведочный анализ данных для предсказания отклика клиентов на предложение банка.')
st.markdown('<p>Используемые признаки:</p><ul> \
            <li>AGREEMENT_RK — уникальный идентификатор объекта в выборке;</li> \
            <li>TARGET — целевая переменная: отклик на маркетинговую кампанию;</li> \
            <li>AGE — возраст клиента; \
            <li>SOCSTATUS_WORK_FL — социальный статус клиента относительно работы;</li> \
            <li>SOCSTATUS_PENS_FL — социальный статус клиента относительно пенсии;</li> \
            <li>GENDER — пол клиента;</li> \
            <li>CHILD_TOTAL — количество детей клиента;</li> \
            <li>DEPENDANTS — количество иждивенцев клиента;</li> \
            <li>PERSONAL_INCOME — личный доход клиента (в рублях);</li> \
            <li>LOAN_NUM_TOTAL — количество ссуд клиента;</li> \
            <li>LOAN_NUM_CLOSED — количество погашенных ссуд клиента;</li> \
            <li>LOAN_NUM_OPEN — количество открытых ссуд клиента.</li></ul>', unsafe_allow_html=True)
st.header('Описание числовых признаков')
st.markdown('Ниже представлена таблица с описанием числовых признаков.')
st.dataframe(df.describe())

st.header('Построение гистограммы')
st.markdown('Здесь можно построить гистограмму для различных признаков.')
row_name_1 = st.selectbox('Признаки:', ('AGE', 'SOCSTATUS_WORK_FL',
       'SOCSTATUS_PENS_FL', 'GENDER', 'CHILD_TOTAL', 'DEPENDANTS',
       'PERSONAL_INCOME', 'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED',
       'LOAN_NUM_OPEN'))
fig1 = px.histogram(df, x=row_name_1, color='TARGET')
st.plotly_chart(fig1, use_container_width=True)

st.header('Построение ящика с усами')
st.markdown('Здесь можно визуально оценить разницу в распределении числовых признаков при разном значении целевой переменной.')
row_name_2 = st.selectbox('Признаки:', ('AGE', 'CHILD_TOTAL', 'DEPENDANTS',
       'PERSONAL_INCOME', 'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED',
       'LOAN_NUM_OPEN'))
fig2 = px.box(df, x='TARGET', y=row_name_2)
st.plotly_chart(fig2, use_container_width=True)

st.header('Построение круговой гистограммы')
st.markdown('Для бинарных признаков.')
row_name_3 = st.selectbox('Название стобца:', ('SOCSTATUS_WORK_FL',
       'SOCSTATUS_PENS_FL', 'GENDER', 'TARGET'))
fig3 = px.pie(df, names=row_name_3)
st.plotly_chart(fig3, use_container_width=True)

st.header('Построение круговой гистограммы по выбранным значениям признаков')
st.markdown('Здесь можно, выбирая признаки для фильтрации, посмотреть процент откликнувшихся в различных категориях.')
gender = st.radio('Пол:', np.append(df['GENDER'].unique(),'оба варианта'))
work = st.radio('Статус работы:', np.append(df['SOCSTATUS_WORK_FL'].unique(), 'оба варианта'))
pens = st.radio('Статус пенсии:', np.append(df['SOCSTATUS_PENS_FL'].unique(), 'оба варианта'))
filtred = df.copy()
if gender != 'оба варианта':
    filtred = df[(df['GENDER'].isin([gender]))]
if work != 'оба варианта':
    filtred = df[df['SOCSTATUS_WORK_FL'].isin([work])]
if pens != 'оба варианта':
    filtred = df[df['SOCSTATUS_PENS_FL'].isin([pens])]
fig4 = px.pie(filtred, names='TARGET')
st.plotly_chart(fig4, use_container_width=True)