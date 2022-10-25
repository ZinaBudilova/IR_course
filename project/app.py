import streamlit as st
from datetime import datetime
from funcs import search, matrices, titles, tools


st.title('Поисковик по ОтВеТаМ мЭйЛ рУ')

query = st.text_input('Введите запрос')

method = st.selectbox(
    'Выберите метод поиска',
    ('TF-IDF', 'BM25', 'BERT'))

placeholder = st.empty()

if st.button('Поиск'):
    placeholder = st.empty()
    with placeholder.container():
        st.write("**Ваш запрос**:", query)
        st.write("**Выбранный метод поиска:**", method)
        start_time = datetime.now()
        res = search(query, method, matrices, titles, tools)
        st.write("**Время поиска** составило", datetime.now() - start_time)
        st.write('**Результаты:**')
        for item in res:
            st.write(item[0])
