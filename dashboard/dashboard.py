import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Загрузка данных
current_dir = Path(__file__).resolve().parent
file_path = current_dir.parent / 'data' / 'data_clean.csv'  # Укажите правильный путь к вашему файлу
data = pd.read_csv(file_path, encoding='utf-8')

# Преобразование данных (если необходимо)
data['Цена(в $)'] = data['Цена(в $)'].str.replace(' ', '').astype(int)
data['Пробег'] = data['Пробег'].str.replace(' тыс.км', '').str.replace(' ', '').astype(float)
data['Год'] = pd.to_numeric(data['Год'], errors='coerce').astype('Int64')


# --- Главная страница ---
def home():
    st.title('Анализ объявлений о продаже автомобилей')
    st.write('Этот дашборд представляет анализ данных об объявлениях о продаже автомобилей. Данные были получены в результате парсинга сайта Auto.ria (см. `parser.ipynb`).')
    st.write('Дальнейший анализ и очистка данных производились в `analysis.ipynb` и `analysis.py`.')
    st.markdown("### Краткое описание колонок")
    st.markdown(
        """
        -   **Цена(в $)**: Цена автомобиля в долларах США.
        -   **Пробег**: Пробег автомобиля в тысячах километров.
        -   **Марка**: Марка автомобиля (например, Toyota, Honda).
        -   **Модель**: Модель автомобиля (например, Camry, Civic).
        -   **Год**: Год выпуска автомобиля.
        """
    )

# --- Раздел "Данные" ---
def data_section():
    st.header('Раздел "Данные"')

    # Интерактивная таблица
    st.subheader('Интерактивная таблица данных')
    st.dataframe(data, width=800, height=400)

    # Счётчики
    st.subheader('Счётчики')
    col1, col2, col3 = st.columns(3)
    col1.metric("Общее число записей", len(data))
    col2.metric("Количество пропусков", data.isnull().sum().sum())  # общее кол-во пропусков
    col3.metric("Уникальных марок", data['Марка'].nunique())

    # Распределение по основным категориям
    st.subheader('Распределение по маркам')
    brand_counts = data['Марка'].value_counts().reset_index()
    brand_counts.columns = ['Марка', 'Количество']

    chart = alt.Chart(brand_counts.head(20)).mark_bar().encode(
        x=alt.X('Марка', sort='-y'),
        y='Количество',
        tooltip=['Марка', 'Количество']
    ).properties(
        title='Топ-20 Марок Автомобилей'
    )
    st.altair_chart(chart, use_container_width=True)


# --- Раздел "EDA" ---
def eda_section():
    st.header('Раздел "EDA (Первичный анализ)"')

    # 1. Распределение цен
    st.subheader('Распределение цен')
    fig_price, ax_price = plt.subplots()
    sns.histplot(data['Цена(в $)'], bins=30, kde=True, color='skyblue', ax=ax_price)
    ax_price.set_title('Распределение цен на автомобили', fontsize=16)
    ax_price.set_xlabel('Цена ($)', fontsize=14)
    ax_price.set_ylabel('Количество автомобилей', fontsize=14)
    ax_price.axvline(data['Цена(в $)'].mean(), color='red', linestyle='--', label=f'Среднее: ${data["Цена(в $)"].mean():,.0f}')
    ax_price.axvline(data['Цена(в $)'].median(), color='green', linestyle='--', label=f'Медиана: ${data["Цена(в $)"].median():,.0f}')
    ax_price.legend()
    st.pyplot(fig_price)

    # 2. Распределение пробега
    st.subheader('Распределение Пробега')
    fig_mileage, ax_mileage = plt.subplots()
    sns.histplot(data['Пробег'], bins=30, kde=True, color='lightgreen', ax=ax_mileage)
    ax_mileage.set_title('Распределение Пробега автомобилей', fontsize=16)
    ax_mileage.set_xlabel('Пробег (тыс. км)', fontsize=14)
    ax_mileage.set_ylabel('Количество автомобилей', fontsize=14)
    st.pyplot(fig_mileage)

    # 3. Гистограмма распределения года выпуска
    st.subheader('Гистограмма распределения года выпуска')
    fig_year_hist, ax_year_hist = plt.subplots()
    sns.histplot(data['Год'].dropna(), bins=30, kde=False, color='orange', ax=ax_year_hist)  # Drop NaN values
    ax_year_hist.set_title('Распределение года выпуска', fontsize=16)
    ax_year_hist.set_xlabel('Год выпуска', fontsize=14)
    ax_year_hist.set_ylabel('Количество автомобилей', fontsize=14)
    st.pyplot(fig_year_hist)

    # 4. Ящик с усами для цен
    st.subheader('Boxplot для цен по топ-10 маркам')
    fig_boxplot, ax_boxplot = plt.subplots(figsize=(14, 8))
    top_10_brands = data['Марка'].value_counts().head(10).index
    sns.boxplot(data=data[data['Марка'].isin(top_10_brands)], x='Марка', y='Цена(в $)', palette='Set3', ax=ax_boxplot)
    ax_boxplot.set_title('Распределение цен по топ-10 маркам', fontsize=16)
    ax_boxplot.set_xlabel('Марка автомобиля', fontsize=14)
    ax_boxplot.set_ylabel('Цена ($)', fontsize=14)
    ax_boxplot.tick_params(axis='x', rotation=45)
    st.pyplot(fig_boxplot)


    # 5. Диаграмма рассеивания: цена vs пробег
    st.subheader('Диаграмма рассеивания: цена vs пробег')
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x='Пробег', y='Цена(в $)', ax=ax_scatter)
    ax_scatter.set_title('Зависимость цены от пробега', fontsize=16)
    ax_scatter.set_xlabel('Пробег (тыс. км)', fontsize=14)
    ax_scatter.set_ylabel('Цена ($)', fontsize=14)
    st.pyplot(fig_scatter)

    # 6. Корреляционный анализ
    st.subheader('Корреляционная матрица')
    correlation_matrix = data[['Цена(в $)', 'Пробег', 'Год']].corr()
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=.5, ax=ax_heatmap)
    ax_heatmap.set_title('Корреляционная матрица', fontsize=16)
    st.pyplot(fig_heatmap)

# --- Раздел "Тренды и закономерности" ---
def trends_section():
    st.header('Раздел "Тренды и закономерности"')

    # Фильтры
    st.subheader('Фильтры')
    min_year, max_year = int(data['Год'].min()), int(data['Год'].max())
    year_range = st.slider("Выберите диапазон годов выпуска", min_value=min_year, max_value=max_year, value=(min_year, max_year))
    selected_brands = st.multiselect("Выберите марки автомобилей", options=data['Марка'].unique(), default=data['Марка'].unique()[:5])

    # Применение фильтров
    filtered_data = data[
        (data['Год'].astype(int) >= year_range[0]) & (data['Год'].astype(int) <= year_range[1]) & data['Марка'].isin(selected_brands)
    ]

    # Вывод отфильтрованных данных
    st.write(f"Отображено {len(filtered_data)} записей.")
    st.dataframe(filtered_data.head(10))

    # Визуализации на основе отфильтрованных данных
    st.subheader('Визуализации')

    # 1. Зависимость цены от пробега (после фильтрации)
    st.subheader('Зависимость цены от пробега (после фильтрации)')
    fig_scatter_filtered, ax_scatter_filtered = plt.subplots()
    sns.scatterplot(data=filtered_data, x='Пробег', y='Цена(в $)', ax=ax_scatter_filtered)
    ax_scatter_filtered.set_title('Цена vs Пробег (Отфильтрованные данные)')
    ax_scatter_filtered.set_xlabel('Пробег (тыс. км)')
    ax_scatter_filtered.set_ylabel('Цена ($)')
    st.pyplot(fig_scatter_filtered)

    # 2. Распределение цен по маркам (после фильтрации) - Boxplot
    if selected_brands:
        st.subheader('Распределение цен по маркам (после фильтрации)')
        fig_box_filtered, ax_box_filtered = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=filtered_data, x='Марка', y='Цена(в $)', ax=ax_box_filtered)
        ax_box_filtered.set_title('Распределение цен по маркам (Отфильтрованные данные)')
        ax_box_filtered.set_xlabel('Марка')
        ax_box_filtered.set_ylabel('Цена ($)')
        ax_box_filtered.tick_params(axis='x', rotation=45)
        st.pyplot(fig_box_filtered)
    else:
        st.warning("Пожалуйста, выберите хотя бы одну марку автомобиля для отображения графика.")


# --- Раздел "Выводы и рекомендации" ---
def conclusions_section():
    st.header('Раздел "Выводы и рекомендации"')

    st.subheader('Ключевые инсайты')
    st.markdown(
        """
        -   Большинство автомобилей (около 75%) имеют цену до 50 000, что указывает на преобладание доступных моделей на рынке. Видны единичные дорогие экземпляры (выше 150 000), вероятно, это люксовые автомобили или редкие модели.
        -   Большинство автомобилей (около 80%) имеют пробег до 200 тыс. км. Автомобили с пробегом свыше 400 тыс. км встречаются редко, также как и новые авто с пробегом менее 10 тыс. км.
        -   Пик предложений приходится на 2015–2020 годы — период, когда автомобили сохраняют современные технологии, но уже успели опуститься в цене. Мало предложений до 2010 года. Автомобили 2023–2025 годов встречаются редко.
        -   На рынке доминируют автомобили возрастом 5–10 лет, что оптимально для соотношения цены и качества.
        -   Дорогие марки - Porsche, Mercedes-Benz, BMW, Audi — медианная цена выше $40 000.
        -   Бюджетные марки - Volkswagen, Renault, Ford — медианная цена ниже $20 000.
        -   Высокий разброс цен у Land Rover и Volvo. Также у всех марок есть аномально дорогие предложения.
        -   Чем выше пробег, тем ниже цена.
        -   Чем новее автомобиль, тем выше цена.
        -   Эти 2 фактора являются ключевыми для ценообразования.
        -   Цена и год выпуска: +0.53 (умеренная связь).
        -   Цена и пробег: -0.49 (умеренная обратная связь).
        -   Пробег и год выпуска: -0.63 (сильная обратная связь: новые авто обычно имеют меньший пробег).
        -   Самыми дорогими марками автомобилей являются Cadillac, Maserati, Porsche, Land Rover, Lexus. Автомобили марки Van Hool выделяются по цене из-за направленности марки на автобусный и грузовой транспорт, а также из-за редкости моделей.
        -   **Вывод:** Рынок ориентирован на бюджетные и среднеценовые автомобили, но присутствует также присутствуют и премиальные модели.
        -   **Вывод:** Покупатели предпочитают автомобили с умеренным пробегом, что снижает риски износа.
        -   **Вывод:** Немецкие премиум-бренды лидируют по стоимости, а массовые марки обеспечивают доступный сегмент.
        -   **Вывод:** Для предсказания цены важнее всего год, затем — пробег.
        -   **Вывод:** Графики и статистика подтверждают, что рынок подержанных автомобилей структурирован вокруг логичных зависимостей: цена падает с пробегом и растет с новизной, а бренд определяет позиционирование.
        """
    )

    st.subheader('Рекомендации и дальнейшие шаги')
    st.markdown(
        """
        -   **Улучшение данных**: Рассмотреть возможность добавления информации о комплектации автомобилей (например, тип двигателя, трансмиссия) для более глубокого анализа.
        -   **Географический анализ**: Провести анализ цен и спроса на автомобили в разных регионах.
        -   **Расширение данных**: Увеличения объема данных для анализа(не 30 страниц как сейчас, а например 100)
        """
    )

# --- Навигация ---
menu = {
    "Главная": home,
    "Данные": data_section,
    "EDA": eda_section,
    "Тренды и закономерности": trends_section,
    "Выводы и рекомендации": conclusions_section
}

selected_page = st.sidebar.selectbox("Выберите раздел", options=list(menu.keys()))

if selected_page:
    menu[selected_page]()
