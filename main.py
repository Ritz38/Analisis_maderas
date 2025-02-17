import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import scipy.stats as stats
import re
import streamlit as st

def cargar_datos():
    """Permite al usuario subir un archivo CSV o ingresar una URL."""
    url_predeterminado = 'https://raw.githubusercontent.com/Ritz38/Analisis_maderas/refs/heads/main/Base_de_datos_relacionada_con_madera_movilizada_proveniente_de_Plantaciones_Forestales_Comerciales_20250217.csv'
    
    url = st.text_input("Ingrese la URL del archivo CSV:", url_predeterminado)
    archivo = st.file_uploader("O suba un archivo CSV", type=["csv"])
    
    return pd.read_csv(archivo) if archivo else pd.read_csv(url) if url else None

def maderas_comunes(df):
    """Identifica las especies de madera más comunes."""
    agrupados_madera_concurrencias = df.groupby('ESPECIE').size().sort_values(ascending=False).head()
    volumenes = df.groupby('ESPECIE')['VOLUMEN M3'].sum()
    st.write('Las 5 maderas mas comunes  y su respectivo volumen son:\n', volumenes.loc[agrupados_madera_concurrencias.index])

    eleccion = st.selectbox('Elija el departamento del cual quiere saber las maderas mas comunes', list(df['DPTO'].unique()))
    agrupados_madera_concurrencias_dpto = df[df['DPTO']==eleccion].groupby('ESPECIE').size().sort_values(ascending=False).head()
    volumenes_dpto = df[df['DPTO']==eleccion].groupby('ESPECIE')['VOLUMEN M3'].sum()
    st.write('Las 5 maderas mas comunes  y su respectivo volumen son:\n', volumenes_dpto.loc[agrupados_madera_concurrencias_dpto.index])
    

def grafico_maderas(df):
    """Genera un gráfico de barras de las 10 especies con mayor volumen movilizado."""
    especies = df.groupby('ESPECIE')['VOLUMEN M3'].sum().nlargest(10)
    fig, ax = plt.subplots()
    especies.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title("Top 10 especies con mayor volumen movilizado")
    ax.set_ylabel("Volumen")
    st.pyplot(fig)

def mapa_calor(df):
    """Genera un mapa de calor de volúmenes de madera por departamento."""
    fig, ax = plt.subplots()
    
    vol_por_dpto = df.groupby('DPTO')['VOLUMEN M3'].sum().reset_index()
    df_geo = colombia.merge(vol_por_dpto, left_on='NOMBRE_DPT', right_on='DPTO')
    
    df_geo.plot(column='VOLUMEN M3', cmap='OrRd', linewidth=0.8, edgecolor='k', legend=True, ax=ax)
    
    # Establecer el título
    ax.set_title("Distribución de volúmenes de madera por departamento")
    ax.set_title("Distribución de volúmenes de madera por departamento")
    st.pyplot(fig)

def municipios_mayor_movilidad(df):
    """Muestra los 10 municipios con mayor movilización de madera en un mapa."""
    
    municipios = gpd.read_file('https://raw.githubusercontent.com/Ritz38/Analisis_maderas/refs/heads/main/puntos_municipios.csv')
    
    municipios['geometry'] = gpd.GeoSeries.from_wkt(municipios['Geo Municipio'])
    municipios = municipios.set_geometry('geometry')
    
    municipios_movidos = df.groupby('MUNICIPIO').size().sort_values(ascending=False).head(10)

    # Filtrar los municipios que están en los 10 más movidos
    municipios = municipios[municipios['NOM_MPIO'].isin(municipios_movidos.index)]
    
    fig, ax = plt.subplots()
    colombia.plot(ax=ax, color='white', edgecolor='black')
    municipios.plot(ax=ax, color='red')
    st.pyplot(fig)

def evolucion_temporal(df):
    """Grafica la evolución del volumen de madera movilizada por especie a lo largo del tiempo."""
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    evolucion = df.groupby([df['FECHA'].dt.year, 'ESPECIE'])['VOLUMEN'].sum().unstack()
    
    fig, ax = plt.subplots()
    evolucion.plot(ax=ax)
    ax.set_title("Evolución temporal del volumen movilizado por especie")
    ax.set_ylabel("Volumen")
    st.pyplot(fig)

def detectar_outliers(df):
    """Identifica valores atípicos en el volumen de madera movilizada."""
    z_scores = stats.zscore(df['VOLUMEN'].dropna())
    return df.loc[np.abs(z_scores) > 3]

def diversidad_shannon(df):
    """Calcula el índice de diversidad de Shannon para evaluar la diversidad de especies por departamento."""
    def shannon_entropy(x):
        p = x / x.sum()
        return -np.sum(p * np.log2(p))
    
    return df.groupby('DEPARTAMENTO')['ESPECIE'].value_counts().unstack(fill_value=0).apply(shannon_entropy, axis=1)

def clustering_departamentos(df):
    """Agrupa departamentos con patrones similares de movilización de madera."""
    from scipy.cluster.hierarchy import linkage, fcluster
    
    vol_por_dep = df.groupby('DEPARTAMENTO')['VOLUMEN'].sum().values.reshape(-1, 1)
    clusters = fcluster(linkage(vol_por_dep, method='ward'), 3, criterion='maxclust')
    
    return pd.DataFrame({'DEPARTAMENTO': df['DEPARTAMENTO'].unique(), 'Cluster': clusters})

def main():
    """Función principal de la aplicación en Streamlit."""
    st.title("Análisis de Plantaciones Forestales Comerciales en Colombia")
    
    df = cargar_datos()
    df['DPTO'] = df['DPTO'].str.upper()
    df['MUNICIPIO'] = df['MUNICIPIO'].str.upper()
    if df is not None:
        st.write("Vista previa de los datos:")
        st.write(df.head())
        global colombia
        colombia = gpd.read_file('https://gist.githubusercontent.com/john-guerra/43c7656821069d00dcbc/raw/be6a6e239cd5b5b803c6e7c2ec405b793a9064dd/Colombia.geo.json')
        
        maderas_comunes(df)
        
        st.subheader("Gráfico de especies con mayor volumen")
        grafico_maderas(df)
        
        st.subheader("Mapa de calor por departamento")
        mapa_calor(df)
        
        st.subheader("Municipios con mayor movilización")
        municipios_mayor_movilidad(df)
        
        st.subheader("Evolución temporal del volumen movilizado")
        evolucion_temporal(df)
        
        st.subheader("Detección de valores atípicos")
        st.write(detectar_outliers(df))
        
        st.subheader("Índice de diversidad de Shannon por departamento")
        st.write(diversidad_shannon(df))
        
        st.subheader("Clustering de departamentos")
        st.write(clustering_departamentos(df))

if __name__ == "__main__":
    main()
