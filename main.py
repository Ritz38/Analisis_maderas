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

def agregar_nombre_municipio(row, ax):
    """Agrega el nombre del municipio en el gráfico en las coordenadas del punto."""
    ax.text(row.geometry.x, row.geometry.y, row['NOM_MPIO'], fontsize=8, ha='right', color='black')

def municipios_mayor_movilidad(df):
    """Muestra los 10 municipios con mayor movilización de madera en un mapa."""
    
    municipios_movidos = df.groupby('MUNICIPIO').size().sort_values(ascending=False).head(10)

    # Filtrar los municipios que están en los 10 más movidos
    municipios = municipios[municipios['NOM_MPIO'].isin(municipios_movidos.index)]
    
    fig, ax = plt.subplots()
    colombia.plot(ax=ax, color='white', edgecolor='black')
    municipios.plot(ax=ax, color='red')
    municipios.apply(agregar_nombre_municipio, ax=ax, axis=1)
    st.pyplot(fig)

def evolucion_temporal_especie_producto(df):
    """Grafica la evolución del volumen de madera movilizada por especie y tipo de producto a lo largo del tiempo."""
    # Seleccionar la especie
    especies = df['ESPECIE'].unique()
    especie_seleccionada = st.selectbox('Seleccione la especie de madera:', especies)
    
    # Filtrar los tipos de producto disponibles para la especie seleccionada
    tipos_producto_disponibles = df[df['ESPECIE'] == especie_seleccionada]['TIPO PRODUCTO'].unique()
    
    # Seleccionar el tipo de producto (solo mostrar los disponibles para la especie seleccionada)
    tipo_producto_seleccionado = st.selectbox('Seleccione el tipo de producto:', tipos_producto_disponibles)
    
    # Filtrar el DataFrame por la especie y el tipo de producto seleccionados
    df_filtrado = df[(df['ESPECIE'] == especie_seleccionada) & (df['TIPO PRODUCTO'] == tipo_producto_seleccionado)]

    granularidad = st.selectbox("Seleccione la granularidad temporal:", ["AÑO", "SEMESTRE", "TRIMESTRE"])
    
    # Verificar si el DataFrame filtrado está vacío
    if df_filtrado.empty:
        st.warning(f"No hay datos disponibles para la especie '{especie_seleccionada}' y el tipo de producto '{tipo_producto_seleccionado}'.")
    else:
        # Crear una columna de fecha basada en la selección de granularidad
        if granularidad == "AÑO":
            df_filtrado["FECHA"] = pd.to_datetime(df_filtrado["AÑO"].astype(str), format="%Y")
        
        elif granularidad == "SEMESTRE":
            # Convertir "I" -> 1 y "II" -> 2
            df_filtrado["SEMESTRE_NUM"] = df_filtrado["SEMESTRE"].map({"I": 1, "II": 2})
            df_filtrado["FECHA"] = pd.to_datetime(df_filtrado["AÑO"].astype(str) + "-" + 
                                                  (df_filtrado["SEMESTRE_NUM"] * 6 - 5).astype(str), format="%Y-%m")
        
        elif granularidad == "TRIMESTRE":
            # Convertir "I", "II", "III", "IV" a 1, 2, 3, 4
            df_filtrado["TRIMESTRE_NUM"] = df_filtrado["TRIMESTRE"].map({"I": 1, "II": 2, "III": 3, "IV": 4})
            df_filtrado["FECHA"] = pd.to_datetime(df_filtrado["AÑO"].astype(str) + "-" + 
                                                  (df_filtrado["TRIMESTRE_NUM"] * 3 - 2).astype(str), format="%Y-%m")
        # Agrupar por año y calcular el volumen total
        evolucion = df_filtrado.groupby('FECHA')['VOLUMEN M3'].sum()
        
        # Graficar
        fig, ax = plt.subplots()
        evolucion.plot(kind='line', ax=ax, marker='o')
        ax.set_title(f"Evolución temporal del volumen movilizado de {especie_seleccionada} ({tipo_producto_seleccionado})")
        ax.set_xlabel("Año")
        ax.set_ylabel("Volumen (M3)")
        st.pyplot(fig)

def detectar_outliers(df):
    """Identifica valores atípicos en el volumen de madera movilizada."""
    z_scores = stats.zscore(df['VOLUMEN M3'])
    return df.loc[np.abs(z_scores) > 3].loc[:,['MUNICIPIO', 'ESPECIE', 'VOLUMEN M3']]

def volumen_por_municipio(df):
    """Calcula el volumen total de madera movilizada por municipio."""
    return df.groupby('MUNICIPIO')['VOLUMEN M3'].sum().reset_index()

def diversidad_shannon(df):
    """Calcula el índice de diversidad de Shannon para evaluar la diversidad de especies por departamento."""
    def shannon_entropy(x):
        p = x / x.sum()
        return -np.sum(p * np.log2(p))
    
    return df.groupby('DPTO')['ESPECIE'].value_counts().unstack(fill_value=0).apply(shannon_entropy, axis=1)

def clustering_departamentos(df):
    """Agrupa departamentos con patrones similares de movilización de madera."""
    from scipy.cluster.hierarchy import linkage, fcluster
    
    vol_por_dep = df.groupby('DPTO')['VOLUMEN M3'].sum().values.reshape(-1, 1)
    clusters = fcluster(linkage(vol_por_dep, method='ward'), 3, criterion='maxclust')
    
    df_clusters = pd.DataFrame({'DPTO': df['DPTO'].unique(), 'Cluster': clusters})
    
    fig, ax = plt.subplots()
    colombia.merge(df_clusters, left_on='NOMBRE_DPT', right_on='DPTO').plot(column='Cluster', cmap='Set3', legend=True, ax=ax)
    ax.set_title("Clustering de departamentos por volumen de madera movilizada")
    st.pyplot(fig)

def especies_menor_volumen(df):
    """Identifica las especies con menor volumen movilizado y su distribución geográfica."""
    especies_menor_volumen = df.groupby('ESPECIE')['VOLUMEN M3'].sum().nsmallest(10)
    df_menor_volumen = df[df['ESPECIE'].isin(especies_menor_volumen.index)]
    
    fig, ax = plt.subplots()
    colombia.plot(ax=ax, color='white', edgecolor='black')
    df_menor_volumen = municipios[municipios['NOM_MPIO'].isin(df_menor_volumen['ESPECIE'].unique())]
    df_menor_volumen.plot(ax=ax, color='red', marker='o', markersize=5)
    ax.set_title("Distribución geográfica de especies con menor volumen movilizado")
    st.pyplot(fig)

def distribucion_especies_entre_departamentos(df):
    """Compara la distribución de especies entre departamentos."""
    distribucion = df.groupby(['DPTO', 'ESPECIE'])['VOLUMEN M3'].sum().unstack()
    
    fig, ax = plt.subplots()
    distribucion.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title("Distribución de especies entre departamentos")
    ax.set_ylabel("Volumen")
    st.pyplot(fig)

def comparar_diversidad_regiones(df):
    """Compara la diversidad de especies entre regiones."""
    diversidad = diversidad_shannon(df)
    diversidad.plot(kind='bar', color='skyblue')
    plt.title("Comparación de diversidad de especies entre regiones")
    plt.ylabel("Índice de Shannon")
    st.pyplot()

def main():
    """Función principal de la aplicación en Streamlit."""
    st.title("Análisis de Plantaciones Forestales Comerciales en Colombia")
    
    df = cargar_datos()
    df['DPTO'] = df['DPTO'].str.upper()
    df['MUNICIPIO'] = df['MUNICIPIO'].str.upper()
    if df is not None:
        st.write("Vista previa de los datos:")
        st.write(df.head())
        global colombia, municipios
        colombia = gpd.read_file('https://gist.githubusercontent.com/john-guerra/43c7656821069d00dcbc/raw/be6a6e239cd5b5b803c6e7c2ec405b793a9064dd/Colombia.geo.json')
        municipios = gpd.read_file('https://raw.githubusercontent.com/Ritz38/Analisis_maderas/refs/heads/main/puntos_municipios.csv')
        municipios['geometry'] = gpd.GeoSeries.from_wkt(municipios['Geo Municipio'])
        municipios = municipios.set_geometry('geometry')
        
        maderas_comunes(df)
        
        st.subheader("Gráfico de especies con mayor volumen")
        grafico_maderas(df)
        
        st.subheader("Mapa de calor por departamento")
        mapa_calor(df)
        
        st.subheader("Municipios con mayor movilización")
        municipios_mayor_movilidad(df)
        
        st.subheader("Evolución temporal por especie y tipo de producto")
        evolucion_temporal_especie_producto(df)
        
        st.subheader("Detección de valores atípicos")
        st.write(detectar_outliers(df))
        
        st.subheader("Índice de diversidad de Shannon por departamento")
        st.write(diversidad_shannon(df))
        
        st.subheader("Clustering de departamentos")
        st.write(clustering_departamentos(df))
        
        st.subheader("Volumen por municipio")
        st.write(volumen_por_municipio(df))
        
        st.subheader("Especies con menor volumen movilizado")
        especies_menor_volumen(df)
        
        st.subheader("Distribución de especies entre departamentos")
        distribucion_especies_entre_departamentos(df)
        
        st.subheader("Comparación de diversidad entre regiones")
        comparar_diversidad_regiones(df)

if __name__ == "__main__":
    main()
