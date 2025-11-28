import pandas as pd
import glob
import os
from typing import Optional

def leer_csv_si_es_victoria(nombre_archivo: str) -> Optional[pd.DataFrame]:
    resultado_partida = ""
    
    try:
        with open(nombre_archivo, 'r') as f:
            lineas = f.readlines()
            if lineas:
                resultado_partida = lineas[-1].strip().lower()
                
    except Exception:
        return None

    if resultado_partida == 'win':
        try:
            df = pd.read_csv(nombre_archivo, skipfooter=1, engine='python')
            return df
        except pd.errors.ParserError:
            print(f"Error al parsear el CSV de victoria: {os.path.basename(nombre_archivo)}")
            return None
    elif resultado_partida == 'gameover':
        return None
    else:
        print(f"Resultado final inesperado en: {os.path.basename(nombre_archivo)} con '{resultado_partida}'")
        return None

#definir como funcion en un futuro y llamarla desde el script principal
def main():
    ruta_carpeta = 'raw_data_sets' 
    patron_archivos = 'TankTraining_*.csv' 
    
    if not os.path.isdir(ruta_carpeta):
        print(f"ERROR: La carpeta '{ruta_carpeta}' no existe o la ruta es incorrecta.")
        return

    ruta_busqueda = os.path.join(ruta_carpeta, patron_archivos)
    lista_archivos = glob.glob(ruta_busqueda)

    if not lista_archivos:
        print(f"ERROR: No se encontraron archivos con el patrón '{patron_archivos}' en '{ruta_carpeta}'.")
        return
    
    print(f"Archivos encontrados: {len(lista_archivos)}")

    lista_dataframes = []
    victorias_contadas = 0

    for nombre_archivo in lista_archivos:
        df_individual = leer_csv_si_es_victoria(nombre_archivo)
        
        if df_individual is not None:
            lista_dataframes.append(df_individual)
            victorias_contadas += 1

    print(f"Partidas ganadas incluidas: {victorias_contadas}")
            
    if not lista_dataframes:
        print("No se incluyó ningún archivo. Todas las partidas encontradas terminaron en 'gameover'.")
        return
        
    df_total_victorias = pd.concat(lista_dataframes, ignore_index=True)

    print("-" * 50)
    filas_totales = df_total_victorias.shape[0]
    
    print(f"Proceso completado. Filas totales del DataFrame final (Solo Victorias): {filas_totales}")
    
    nombre_salida = 'TankTraining_Victorias_Filtradas.csv'
    df_total_victorias.to_csv(nombre_salida, index=False)
    print(f"Dataset de Victorias guardado como '{nombre_salida}'.")

if __name__ == "__main__":
    main()