from tkinter import filedialog
import numpy as np
import pandas as pd
import cv2 ## Tratamiento de imagenes

import tkinter as tk ## GUI
import pdf2image
import os
import google.generativeai as ai

from datetime import datetime
from dotenv import load_dotenv
from PIL import Image ## GUI

# Cargar las variables de entorno desde el archivo .env
load_dotenv()  

## Creamos la ventana con la opción de seleccionar el archivo
root = tk.Tk()
root.geometry("1000x1000")
root.title("Modificar stock de merchandising")

global Camiseta_basica_S, Camiseta_basica_M, Camiseta_basica_L, Camiseta_basica_XL, Camiseta_basica_XXL # B,C,D,E,F 10
global Camiseta_panther_S, Camiseta_panther_M, Camiseta_panther_L, Camiseta_panther_XL, Camiseta_panther_XXL # B,C,D,E,F 15
global Chaqueta_S, Chaqueta_M, Chaqueta_L, Chaqueta_XL, Chaqueta_XXL # B,C,D,E,F 5
global Botella # C20

Camiseta_basica_S = Camiseta_basica_M = Camiseta_basica_L = Camiseta_basica_XL = Camiseta_basica_XXL = 0
Camiseta_panther_S = Camiseta_panther_M = Camiseta_panther_L = Camiseta_panther_XL = Camiseta_panther_XXL = 0
Chaqueta_S = Chaqueta_M = Chaqueta_L = Chaqueta_XL = Chaqueta_XXL = 0
Botella = 0

def seleccionar_archivo():
    global ruta_archivo
    
    ruta_archivo = ""

    date = datetime.today().strftime("%d-%m-%Y")
    # ruta_archivo = "./assets/Escaneo_test_impresora.jpg"
    ruta_archivo = filedialog.askopenfilename(
            title="Seleccionar archivo pdf",
        )
    if ruta_archivo:
        image = pdf2image.convert_from_path(ruta_archivo, dpi=300)
        for image in image:
            # Convertir la imagen a un formato compatible con PIL
            image = image.convert("RGB")
            # Guardar la imagen en un archivo temporal
            ruta_archivo = f"{date}stock.jpg"
            image.save(ruta_archivo, "JPEG")
    pre_procesar_imagen(ruta_archivo)

    return ruta_archivo

def parsear_respuesta_gemini(respuesta, i):
    """
    Parsea la respuesta de Gemini y actualiza el producto.

    Args:
        respuesta: Respuesta de Gemini en formato de texto.

    Returns:
        Una tupla con los datos extraídos o None si la respuesta es 'NO APLICA'.
    """

    global Camiseta_basica_S, Camiseta_basica_M, Camiseta_basica_L, Camiseta_basica_XL, Camiseta_basica_XXL
    global Camiseta_panther_S, Camiseta_panther_M, Camiseta_panther_L, Camiseta_panther_XL, Camiseta_panther_XXL
    global Chaqueta_S, Chaqueta_M, Chaqueta_L, Chaqueta_XL, Chaqueta_XXL
    global Botella

    # Si la respuesta contiene 'NO APLICA', devolver None
    if 'NO APLICA' in respuesta:
        print(f"Fila {i}: NO APLICA\n")
        return None
    
    # Buscar patrón de tupla en la respuesta
    import re
    
    # Buscar patrones de tupla con paréntesis y elementos separados por punto y coma o coma
    patron_tupla = r"\((.*?)\)"
    coincidencia = re.search(patron_tupla, respuesta)
    
    if not coincidencia:
        print(f"Fila {i}: No se encontró la coincidencia\n")
        return None
    
    # Extraer el contenido de la tupla
    contenido_tupla = coincidencia.group(1)
    
    # Verificar si los elementos están separados por punto y coma o por coma
    if ';' in contenido_tupla:
        elementos = contenido_tupla.split(';')
    else:
        # Si no hay punto y coma
        print(f"Fila {i}: No se encontró el separador ';', el contenido proporcionado es: {contenido_tupla}\n")
        return None
    
    # Limpiar cada elemento (quitar comillas y espacios)
    elementos_limpios = []
    for elemento in elementos:
        # Quitar comillas simples o dobles y espacios en blanco
        elemento = elemento.strip()
        if elemento.startswith(("'", '"')) and elemento.endswith(("'", '"')):
            elemento = elemento[1:-1]
        
        # Intentar convertir números a enteros
        try:
            if elemento.isdigit():
                elemento = int(elemento)
        except (ValueError, AttributeError):
            pass
        
        elementos_limpios.append(elemento)
    
    # Si hay 3 o 4 elementos (dependiendo de si es ropa o no), devolver la tupla
    if 3 <= len(elementos_limpios) <= 4:
        print(f"Fila {i} correcta, la información es: {elementos_limpios}")
        # Actualizar los datos según el producto
        if len(elementos_limpios) == 4:
            try:
                cantidad = int(elementos_limpios[2])
            except ValueError:
                print(f"Fila {i}: Error al convertir la cantidad a entero. La cantidad es: {elementos_limpios[2]}\n")
                return None
            print("cantidad: ", elementos_limpios[2])
            if elementos_limpios[0] == "Camiseta básica":
                if elementos_limpios[1] == "S":
                    Camiseta_basica_S += cantidad
                elif elementos_limpios[1] == "M":
                    Camiseta_basica_M += cantidad
                elif elementos_limpios[1] == "L":
                    Camiseta_basica_L += cantidad
                elif elementos_limpios[1] == "XL":
                    Camiseta_basica_XL += cantidad
                elif elementos_limpios[1] == "XXL":
                    Camiseta_basica_XXL += cantidad
            elif elementos_limpios[0] == "Camiseta panther":
                if elementos_limpios[1] == "S":
                    Camiseta_panther_S += cantidad
                elif elementos_limpios[1] == "M":
                    Camiseta_panther_M += cantidad
                elif elementos_limpios[1] == "L":
                    Camiseta_panther_L += cantidad
                elif elementos_limpios[1] == "XL":
                    Camiseta_panther_XL += cantidad
                elif elementos_limpios[1] == "XXL":
                    Camiseta_panther_XXL += cantidad
            elif elementos_limpios[0] == "Chaqueta":
                if elementos_limpios[1] == "S":
                    Chaqueta_S += cantidad
                elif elementos_limpios[1] == "M":
                    Chaqueta_M += cantidad
                elif elementos_limpios[1] == "L":
                    Chaqueta_L += cantidad
                elif elementos_limpios[1] == "XL":
                    Chaqueta_XL += cantidad
                elif elementos_limpios[1] == "XXL":
                    Chaqueta_XXL += cantidad
            elif elementos_limpios[0] == "Botella":
                Botella += cantidad
        elif len(elementos_limpios) == 3:
            try:
                cantidad = int(elementos_limpios[1])
            except ValueError:
                print(f"Fila {i}: Error al convertir la cantidad a entero. La cantidad es: {elementos_limpios[2]}\n")
                return None
            if elementos_limpios[0] == "Botella":
                Botella += cantidad
            else:
                print(f"Esta fila tiene un producto nuevo y hay que añadirlo a la base de datos, el producto es: {elementos_limpios[0]}\n")

        return tuple(elementos_limpios)
    
    return None

def leer_imagen_completa_ai(imagen, i):
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("Error: API_KEY no encontrada en el archivo .env")
        return
    
    ai.configure(api_key=api_key)
    
    # Convertir la imagen al formato adecuado según su tipo
    if isinstance(imagen, np.ndarray):
        # Asegurar que los valores estén en el rango correcto (0-255)
        if imagen.max() == 1:
            imagen_procesada = (imagen * 255).astype(np.uint8)
        else:
            imagen_procesada = imagen.astype(np.uint8)
        
        # Convertir a PIL Image
        pil_imagen = Image.fromarray(imagen_procesada)
        
        # Si la imagen es de un solo canal (escala de grises), convertirla a RGB
        if pil_imagen.mode != 'RGB':
            pil_imagen = pil_imagen.convert('RGB')
        
        imagen_para_prompt = pil_imagen
    elif isinstance(imagen, str):
        # Si es una ruta de archivo, cargar directamente
        imagen_para_prompt = Image.open(imagen)
    else:
        print("Error: Tipo de imagen no soportado. Debe ser un array NumPy o una ruta de archivo.")
        return
    
    prompt = "Eres un trabajador de una empresa que se dedica a actualizar el stock de productos. Para ello, debes analizar filas de una tabla, " \
    "cada fila tiene 4 columnas: producto, talla, cantidad y fecha. La primera columna (producto) tiene los posibles productos en formato checkbox y el producto" \
    "seleccionado está marcado (si se ha seleccionado 'Otro', el producto es el que está escrito justo después) la segunda columna (talla)" \
    " también tiene las tallas en formato checkbox y la talla seleccionada está marcada, la tercera columna (cantidad) contiene un número y" \
    " la cuarta columna (fecha) contiene una fecha en formato DD/MM/YYYY. Para la fila que has recibido, debes devolver sólamente una tupla" \
    " con el formato (Producto; Talla; Cantidad; Fecha), omite la talla si el producto no es una prenda. Además, si no hay información en la" \
    " celda de cantidad devuelve el texto 'NO APLICA' en vez de la tupla que se te pide.", imagen_para_prompt

    model = ai.GenerativeModel(model_name= "gemini-2.0-flash")

    chat = model.start_chat()
    response = chat.send_message(prompt)

    parsear_respuesta_gemini(response.text, i)

# Pre-procesar la imagen: convertir a escala de grises, desenfocar, binarizar y detectar líneas
# Esta función toma la ruta de la imagen y devuelve la imagen binarizada
def pre_procesar_imagen(ruta):

    global imagen_grayscale, imagen_desenfocada, imagen_binarizada

    # Leer la imagen original
    imagen = cv2.imread(ruta)
    if imagen is None:
        print("Error al cargar la imagen.")
        return
    
    # Convertir la imagen a escala de grises
    imagen_grayscale = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar un desenfoque gaussiano para reducir el ruido
    imagen_desenfocada = cv2.GaussianBlur(imagen_grayscale, (5, 5), 0)
    
    # Aplicar la binarización adaptativa
    imagen_binarizada = cv2.adaptiveThreshold(imagen_desenfocada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    detectar_lineas_horizontales(imagen_binarizada)

    return imagen_binarizada

# Detectar líneas horizontales en la imagen binarizada
# Devuelve una lista de líneas horizontales con formato (x1, y, x2)
def detectar_lineas_horizontales(imagen_binarizada, umbral_y_cercania=10, longitud_minima=2000):
    global lineas_horizontales
    
    """
    Detecta líneas horizontales en una imagen binarizada usando la Transformada de Hough Probabilística.

    Args:
        imagen_binarizada: Imagen binarizada (blanco sobre negro).
        umbral_y_cercania: Si 2 lineas estan a menos de este umbral se unen
        longitud_minima: Longitud minima para que una línea se devuelva.

    Returns:
        Una lista de líneas horizontales detectadas, donde cada línea es una tupla (y1, x1, x2).
    """

    lineas = cv2.HoughLinesP(imagen_binarizada, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=20)

    altura = imagen_binarizada.shape[0]
    margen = int(altura * 0.1)

    if lineas is None:
        return []

    lineas_horizontales = []
    for linea in lineas:
        x1, y1, x2, y2 = linea[0]
        if abs(y1 - y2) < umbral_y_cercania:
            # Guardamos las líneas como (y, x_inicial, x_final) para facilitar la unión
            if y1 > margen and y1 < (altura - margen) and \
               y2 > margen and y2 < (altura - margen):
                lineas_horizontales.append((y1, min(x1, x2), max(x1, x2)))

    lineas_horizontales.sort(key=lambda x: x[0])  # Ordenar por 'y'

    lineas_unidas = []
    if lineas_horizontales:
        linea_actual_y = lineas_horizontales[0][0]
        x_min_actual = lineas_horizontales[0][1]
        x_max_actual = lineas_horizontales[0][2]

        for i in range(1, len(lineas_horizontales)):
            y, x_min, x_max = lineas_horizontales[i]
            if abs(y - linea_actual_y) < umbral_y_cercania:
                x_min_actual = min(x_min_actual, x_min)
                x_max_actual = max(x_max_actual, x_max)
            else:
                if (x_max_actual - x_min_actual) >= longitud_minima:
                    lineas_unidas.append((x_min_actual, linea_actual_y, x_max_actual))
                linea_actual_y = y
                x_min_actual = x_min
                x_max_actual = x_max

        # Añadir la última línea unida (si cumple la longitud mínima)
        if (x_max_actual - x_min_actual) >= longitud_minima:
            lineas_unidas.append((x_min_actual, linea_actual_y, x_max_actual))

    # Visualizar las líneas unidas y filtradas
    imagen_con_lineas = cv2.cvtColor(imagen_binarizada.copy(), cv2.COLOR_GRAY2BGR)
    for x_inicial, y, x_final in lineas_unidas:
        cv2.line(imagen_con_lineas, (x_inicial, y), (x_final, y), (0, 0, 255), 2, cv2.LINE_AA)

    segmentar_imagenes(imagen_binarizada, lineas_unidas)   

    return lineas_unidas

def segmentar_imagenes(imagen_binarizada, lineas_horizontales):

    """
    Segmenta la imagen en partes utilizando las líneas horizontales detectadas.

    Args:
        imagen_binarizada: Imagen binarizada
        lineas_horizontales: Lista de líneas horizontales detectadas.
    
    Returns:
        Una lista de imágenes segmentadas.
    """

    filas_segmentadas = []
    if not lineas_horizontales:
        return filas_segmentadas

    # Extraer las coordenadas 'y' únicas y ordenarlas
    coordenadas_y = sorted(list(set([int(y) for _, y, _ in lineas_horizontales])))

    # Asegurarse de tener al menos dos coordenadas 'y' para delimitar filas
    if len(coordenadas_y) < 2:
        return filas_segmentadas

    # La primera fila estará desde el borde superior de la imagen hasta la primera línea
    y_superior_anterior = coordenadas_y[0]
    for y_inferior in coordenadas_y[1:]:  # Comenzar desde el segundo valor
        # Asegurarse de que la diferencia vertical sea positiva
        if y_inferior > y_superior_anterior:
            fila = imagen_binarizada[y_superior_anterior:y_inferior, :]
            filas_segmentadas.append(fila)
            y_superior_anterior = y_inferior

    num_fila = 1
    for fila in filas_segmentadas:
        leer_imagen_completa_ai(fila, num_fila)
        num_fila += 1
    
    sincronizar_inputs()

    return filas_segmentadas


# Botón para seleccionar archivo
boton_seleccionar_archivo = tk.Button(root, text="Seleccionar archivo", command=seleccionar_archivo)
boton_seleccionar_archivo.pack(pady=20)

# Inputs para los productos
label_camiseta_basica_S = tk.Label(root, text="Camisetas talla básica S")
label_camiseta_basica_S.pack()
entry_camiseta_basica_S = tk.Entry(root)
entry_camiseta_basica_S.insert(0, str(Camiseta_basica_S))
entry_camiseta_basica_S.pack()
label_camiseta_basica_M = tk.Label(root, text="Camisetas talla básica M")
label_camiseta_basica_M.pack()
entry_camiseta_basica_M = tk.Entry(root)
entry_camiseta_basica_M.insert(0, str(Camiseta_basica_M))
entry_camiseta_basica_M.pack()
label_camiseta_basica_L = tk.Label(root, text="Camisetas talla básica L")
label_camiseta_basica_L.pack()
entry_camiseta_basica_L = tk.Entry(root)
entry_camiseta_basica_L.insert(0, str(Camiseta_basica_L))
entry_camiseta_basica_L.pack()
label_camiseta_basica_XL = tk.Label(root, text="Camisetas talla básica XL")
label_camiseta_basica_XL.pack()
entry_camiseta_basica_XL = tk.Entry(root)
entry_camiseta_basica_XL.insert(0, str(Camiseta_basica_XL))
entry_camiseta_basica_XL.pack()
label_camiseta_basica_XXL = tk.Label(root, text="Camisetas talla básica XXL")
label_camiseta_basica_XXL.pack()
entry_camiseta_basica_XXL = tk.Entry(root)
entry_camiseta_basica_XXL.insert(0, str(Camiseta_basica_XXL))
entry_camiseta_basica_XXL.pack()

label_camiseta_panther_S = tk.Label(root, text="Camisetas talla panther S")
label_camiseta_panther_S.pack()
entry_camiseta_panther_S = tk.Entry(root)
entry_camiseta_panther_S.insert(0, str(Camiseta_panther_S))
entry_camiseta_panther_S.pack()
label_camiseta_panther_M = tk.Label(root, text="Camisetas talla panther M")
label_camiseta_panther_M.pack()
entry_camiseta_panther_M = tk.Entry(root)
entry_camiseta_panther_M.insert(0, str(Camiseta_panther_M))
entry_camiseta_panther_M.pack()
label_camiseta_panther_L = tk.Label(root, text="Camisetas talla panther L")
label_camiseta_panther_L.pack()
entry_camiseta_panther_L = tk.Entry(root)
entry_camiseta_panther_L.insert(0, str(Camiseta_panther_L))
entry_camiseta_panther_L.pack()
label_camiseta_panther_XL = tk.Label(root, text="Camisetas talla panther XL")
label_camiseta_panther_XL.pack()
entry_camiseta_panther_XL = tk.Entry(root)
entry_camiseta_panther_XL.insert(0, str(Camiseta_panther_XL))
entry_camiseta_panther_XL.pack()
label_camiseta_panther_XXL = tk.Label(root, text="Camisetas talla panther XXL")
label_camiseta_panther_XXL.pack()
entry_camiseta_panther_XXL = tk.Entry(root)
entry_camiseta_panther_XXL.insert(0, str(Camiseta_panther_XXL))
entry_camiseta_panther_XXL.pack()

label_chaqueta_S = tk.Label(root, text="Chaquetas talla S")
label_chaqueta_S.pack()
entry_chaqueta_S = tk.Entry(root)
entry_chaqueta_S.insert(0, str(Chaqueta_S))
entry_chaqueta_S.pack()
label_chaqueta_M = tk.Label(root, text="Chaquetas talla M")
label_chaqueta_M.pack()
entry_chaqueta_M = tk.Entry(root)
entry_chaqueta_M.insert(0, str(Chaqueta_M))
entry_chaqueta_M.pack()
label_chaqueta_L = tk.Label(root, text="Chaquetas talla L")
label_chaqueta_L.pack()
entry_chaqueta_L = tk.Entry(root)
entry_chaqueta_L.insert(0, str(Chaqueta_L))
entry_chaqueta_L.pack()
label_chaqueta_XL = tk.Label(root, text="Chaquetas talla XL")
label_chaqueta_XL.pack()
entry_chaqueta_XL = tk.Entry(root)
entry_chaqueta_XL.insert(0, str(Chaqueta_XL))
entry_chaqueta_XL.pack()
label_chaqueta_XXL = tk.Label(root, text="Chaquetas talla XXL")
label_chaqueta_XXL.pack()
entry_chaqueta_XXL = tk.Entry(root)
entry_chaqueta_XXL.insert(0, str(Chaqueta_XXL))
entry_chaqueta_XXL.pack()

label_botella = tk.Label(root, text="Botellas")
label_botella.pack()
entry_botella = tk.Entry(root)
entry_botella.insert(0, str(Botella))
entry_botella.pack()

def actualizar_variables():
    global Camiseta_basica_S, Camiseta_basica_M, Camiseta_basica_L, Camiseta_basica_XL, Camiseta_basica_XXL
    global Camiseta_panther_S, Camiseta_panther_M, Camiseta_panther_L, Camiseta_panther_XL, Camiseta_panther_XXL
    global Chaqueta_S, Chaqueta_M, Chaqueta_L, Chaqueta_XL, Chaqueta_XXL
    global Botella

    Camiseta_basica_S = int(entry_camiseta_basica_S.get())
    Camiseta_basica_M = int(entry_camiseta_basica_M.get())
    Camiseta_basica_L = int(entry_camiseta_basica_L.get())
    Camiseta_basica_XL = int(entry_camiseta_basica_XL.get())
    Camiseta_basica_XXL = int(entry_camiseta_basica_XXL.get())

    Camiseta_panther_S = int(entry_camiseta_panther_S.get())
    Camiseta_panther_M = int(entry_camiseta_panther_M.get())
    Camiseta_panther_L = int(entry_camiseta_panther_L.get())
    Camiseta_panther_XL = int(entry_camiseta_panther_XL.get())
    Camiseta_panther_XXL = int(entry_camiseta_panther_XXL.get())

    Chaqueta_S = int(entry_chaqueta_S.get())
    Chaqueta_M = int(entry_chaqueta_M.get())
    Chaqueta_L = int(entry_chaqueta_L.get())
    Chaqueta_XL = int(entry_chaqueta_XL.get())
    Chaqueta_XXL = int(entry_chaqueta_XXL.get())

    Botella = int(entry_botella.get())

    dataframe = pd.read_excel("./assets/STOCK_MERCHANDISING.xlsx")

    dataframe.iloc[8, 1] -= Camiseta_basica_S
    dataframe.iloc[8, 2] -= Camiseta_basica_M
    dataframe.iloc[8, 3] -= Camiseta_basica_L
    dataframe.iloc[8, 4] -= Camiseta_basica_XL
    dataframe.iloc[8, 5] -= Camiseta_basica_XXL

    dataframe.iloc[13, 1] -= Camiseta_panther_S
    dataframe.iloc[13, 2] -= Camiseta_panther_M
    dataframe.iloc[13, 3] -= Camiseta_panther_L
    dataframe.iloc[13, 4] -= Camiseta_panther_XL
    dataframe.iloc[13, 5] -= Camiseta_panther_XXL

    dataframe.iloc[3, 1] -= Chaqueta_S
    dataframe.iloc[3, 2] -= Chaqueta_M
    dataframe.iloc[3, 3] -= Chaqueta_L
    dataframe.iloc[3, 4] -= Chaqueta_XL
    dataframe.iloc[3, 5] -= Chaqueta_XXL

    dataframe.iloc[18, 2] -= Botella

    date = datetime.today().strftime("%d-%m-%Y")
    dataframe.to_excel(f"./assets/STOCK_MERCHANDISING{date}.xlsx", index=False)

    sincronizar_inputs()

def sincronizar_inputs():
    """Actualiza los valores de los inputs basándose en las variables globales."""
    entry_camiseta_basica_S.delete(0, tk.END)
    entry_camiseta_basica_S.insert(0, str(Camiseta_basica_S))
    entry_camiseta_basica_M.delete(0, tk.END)
    entry_camiseta_basica_M.insert(0, str(Camiseta_basica_M))
    entry_camiseta_basica_L.delete(0, tk.END)
    entry_camiseta_basica_L.insert(0, str(Camiseta_basica_L))
    entry_camiseta_basica_XL.delete(0, tk.END)
    entry_camiseta_basica_XL.insert(0, str(Camiseta_basica_XL))
    entry_camiseta_basica_XXL.delete(0, tk.END)
    entry_camiseta_basica_XXL.insert(0, str(Camiseta_basica_XXL))

    entry_camiseta_panther_S.delete(0, tk.END)
    entry_camiseta_panther_S.insert(0, str(Camiseta_panther_S))
    entry_camiseta_panther_M.delete(0, tk.END)
    entry_camiseta_panther_M.insert(0, str(Camiseta_panther_M))
    entry_camiseta_panther_L.delete(0, tk.END)
    entry_camiseta_panther_L.insert(0, str(Camiseta_panther_L))
    entry_camiseta_panther_XL.delete(0, tk.END)
    entry_camiseta_panther_XL.insert(0, str(Camiseta_panther_XL))
    entry_camiseta_panther_XXL.delete(0, tk.END)
    entry_camiseta_panther_XXL.insert(0, str(Camiseta_panther_XXL))

    entry_chaqueta_S.delete(0, tk.END)
    entry_chaqueta_S.insert(0, str(Chaqueta_S))
    entry_chaqueta_M.delete(0, tk.END)
    entry_chaqueta_M.insert(0, str(Chaqueta_M))
    entry_chaqueta_L.delete(0, tk.END)
    entry_chaqueta_L.insert(0, str(Chaqueta_L))
    entry_chaqueta_XL.delete(0, tk.END)
    entry_chaqueta_XL.insert(0, str(Chaqueta_XL))
    entry_chaqueta_XXL.delete(0, tk.END)
    entry_chaqueta_XXL.insert(0, str(Chaqueta_XXL))

    entry_botella.delete(0, tk.END)
    entry_botella.insert(0, str(Botella))

    print(f"Se han actualizado los siguientes valores:\n Camiseta basica s: {Camiseta_basica_S}, Camiseta panther xxl: {Camiseta_panther_XXL}, Botellas: {Botella}, Chaqueta L: {Chaqueta_L}")

# Botón para actualizar stock
boton_actualizar_stock = tk.Button(root, text="Actualizar stock", command=actualizar_variables)
boton_actualizar_stock.pack(pady=20)

# Botón para salir
boton_salir = tk.Button(root, text="Salir", command=root.quit)
boton_salir.pack(pady=20)

# Mostrar la ventana
root.mainloop()
