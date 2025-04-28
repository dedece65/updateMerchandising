from tkinter import filedialog
import numpy as np
import pytesseract as pyt ## OCR
import cv2 ## Tratamiento de imagenes
import PIL as pw ## Tratamiento de imagenes
import tkinter as tk ## GUI
import pdf2image
# import keras_ocr
import os
import google.generativeai as ai

from dotenv import load_dotenv
from PIL import Image ## GUI

# Cargar las variables de entorno desde el archivo .env
load_dotenv()  

## Creamos la ventana con la opción de seleccionar el archivo
root = tk.Tk()
root.geometry("1000x1000")
root.title("Modificar stock de merchandising")

Camiseta_basica_S = 0
Camiseta_basica_M = 0
Camiseta_basica_L = 0
Camiseta_basica_XL = 0
Camiseta_basica_XXL  = 0
Camiseta_panther_S = 0
Camiseta_panther_M = 0
Camiseta_panther_L = 0
Camiseta_panther_XL = 0
Camiseta_panther_XXL = 0 
Chaqueta_S = 0
Chaqueta_M = 0
Chaqueta_L = 0
Chaqueta_XL = 0
Chaqueta_XXL  = 0
Botella = 0

## Seleccionamos el archivo
def seleccionar_archivo():
    global ruta_archivo
    ruta_archivo = "./assets/Escaneo_test_impresora.jpg"
    # ruta_archivo = filedialog.askopenfilename(
    #         title="Seleccionar archivo",
    #         initialdir=os.path.expanduser("~"),
    #         filetypes=[
    #             ("All Files", "*.*"),
    #             ("Image Files", "*.jpg;*.jpeg;*.png"),
    #             ("PDF Files", "*.pdf"),
    #         ]
    #     )
    # if ruta_archivo:
    #     image = pdf2image.convert_from_path(ruta_archivo, dpi=300)
    #     for image in image:
    #         # Convertir la imagen a un formato compatible con PIL
    #         image = image.convert("RGB")
    #         # Guardar la imagen en un archivo temporal
    #         ruta_archivo = "temp_image.jpg"
    #         image.save(ruta_archivo, "JPEG")
    pre_procesar_imagen(ruta_archivo)
    # img = Image.open(ruta_archivo)
    # leer_imagen_completa_ai(ruta_archivo)

    return ruta_archivo


def parsear_respuesta_gemini(respuesta):
    """
    Parsea la respuesta de Gemini y devuelve una lista de tuplas con los datos extraídos.

    Args:
        respuesta: Respuesta de Gemini en formato de cadena. -> [('Chaqueta', 'XXL', 3, '16/04/2025'), ('Botella', 7, '16/04/2025'), ('Chaqueta', 'L', 1, '20/04/2025'), ('Boli', 15, '20/04/2025'), ('Camiseta básica', 'S', 8, '16/04/2025')]

    Returns:
        Una lista de tuplas con los datos extraídos.
    """
    # Limpiar la respuesta y convertirla a una lista
    respuesta_limpia = respuesta.strip("[]()").replace("'", "").replace("(", "").replace(")", "").replace(",", " ").replace("  ", " ")

    lista = respuesta_limpia.split("), (")

    for item in lista:
        print(item)

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

    print(f"\nRespuesta de Gemini para la fila {i}: {response.text}\n")

# Inicializar variables globales para almacenar imágenes y líneas
def mostrar_imagen(nombre_ventana, imagen):
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nombre_ventana, 1000, 1000)
    cv2.imshow(nombre_ventana, imagen)

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
    
    print(f"Se encontraron {len(lineas_unidas)} líneas horizontales unidas y filtradas.")

    segmentar_imagenes(imagen_binarizada, lineas_unidas)   

    return lineas_unidas

def detectar_lineas_verticales(imagen_binarizada, umbral_x_cercanía=10, longitud_minima=20):
    """
    Detecta líneas verticales en una imagen binarizada usando la Transformada de Hough Probabilística.

    Args:
        imagen_binarizada: Imagen binarizada
        umbral_x_cercania: Si 2 lineas estan a menos de este umbral se unen
        longitud_minima: Longitud minima para que una línea se devuelva.
    
    Returns:
        Una lista de líneas verticales detectadas, donde cada línea es una tupla (x1, y1, y2).
    """

    lineas = cv2.HoughLinesP(imagen_binarizada, rho=1, theta=np.pi/180, threshold=100, minLineLength=longitud_minima // 2, maxLineGap=20)

    ancho = imagen_binarizada.shape[1]
    margen = int(ancho * 0.1)

    if lineas is None:
        return []

    lineas_verticales = []
    for linea in lineas:
        x1, y1, x2, y2 = linea[0]
        if abs(x1 - x2) < umbral_x_cercanía:
            # Guardamos las líneas como (x, y_inicial, y_final) para facilitar la unión
            if x1 > margen and x1 < (ancho - margen) and \
               x2 > margen and x2 < (ancho - margen):
                lineas_verticales.append((x1, min(y1, y2), max(y1, y2)))

    lineas_verticales.sort(key=lambda x: x[0])  # Ordenar por 'x'

    lineas_unidas = []
    if lineas_verticales:
        linea_actual_x = lineas_verticales[0][0]
        y_min_actual = lineas_verticales[0][1]
        y_max_actual = lineas_verticales[0][2]

        for i in range(1, len(lineas_verticales)):
            x, y_min, y_max = lineas_verticales[i]
            if abs(x - linea_actual_x) < umbral_x_cercanía:
                y_min_actual = min(y_min_actual, y_min)
                y_max_actual = max(y_max_actual, y_max)
            else:
                if (y_max_actual - y_min_actual) >= longitud_minima:
                    lineas_unidas.append((linea_actual_x, y_min_actual, y_max_actual))
                linea_actual_x = x
                y_min_actual = y_min
                y_max_actual = y_max

        # Añadir la última línea unida (si cumple la longitud mínima)
        if (y_max_actual - y_min_actual) >= longitud_minima:
            lineas_unidas.append((linea_actual_x, y_min_actual, y_max_actual))

    # Visualizar las líneas unidas y filtradas
    imagen_con_lineas = cv2.cvtColor(imagen_binarizada.copy(), cv2.COLOR_GRAY2BGR)
    for x, y_inicial, y_final in lineas_unidas:
        cv2.line(imagen_con_lineas, (x, y_inicial), (x, y_final), (255, 0, 0), 2, cv2.LINE_AA)
    
    print(f"Se encontraron {len(lineas_unidas)} líneas verticales unidas y filtradas.")

    segmentar_casillas(imagen_binarizada, lineas_unidas)

    return lineas_unidas

def segmentar_casillas(imagen_binarizada, lineas_verticales):
    """
    Segmenta la imagen en partes utilizando las líneas verticales detectadas.

    Args:
        imagen_binarizada: Imagen binarizada
        lineas_verticales: Lista de líneas verticales detectadas.
    
    Returns:
        Una lista de imágenes segmentadas.
    """

    columnas_segmentadas = []
    if not lineas_verticales:
        return columnas_segmentadas

    # Extraer las coordenadas 'x' únicas y ordenarlas
    coordenadas_x = sorted(list(set([int(x) for x, _, _ in lineas_verticales])))

    # Si sólo hay una línea vertical, segmentar la imagen en dos partes
    if len(coordenadas_x) == 1:
        x = coordenadas_x[0]
        columna_izquierda = imagen_binarizada[:, :x]
        columna_derecha = imagen_binarizada[:, x:]
        if columna_izquierda.shape[1] > 0:
            columnas_segmentadas.append(columna_izquierda)
        if columna_derecha.shape[1] > 0:
            columnas_segmentadas.append(columna_derecha)
        return columnas_segmentadas

    # La primera columna estará desde el borde izquierdo de la imagen hasta la primera línea
    x_inicial_anterior = 0
    x_final_primera = coordenadas_x[0]

    if x_final_primera > x_inicial_anterior:
        columna = imagen_binarizada[:, x_inicial_anterior:x_final_primera]
        columnas_segmentadas.append(columna)
        x_inicial_anterior = x_final_primera

    for x_final in coordenadas_x[1:]:  # Comenzar desde el segundo valor
        # Asegurarse de que la diferencia horizontal sea positiva
        if x_final > x_inicial_anterior:
            columna = imagen_binarizada[:, x_inicial_anterior:x_final]
            columnas_segmentadas.append(columna)
            x_inicial_anterior = x_final

    if coordenadas_x:
        x_inicial_ultima = coordenadas_x[-1]
        columna_derecha = imagen_binarizada[:, x_inicial_ultima:imagen_binarizada.shape[1]]
        if columna_derecha.shape[1] > 0:
            columnas_segmentadas.append(columna_derecha)

    # config = [
    #     '--psm 6 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZáéíóúüñÑÁÉÍÓÚ',
    #     '--psm 6 -c tessedit_char_whitelist= SMLX',
    #     '--psm 7 -c tessedit_char_whitelist= 0123456789',
    #     '--psm 7 -c tessedit_char_whitelist= 0123456789/',
    # ]

    images = []

    for i in range(len(columnas_segmentadas)):
        cv2.imwrite(f"columna_{i}.jpg", columnas_segmentadas[i])
        images.append(f"columna_{i}.jpg")

    # leer_imagen_por_filas(images)

    # for i in range(0, len(columnas_segmentadas)):
    #     leer_imagen_por_filas(columnas_segmentadas[i], config[i])

    print(f"Se segmentaron {len(columnas_segmentadas)} columnas de la imagen.")
    
    return columnas_segmentadas

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

    return filas_segmentadas

seleccionar_archivo()

root.mainloop()
