import numpy as np
import pytesseract as pyt ## OCR
import cv2 ## Tratamiento de imagenes
import PIL as pw ## Tratamiento de imagenes
import tkinter as tk ## GUI

from tkinter import filedialog ## GUI
from PIL import Image, ImageTk ## GUI

## Creamos la ventana con la opción de seleccionar el archivo
root = tk.Tk()
root.geometry("1000x1000")
root.title("Modificar stock de merchandising")

## Seleccionamos el archivo y mostramos la imagen

def seleccionar_archivo():
    global ruta_archivo
    ruta_archivo = filedialog.askopenfilename(title="Seleccionar imagen", filetypes=[("Archivos de imagen", "*.jpg;*.jpeg;*.png")])    
    if ruta_archivo:
        pre_procesar_imagen(ruta_archivo)
        

    return ruta_archivo

def mostrar_imagen(ruta):
    try:
        imagen = Image.open(ruta)
        imagen.thumbnail((600, 600)) 
        imagen_tk = ImageTk.PhotoImage(imagen)
        etiqueta_imagen.config(image=imagen_tk)
        etiqueta_imagen.image = imagen_tk 
    except FileNotFoundError:
        etiqueta_imagen.config(text="No se encontró la imagen")
    except Exception as e:
        etiqueta_imagen.config(text=f"Error al abrir la imagen: {e}")

boton_seleccionar = tk.Button(root, text="Seleccionar imagen", command=seleccionar_archivo)
boton_seleccionar.pack(pady=10) 
                                                                           
etiqueta_imagen = tk.Label(root, text="No se ha seleccionado ninguna imagen")
etiqueta_imagen.pack(pady=10)

# Inicializar variables globales para almacenar imágenes y líneas
def mostrar_imagen_cv2(nombre_ventana, imagen):
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
    
    # mostrar_imagen_cv2("Imagen escalada de grises", imagen_grayscale)
    # mostrar_imagen_cv2("Imagen desenfocada", imagen_desenfocada)
    detectar_lineas_horizontales(imagen_binarizada)

    return imagen_binarizada

# Detectar líneas horizontales en la imagen binarizada
# Devuelve una lista de líneas horizontales con formato (x1, y, x2)
def detectar_lineas_horizontales(imagen_binarizada, umbral_y_cercania=10, longitud_minima=1800):
    global lineas_horizontales
    """
    Detecta líneas horizontales en una imagen binarizada usando la Transformada de Hough Probabilística.

    Args:
        imagen_binarizada: Imagen binarizada (blanco sobre negro).

    Returns:
        Una lista de líneas horizontales detectadas, donde cada línea es una tupla (x1, y1, x2, y2).
    """
    lineas = cv2.HoughLinesP(imagen_binarizada, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=20)

    altura = imagen_binarizada.shape[0]
    margen = int(altura * 0.1)

    if lineas is None:
        return []

    lineas_horizontales = []
    for linea in lineas:
        x1, y1, x2, y2 = linea[0]
        if abs(y1 - y2) < 10:
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
        cv2.line(imagen_con_lineas, (x_inicial, y), (x_final, y), (0, 0, 255), 2)
    
    mostrar_imagen_cv2("Imagen preprocesada", imagen_con_lineas)
    print(f"Se encontraron {len(lineas_unidas)} líneas horizontales unidas y filtradas.")

    return lineas_unidas

# Leer la imagen procesada con tesseract
def leer_imagen_por_filas():
    config = '--psm 6'
    lang = 'spa'  # Idioma español
    texto_tesseract = pyt.image_to_string(imagen_grayscale, config=config, lang=lang)

    print("Salida de Tesseract en bruto:\n", texto_tesseract)

    filas = []
    lineas = texto_tesseract.split('\n')

    for linea in lineas:
        linea = linea.strip()
        if not linea:
            continue

        # Dividir la línea en partes
        partes = linea.split()
        # Verificar si la línea tiene al menos 3 partes
        if len(partes) == 3:
            # Obtener el nombre del producto y el stock
            nombre_producto = partes[0]
            stock = partes[1]
            fecha = partes[2]
            filas.append((nombre_producto, stock, fecha)) 
        elif len(partes) > 3:    
            nombre_producto = " ".join(partes[:2])
            stock = partes[2]
            fecha = partes[3] 
            filas.append((nombre_producto, stock, fecha))
    
    for fila in filas:
        print(fila)

    return filas

root.mainloop()
