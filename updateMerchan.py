import numpy as np
import pytesseract as pyt ## OCR
import cv2 ## Tratamiento de imagenes
import PIL as pw ## Tratamiento de imagenes
import os ## Manejo de archivos
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
        mostrar_imagen(ruta_archivo)
        escala_grises()
        binarizacion()
        corregir_inclinacion()
        # recorte_imagen()
        # leer_imagen_por_filas()
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

def mostrar_imagen_cv2(nombre_ventana, imagen):
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nombre_ventana, 1000, 1000)
    cv2.imshow(nombre_ventana, imagen)


# Pasamos la imagen a escala de grises

def escala_grises():
    global imagen_grayscale
    imagen_grayscale = cv2.imread(ruta_archivo, cv2.IMREAD_GRAYSCALE)
    mostrar_imagen_cv2("Imagen grayscale", imagen_grayscale)

    return imagen_grayscale

boton_escalagrises = tk.Button(root, text="Pasar la imagen a escala de grises", command=escala_grises)
boton_escalagrises.pack(pady=10)

# Pasamos la imagen a binaria (blanco y negro)

def binarizacion():
    global imagen_binarizada

    # cv2.threshold devuelve 2 valores, con _, imagen_binarizada se deshecha el primer valor y asignamos el segundo a la variable imagen_binarizada
    _, imagen_binarizada = cv2.threshold(imagen_grayscale, 210, 255, cv2.THRESH_BINARY) 
    mostrar_imagen_cv2("Imagen binarizada", imagen_binarizada)

    return imagen_binarizada

boton_binarizacion = tk.Button(root, text="Binarizar la imagen", command=binarizacion)
boton_binarizacion.pack(pady=10)

def corregir_inclinacion():
    global imagen_recta

     # 3. Encontrar los puntos donde hay texto (o líneas)
    puntos = np.where(imagen_binarizada == 255)
    coords = np.column_stack((puntos[1], puntos[0]))  # x, y

    # 4. Aplicar el método RANSAC para encontrar la línea que mejor se ajusta
    # a los puntos (esto es robusto a los valores atípicos)
    _, _, w = np.linalg.svd(coords - np.mean(coords, axis=0))
    vector_inclinacion = w[0]
    angulo_rad = np.arctan2(vector_inclinacion[1], vector_inclinacion[0])
    angulo_grados = np.degrees(angulo_rad)

    # 5. Rotar la imagen para corregir la inclinación
    altura, ancho = imagen_binarizada.shape[:2]
    centro = (ancho / 2, altura / 2)
    matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo_grados, 1.0)
    imagen_recta = cv2.warpAffine(imagen_binarizada, matriz_rotacion, (ancho, altura), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # 6. Mostrar la imagen corregida
    mostrar_imagen_cv2("Imagen corregida", imagen_recta)
    
    return imagen_recta

def recorte_imagen():
    #TODO
    pass

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
