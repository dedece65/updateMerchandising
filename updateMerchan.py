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

def mostrar_imagen_cv2(nombre_ventana, imagen):
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nombre_ventana, 1000, 1000)
    cv2.imshow(nombre_ventana, imagen)



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
    
    mostrar_imagen_cv2("Imagen escalada de grises", imagen_grayscale)
    mostrar_imagen_cv2("Imagen desenfocada", imagen_desenfocada)
    mostrar_imagen_cv2("Imagen pre-procesada", imagen_binarizada)

    return imagen_binarizada

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
