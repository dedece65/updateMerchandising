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

    print(type(imagen_grayscale))

    return imagen_grayscale

boton_escalagrises = tk.Button(root, text="Pasar la imagen a escala de grises", command=escala_grises)
boton_escalagrises.pack(pady=10)

# Pasamos la imagen a binaria (blanco y negro)

def binarizacion():
    global imagen_binarizada
    print(type(imagen_grayscale))
    _, imagen_binarizada = cv2.threshold(imagen_grayscale, 140, 255, cv2.THRESH_BINARY)
    mostrar_imagen_cv2("Imagen binarizada", imagen_binarizada)

    return imagen_binarizada

boton_binarizacion = tk.Button(root, text="Binarizar la imagen", command=binarizacion)
boton_binarizacion.pack(pady=10)

def recorte_imagen():
    #TODO
    pass

root.mainloop()
