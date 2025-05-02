import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()  # Oculta la ventana principal

file_path = filedialog.askopenfilename()

if file_path:
    print(f"Archivo seleccionado: {file_path}")
else:
    print("No se seleccionó ningún archivo.")

root.destroy()