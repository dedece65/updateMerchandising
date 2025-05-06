# Herramienta de Actualización de Inventario de Mercancía

## Introducción

Esta aplicación automatiza el proceso de actualización del inventario de mercancía a partir de formularios escaneados. Utiliza técnicas de visión por computadora y Google Gemini AI para extraer datos de documentos PDF que contienen información de stock de mercancía. La herramienta procesa tablas de estos documentos, identifica artículos, tallas y cantidades, y actualiza una hoja de cálculo de Excel con el inventario correspondiente.

## Características

- **Procesamiento de Documentos PDF**: Convierte documentos PDF a imágenes para su procesamiento
- **Segmentación de Imágenes**: Detecta y extrae automáticamente filas de tablas usando visión por computadora
- **Extracción de Datos Potenciada por IA**: Utiliza Google Gemini AI para interpretar el contenido de cada fila
- **Seguimiento de Inventario**: Actualiza automáticamente el inventario para:
  - Camisetas básicas (S, M, L, XL, XXL)
  - Camisetas Panther (S, M, L, XL, XXL)
  - Chaquetas (S, M, L, XL, XXL)
  - Botellas
- **Interfaz Visual**: GUI simple para seleccionar archivos y ver/editar cambios en el inventario
- **Integración con Excel**: Actualiza hojas de cálculo Excel con los conteos de inventario más recientes

## Requisitos

- Python 3.6 o superior
- Paquetes de Python (ver requirements.txt):
  - numpy
  - pandas
  - opencv-python
  - Pillow
  - pdf2image
  - google.generativeai
  - python-dotenv
  - openpyxl
- Clave API de Google Gemini
- Estructura de directorios apropiada con una carpeta `assets` que contenga `STOCK_MERCHANDISING.xlsx`

## Instalación

1. Clona este repositorio:
   ```
   git clone https://github.com/yourusername/updateMerchandising.git
   cd updateMerchandising
   ```

2. Crea y activa un entorno virtual (recomendado):
   ```
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   ```

3. Instala los paquetes requeridos:
   ```
   pip install -r requirements.txt
   ```

4. Crea un archivo `.env` en la raíz del proyecto con tu clave API de Google Gemini:
   ```
   API_KEY=your_gemini_api_key_here
   ```

5. Crea una carpeta `assets` con tu hoja de cálculo de inventario:
   ```
   mkdir -p assets
   # Añade tu archivo STOCK_MERCHANDISING.xlsx a la carpeta assets
   ```

## Uso

1. Ejecuta la aplicación:
   ```
   python updateMerchan.py
   ```

2. Usa la GUI para:
   - Haz clic en "Seleccionar archivo" para seleccionar un documento PDF que contenga información de actualización de inventario
   - Revisa las cantidades extraídas en los campos de entrada
   - Ajusta los valores manualmente si es necesario
   - Haz clic en "Actualizar stock" para actualizar la hoja de cálculo de inventario
   - Se creará un nuevo archivo Excel en la carpeta assets con la fecha actual en su nombre de archivo

## Cómo Funciona

1. La aplicación convierte el PDF seleccionado en una imagen
2. Se utilizan técnicas de procesamiento de imágenes para detectar líneas horizontales que separan las filas de la tabla
3. Cada fila es procesada por Google Gemini AI para extraer información del producto
4. Los datos extraídos actualizan los conteos de inventario mostrados en la GUI
5. Cuando se hace clic en "Actualizar stock", la aplicación lee el archivo Excel existente, aplica los cambios y guarda una nueva versión con fecha

## Notas Importantes

- Debes obtener una clave API de Google Gemini para usar esta aplicación. Visita [Google AI Studio](https://ai.google.dev/) para obtener una clave.
- La aplicación espera un formato específico para los documentos PDF de entrada (formularios con tablas que contienen información de mercancía)
- La estructura de la hoja de cálculo Excel debe coincidir con el formato esperado, con la información del producto en posiciones de celda específicas

## Licencia

Este proyecto está licenciado bajo los términos del archivo LICENSE incluido en este repositorio.
