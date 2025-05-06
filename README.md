# Merchandise Inventory Update Tool

## Introduction

This application automates the process of updating merchandise inventory from scanned forms. It uses computer vision techniques and Google's Gemini AI to extract data from PDF documents containing merchandise stock information. The tool processes tables from these documents, identifies product items, sizes, and quantities, and updates an Excel inventory spreadsheet accordingly.

## Features

- **PDF Document Processing**: Convert PDF documents to images for processing
- **Image Segmentation**: Automatically detect and extract table rows using computer vision
- **AI-Powered Data Extraction**: Uses Google's Gemini AI to interpret the content of each row
- **Inventory Tracking**: Automatically updates the inventory for:
  - Basic T-shirts (S, M, L, XL, XXL)
  - Panther T-shirts (S, M, L, XL, XXL)
  - Jackets (S, M, L, XL, XXL)
  - Bottles
- **Visual Interface**: Simple GUI for selecting files and viewing/editing inventory changes
- **Excel Integration**: Updates Excel spreadsheets with the latest inventory counts

## Requirements

- Python 3.6 or higher
- Python packages (see requirements.txt):
  - numpy
  - pandas
  - opencv-python
  - Pillow
  - pdf2image
  - google.generativeai
  - python-dotenv
  - openpyxl
- Google Gemini API key
- Appropriate directory structure with an `assets` folder containing `STOCK_MERCHANDISING.xlsx`

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/updateMerchandising.git
   cd updateMerchandising
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your Google Gemini API key:
   ```
   API_KEY=your_gemini_api_key_here
   ```

5. Create an `assets` folder with your inventory spreadsheet:
   ```
   mkdir -p assets
   # Add your STOCK_MERCHANDISING.xlsx file to the assets folder
   ```

## Usage

1. Run the application:
   ```
   python updateMerchan.py
   ```

2. Use the GUI to:
   - Click "Seleccionar archivo" to select a PDF document containing inventory update information
   - Review the extracted quantities in the input fields
   - Adjust values manually if needed
   - Click "Actualizar stock" to update the inventory spreadsheet
   - A new Excel file will be created in the assets folder with the current date in its filename

## How It Works

1. The application converts the selected PDF to an image
2. Image processing techniques are used to detect horizontal lines that separate table rows
3. Each row is processed by Google's Gemini AI to extract product information
4. The extracted data updates the inventory counts shown in the GUI
5. When "Actualizar stock" is clicked, the application reads the existing Excel file, applies the changes, and saves a new dated version

## Important Notes

- You must obtain a Google Gemini API key to use this application. Visit [Google AI Studio](https://ai.google.dev/) to get a key.
- The application expects a specific format for the input PDF documents (forms with tables containing merchandise information)
- The Excel spreadsheet structure should match the expected format with product information at specific cell positions

## License

This project is licensed under the terms of the LICENSE file included in this repository.

