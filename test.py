
Camiseta_básica_S = 0
Camiseta_básica_M = 0
Camiseta_básica_L = 0
Camiseta_básica_XL = 0
Camiseta_básica_XXL  = 0
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

respuesta = "('Chaqueta', 'XXL', 3, '16/04/2025');('Camiseta básica', 'L', 7, '16/04/2025');('Chaqueta', 'L', 1, '20/04/2025');('Boli', 15, '20/04/2025');('Camiseta básica', 'S', 8, '16/04/2025')"

def parsear_respuesta_gemini(respuesta):
    """
    Parsea la respuesta de Gemini y devuelve una lista de tuplas con los datos extraídos.

    Args:
        respuesta: Respuesta de Gemini en formato de cadena. -> [('Chaqueta', 'XXL', 3, '16/04/2025'), ('Botella', 7, '16/04/2025'), ('Chaqueta', 'L', 1, '20/04/2025'), ('Boli', 15, '20/04/2025'), ('Camiseta básica', 'S', 8, '16/04/2025')]

    Returns:
        Una lista de tuplas con los datos extraídos.
    """

    # Limpiar la respuesta y convertirla a una lista
    lista_de_entradas = respuesta.strip("[]").split(";")

    # Convertir cada elemento de la lista en una tupla
    lista_de_entradas = [tuple(item.strip("()").split(", ")) for item in lista_de_entradas]

    for item in lista_de_entradas:
        print (f"item: {item}")
        if len(item) == 4:
            articulo = item[0].strip("'").replace(" ", "_")+"_"+item[1].strip("'")
            cantidad = int(item[2])
            fecha = item[3]
            print(f"articulo: {articulo}, cantidad: {cantidad}, fecha: {fecha}")
        elif len(item) == 3:
            articulo = item[0].strip("'")
            cantidad = int(item[1])
            fecha = item[2]
            print(f"articulo: {articulo}, cantidad: {cantidad}, fecha: {fecha}")

parsear_respuesta_gemini(respuesta)