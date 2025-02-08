# Instrucciones para ejecutar el proyecto

Para ejecutar el proyecto es necesario hacerlo a partir de dos archivos de google colab y extraer los datasets desde google drive. Sin embargo, el código por sí solo se encarga de esto, para lo cuál solo hay que seguir los pasos:

1. - Abre el enlace compartido de los datasets: [Enlace a la carpeta compartida](https://drive.google.com/drive/u/2/folders/1sk0ZLmCjKVPfSygOlr-_2cdEHlqYzJjR)
   - Haz clic en "Añadir a Mi Unidad" para copiar los archivos a tu Drive.
   - Monta tu Google Drive en Colab:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```

2. Abre el notebook en Google Colab: ProyectoDeteccionPlacasYOLO_gt.ipynb

3. Ejecuta todas las celdas del notebook.

4. Obtendrás una carpeta llamada runs que contiene el modelo, la cuál ya se encuentra también en el drive llamada como la subcarpeta de runs: detect. Sin embargo, puedes modificarla por la que entrenaste nuevamente reemplazandola por la nueva carpeta \content\runs\detect

5. Abre el notebook en Google Colab: ProyectoPlacasIntegracionYOLO_OCR_gt.ipynb

6. Ejecuta todas las celdas del notebook.