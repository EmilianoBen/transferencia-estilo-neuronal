from flask import Flask, render_template, request, redirect, url_for
from transferencia_estilo import load_image, style_transfer, guardar_resultado, cargar_estilos
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULTADO_PATH = 'static/resultado.jpg'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cargar estilos desde carpeta
ESTILOS = cargar_estilos("estilos", max_dim=512)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        estilo_nombre = request.form['estilo']
        archivo = request.files['imagen']
        if archivo and archivo.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            ruta_imagen = os.path.join(UPLOAD_FOLDER, archivo.filename)
            archivo.save(ruta_imagen)

            contenido = load_image(ruta_imagen, max_dim=512)
            estilo = ESTILOS[estilo_nombre]

            resultado = style_transfer(contenido, estilo, steps=500, content_weight=1.0, style_weight=1e6, lr=0.01,print_interval=50,init_from_noise=False)
            guardar_resultado(resultado, RESULTADO_PATH)

            return redirect(url_for('resultado'))

    return render_template('index.html', estilos=ESTILOS.keys())

@app.route('/resultado')
def resultado():
    return render_template('resultado.html', imagen='resultado.jpg')

if __name__ == '__main__':
    app.run(debug=True)
