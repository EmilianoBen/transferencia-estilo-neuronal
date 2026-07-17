# Transferencia de Estilo Neuronal

Proyecto que hice para aplicar transferencia de estilo artístico a imágenes usando una red VGG19 y PyTorch, con una interfaz web en Flask para poder usarlo sin tocar código cada vez. Sigue el enfoque del paper original de Gatys, Ecker y Bethge ("A Neural Algorithm of Artistic Style", 2015).

La idea es simple: subes una foto, eliges un estilo (cubismo, anime, LEGO, etc.) y el modelo genera una nueva imagen que mantiene el contenido de tu foto pero con la textura y los colores del estilo elegido.

Cómo funciona

Se usa una VGG19 pre-entrenada en ImageNet solo como extractor de características — no se entrena nada, los pesos quedan congelados todo el proceso. De ahí se sacan dos pérdidas:

**Pérdida de contenido**: compara activaciones de una capa intermedia entre la imagen generada y la original, para que no se pierda la estructura de la foto.

**Pérdida de estilo**: usa la Gram Matrix de varias capas para capturar texturas, patrones y distribución de colores, independientemente de dónde estén ubicados los objetos en la imagen.

En vez de entrenar la red, se optimiza directamente los píxeles de la imagen generada con Adam, minimizando ambas pérdidas combinadas. Es más lento que usar un modelo ya entrenado para esto en tiempo real, pero se entiende mejor qué pasa por dentro — que era justo el punto de hacerlo así.

## Decisiones que tomé

Usé VGG19 porque es la arquitectura del paper original, no algo más moderno, precisamente para quedarme lo más cerca posible del método clásico. Los pesos se mantienen congelados todo el tiempo. La transferencia se hace por optimización iterativa sobre la imagen, no con un modelo feed-forward pre-entrenado para esto (que sería mucho más rápido pero no muestra el proceso interno). Agregué detección automática de CUDA para usar GPU cuando hay, y las imágenes se redimensionan a 512x512 para no tardar una eternidad sin sacrificar demasiada calidad.

Con qué está hecho

Python 3, PyTorch + torchvision (para la VGG19), Flask para la parte web, Pillow para procesar imágenes, y CUDA opcional si tu máquina tiene GPU NVIDIA.

 Estilos que tiene ahorita

Están en la carpeta `estilos/` y la app los detecta solos al arrancar — el nombre del archivo (sin extensión) es lo que aparece en el menú: estilo_anime, estilo_cartoon, estilo_cubismo, estilo_lego y estilo_pintura.

Para agregar uno nuevo solo metes la imagen a esa carpeta y reinicias la app. No hay que tocar código.

 Estructura

```
app.py                    → servidor Flask, rutas y flujo general
transferencia_estilo.py   → el modelo: VGG19, Gram Matrix, loop de optimización
diagnostico.py            → pruebas automáticas con imágenes generadas por código
test.py                   → checa si hay GPU disponible
estilos/                  → imágenes de estilo
templates/                → los HTML
static/                   → donde se guarda el resultado
uploads/                  → imágenes que sube el usuario
```

 Cómo correrlo

```bash
git clone https://github.com/EmilianoBen/transferencia-estilo-neuronal.git
cd transferencia-estilo-neuronal
pip install flask torch torchvision pillow
python app.py
```

Entras a `http://localhost:5000`, subes una imagen, eliges estilo y esperas el resultado. Necesitas Python 3.10 o superior. Si tienes GPU NVIDIA con CUDA, la app la usa automáticamente y es bastante más rápido que en CPU.

Para checar si tu máquina tiene GPU disponible:

```bash
python test.py
```

Y si quieres probar el modelo sin pasar por la interfaz web, `diagnostico.py` genera imágenes de prueba solo (no depende de archivos externos) y corre tres variantes: estilo fuerte, balanceado, y una desde ruido que sale más artística.

```bash
python diagnostico.py
```

Parámetros ajustables

Están en `style_transfer()`, dentro de `transferencia_estilo.py`: `steps` (iteraciones, default 500), `content_weight` y `style_weight` (qué tanto pesa cada pérdida, default 1.0 y 1e6), `lr` (learning rate del Adam, default 0.01), e `init_from_noise` (si arranca desde ruido en vez de la imagen original — da resultados más artísticos pero menos fieles al contenido).

Limitaciones

El algoritmo optimiza cada imagen de cero, así que no hay inferencia en tiempo real como en otros métodos de style transfer. En CPU se nota bastante más lento que en GPU. La resolución está limitada a 512x512 por tiempo de procesamiento. Y como no usa un modelo feed-forward pre-entrenado para esto, cada imagen nueva implica correr toda la optimización otra vez — no hay atajos.

Notas tecnicas
Las imagenes se redimensionan a 512x512 pixeles para mantener un equilibrio entre calidad visual y tiempo de procesamiento. El rendimiento mejora significativamente con GPU: las pruebas se realizaron en una RTX 4060, logrando tiempos de procesamiento entre 20 y 40 segundos por imagen. Se eligio PyTorch en lugar de scikit-learn debido a problemas de compatibilidad con los drivers de GPU desactualizados de esta ultima libreria.

Referencias

Gatys, L., Ecker, A. y Bethge, M. — "A Neural Algorithm of Artistic Style" (2015), que es la base de todo el proyecto. Documentación de PyTorch y torchvision para la parte de implementación.

---

Miguel Emiliano Benítez Cedillo — [github.com/EmilianoBen](https://github.com/EmilianoBen)
