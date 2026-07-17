# test_final.py
import torch
from transferencia_estilo import load_image, style_transfer, guardar_resultado
import numpy as np
from PIL import Image, ImageDraw

def crear_imagenes_prueba():
    """Crea imágenes de prueba realistas"""
    # Contenido: cara simple
    content = Image.new('RGB', (512, 512), 'white')
    draw = ImageDraw.Draw(content)
    # Cara
    draw.ellipse([150, 100, 350, 300], fill='peachpuff', outline='black', width=2)
    # Ojos
    draw.ellipse([200, 150, 220, 170], fill='blue')
    draw.ellipse([280, 150, 300, 170], fill='blue')
    # Boca
    draw.arc([220, 200, 280, 240], 0, 180, fill='red', width=3)
    content.save('test_contenido.jpg')
    
    # Estilo: patrón abstracto
    style = Image.new('RGB', (512, 512), 'darkblue')
    draw = ImageDraw.Draw(style)
    for i in range(0, 512, 32):
        # Rayas diagonales
        draw.line([i, 0, i+32, 512], fill='yellow', width=5)
        draw.line([0, i, 512, i+32], fill='red', width=5)
    # Círculos
    for i in range(100, 412, 100):
        for j in range(100, 412, 100):
            draw.ellipse([i-40, j-40, i+40, j+40], outline='white', width=3)
    style.save('test_estilo.jpg')

if __name__ == "__main__":
    print("🔄 Creando imágenes de prueba...")
    crear_imagenes_prueba()
    
    print("📥 Cargando imágenes...")
    contenido = load_image("test_contenido.jpg", max_dim=512)
    estilo = load_image("test_estilo.jpg", max_dim=512)
    
    print("\n" + "="*70)
    print("🚀 EJECUTANDO TRANSFERENCIA DE ESTILO DEFINITIVA")
    print("="*70)
    
    # Prueba 1: Estilo fuerte
    print("\n🎨 Prueba 1: Estilo fuerte (style_weight=1e7)")
    resultado1 = style_transfer(
        contenido,
        estilo,
        steps=200,
        content_weight=1.0,
        style_weight=1e7,  # ¡Muy alto!
        lr=0.005,
        print_interval=25,
        init_from_noise=False
    )
    guardar_resultado(resultado1, "resultado_fuerte.jpg")
    
    # Prueba 2: Balanceado
    print("\n🎨 Prueba 2: Balanceado (style_weight=5e6)")
    resultado2 = style_transfer(
        contenido,
        estilo,
        steps=200,
        content_weight=1.0,
        style_weight=5e6,  # Balanceado
        lr=0.005,
        print_interval=25,
        init_from_noise=False
    )
    guardar_resultado(resultado2, "resultado_balanceado.jpg")
    
    # Prueba 3: Desde ruido (más artístico)
    print("\n🎨 Prueba 3: Desde ruido (init_from_noise=True)")
    resultado3 = style_transfer(
        contenido,
        estilo,
        steps=200,
        content_weight=1.0,
        style_weight=1e7,
        lr=0.005,
        print_interval=25,
        init_from_noise=True
    )
    guardar_resultado(resultado3, "resultado_artistico.jpg")
    
    print("\n" + "="*70)
    print("✅ TODAS LAS PRUEBAS COMPLETADAS")
    print("="*70)
    print("Resultados guardados:")
    print("1. resultado_fuerte.jpg - Estilo muy fuerte")
    print("2. resultado_balanceado.jpg - Balance contenido/estilo")
    print("3. resultado_artistico.jpg - Desde ruido (artístico)")
    print("\n🎉 ¡Revisa las imágenes generadas!")