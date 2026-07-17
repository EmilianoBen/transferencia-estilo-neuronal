import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19, VGG19_Weights
from torchvision import transforms
from PIL import Image
import os

# ---- Configuración de dispositivo ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Transformaciones ----
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# ---- Funciones de normalización ----
def normalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    return (tensor - mean) / std

# ---- Carga de imagen ----
def load_image(path, max_dim=512):
    img = Image.open(path).convert("RGB")
    img = img.resize((max_dim, max_dim), Image.Resampling.LANCZOS)
    tensor = preprocess(img).unsqueeze(0).to(device)
    return tensor

# ---- Guardar imagen ----
def deprocess(tensor):
    t = torch.clamp(tensor, 0, 1)
    t = t.cpu().squeeze(0)
    to_pil = transforms.ToPILImage()
    return to_pil(t)

def guardar_resultado(tensor, ruta):
    img = deprocess(tensor)
    img.save(ruta)
    print(f"Imagen guardada en: {ruta}")

# ---- Gram matrix ----
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, -1)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)

# ---- Modelo VGG19 ----
class VGGFeatures(nn.Module):
    def __init__(self, content_idxs, style_idxs):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.content_idxs = content_idxs
        self.style_idxs = style_idxs
        self.all_idxs = sorted(set(content_idxs + style_idxs))

    def forward(self, x):
        outputs = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.all_idxs:
                outputs[i] = x
        
        content_features = [outputs[idx] for idx in self.content_idxs]
        style_features = [outputs[idx] for idx in self.style_idxs]
        
        return content_features, style_features

# ---- Índices de capas ----
content_layer_idxs = [21]   # relu4_2
style_layer_idxs = [0, 5, 10, 19, 28]

# ---- Transferencia de estilo SIMPLIFICADA ----
def style_transfer(content_img, style_img, steps=500, content_weight=1.0, 
                   style_weight=1e6, lr=0.01, print_interval=50, 
                   init_from_noise=False):
    """Transferencia de estilo - Versión simplificada"""
    
    # Inicializar modelo
    model = VGGFeatures(content_layer_idxs, style_layer_idxs).to(device)
    
    # Normalizar imágenes
    content_norm = normalize(content_img)
    style_norm = normalize(style_img)
    
    # Extraer features
    content_features, _ = model(content_norm)
    _, style_features = model(style_norm)
    
    # Calcular Gram matrices del estilo
    gram_style = [gram_matrix(feat) for feat in style_features]
    
    # Inicializar imagen objetivo
    if init_from_noise:
        input_img = content_img.clone() * 0.9 + torch.randn_like(content_img) * 0.1
    else:
        input_img = content_img.clone()
    
    input_img = input_img.requires_grad_(True).to(device)
    optimizer = optim.Adam([input_img], lr=lr)
    
    # Optimización
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        
        # Forward pass
        input_norm = normalize(input_img)
        content_features_img, style_features_img = model(input_norm)
        
        # Pérdida de contenido
        content_loss = torch.tensor(0.0, device=device)
        for cf_img, cf in zip(content_features_img, content_features):
            content_loss += torch.mean((cf_img - cf) ** 2)
        content_loss = content_loss / len(content_features_img)
        weighted_content_loss = content_weight * content_loss
        
        # Pérdida de estilo
        style_loss = torch.tensor(0.0, device=device)
        gram_img = [gram_matrix(feat) for feat in style_features_img]
        
        for gm_img, gm_style in zip(gram_img, gram_style):
            style_loss += torch.mean((gm_img - gm_style) ** 2)
        
        style_loss = style_loss / len(gram_img)
        weighted_style_loss = style_weight * style_loss
        
        # Pérdida total
        total_loss = weighted_content_loss + weighted_style_loss
        
        # Backward y optimización
        total_loss.backward()
        optimizer.step()
        
        # Mantener en rango válido
        with torch.no_grad():
            input_img.clamp_(0, 1)
        
        # Solo mostrar parámetros en los pasos indicados
        if step % print_interval == 0 or step == 1 or step == steps:
            print(f"Paso {step:4d}/{steps} | Loss: {total_loss.item():7.2f} | Cont: {weighted_content_loss.item():5.2f} | Estilo: {weighted_style_loss.item():7.2f}")
    
    return input_img.detach()

# ---- Cargar estilos ----
def cargar_estilos(carpeta_estilos, max_dim=512):
    estilos = {}
    
    if not os.path.exists(carpeta_estilos):
        return estilos
    
    for nombre in os.listdir(carpeta_estilos):
        ruta = os.path.join(carpeta_estilos, nombre)
        if os.path.isfile(ruta) and nombre.lower().endswith(('.png', '.jpg', '.jpeg')):
            key = os.path.splitext(nombre)[0]
            estilos[key] = load_image(ruta, max_dim=max_dim)
    
    return estilos
