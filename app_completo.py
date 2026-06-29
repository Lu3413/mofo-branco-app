"""
🌱 App Completo - Detector de Mofo Branco
📸 Câmera + Upload + IA + Manual | Juliatti et al. (2013)
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Mofo Branco Completo", page_icon="🌱", layout="wide")
st.title("🌱 Detector de Mofo Branco - Completo")
st.markdown("### 📸 Câmera + Upload + IA + Escala Juliatti et al. (2013)")
st.markdown("---")
# ===== LOGIN =====
if 'autenticado' not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    st.title("🔐 Acesso Restrito")
    usuario = st.text_input("Usuário")
    senha = st.text_input("Senha", type="password")
    
    if st.button("Entrar"):
        if usuario == "admin" and senha == "mofo2024":
            st.session_state.autenticado = True
            st.rerun()
        else:
            st.error("Usuário ou senha incorretos!")
    st.stop()

class EscalaJuliatti:
    def __init__(self):
        self.escala = {
            (0, 1): ("Nota 1 (0%)", "Monitorar"),
            (1, 5): ("Nota 2 (2%)", "Monitorar 3 dias"),
            (5, 15): ("Nota 3 (10%)", "Controle biologico"),
            (15, 35): ("Nota 4 (20%)", "Controle seletivo"),
            (35, 60): ("Nota 5 (50%)", "Controle area total"),
            (60, 100): ("Nota 6 (70%+)", "EMERGENCIAL!")
        }
    
    def classificar(self, severidade):
        for (min_v, max_v), (nota, rec) in self.escala.items():
            if min_v <= severidade < max_v:
                return nota, rec
        return "Nota 6 (70%+)", "EMERGENCIAL!"

def extrair_features(imagem):
    features = []
    img = cv2.resize(imagem, (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    for canal in [img, hsv, lab]:
        for i in range(3):
            features.append(np.mean(canal[:,:,i]))
            features.append(np.std(canal[:,:,i]))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    features.append(np.sum(gray > 200) / gray.size)
    features.append(np.sum(gray > 150) / gray.size)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    features.append(np.mean(np.abs(gray.astype(float) - blur.astype(float))))
    return np.array(features)

@st.cache_resource
def treinar_modelo(dataset_path="C:\\Users\\Usuario\\dataset"):
    X, y = [], []
    for classe, label in [("mofo", 1), ("saudavel", 0)]:
        path = os.path.join(dataset_path, classe)
        if os.path.exists(path):
            for arq in os.listdir(path):
                if arq.endswith(('.jpg', '.png', '.jpeg', '.webp')):
                    img = cv2.imread(os.path.join(path, arq))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        X.append(extrair_features(img))
                        y.append(label)
    if len(X) < 4:
        return None, 0
    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    acc = accuracy_score(y_test, modelo.predict(X_test))
    return modelo, acc

class DetectorCores:
    def remover_reflexos(self, imagem, mascara_branco):
        hsv = cv2.cvtColor(imagem, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
        _, _, v = cv2.split(hsv)
        mascara_brilho = cv2.inRange(v, 220, 255)
        blur = cv2.GaussianBlur(gray.astype(float), (15, 15), 0)
        blur2 = cv2.GaussianBlur(gray.astype(float)**2, (15, 15), 0)
        variancia = blur2 - blur**2
        variancia = np.clip(variancia, 0, 255).astype(np.uint8)
        mascara_liso = cv2.inRange(variancia, 0, 8)
        s = hsv[:,:,1]
        mascara_sat_zero = cv2.inRange(s, 0, 15)
        mascara_reflexo = cv2.bitwise_and(mascara_brilho, mascara_liso)
        mascara_reflexo = cv2.bitwise_and(mascara_reflexo, mascara_sat_zero)
        return cv2.bitwise_and(mascara_branco, cv2.bitwise_not(mascara_reflexo))
    
    def segmentar(self, imagem, limiar_l, limiar_verde):
        lab = cv2.cvtColor(imagem, cv2.COLOR_RGB2LAB)
        l_channel = lab[:,:,0]
        a_channel = lab[:,:,1]
        b_channel = lab[:,:,2]
        hsv = cv2.cvtColor(imagem, cv2.COLOR_RGB2HSV)
        verde_min = np.array([20, max(20, limiar_verde), 30])
        verde_max = np.array([100, 255, 255])
        mascara_verde = cv2.inRange(hsv, verde_min, verde_max)
        mascara_l = cv2.inRange(l_channel, limiar_l, 255)
        mascara_a = cv2.inRange(a_channel, 118, 142)
        mascara_b = cv2.inRange(b_channel, 118, 142)
        mascara_branco = cv2.bitwise_and(mascara_l, mascara_a)
        mascara_branco = cv2.bitwise_and(mascara_branco, mascara_b)
        mascara_branco = self.remover_reflexos(imagem, mascara_branco)
        mascara_planta = cv2.bitwise_or(mascara_verde, mascara_branco)
        kernel_planta = np.ones((5,5), np.uint8)
        kernel_mofo = np.ones((3,3), np.uint8)
        mascara_planta = cv2.morphologyEx(mascara_planta, cv2.MORPH_CLOSE, kernel_planta)
        mascara_planta = cv2.morphologyEx(mascara_planta, cv2.MORPH_OPEN, kernel_planta)
        mascara_branco = cv2.morphologyEx(mascara_branco, cv2.MORPH_CLOSE, kernel_mofo)
        mascara_branco = cv2.morphologyEx(mascara_branco, cv2.MORPH_OPEN, kernel_mofo)
        return mascara_planta, mascara_branco

# ===== SIDEBAR =====
with st.sidebar:
    st.header("🧠 Menu")
    modo = st.radio("🎯 Modo:", ["🧠 IA (Auto)", "🎨 Manual (Cores)", "📸 Câmera"])
    
    if modo == "📸 Câmera":
        st.info("📷 Use a câmera do celular!")
    
    st.markdown("---")
    st.header("🤖 IA")
    if st.button("🚀 Treinar Random Forest"):
        with st.spinner("Treinando..."):
            modelo, acc = treinar_modelo()
            if modelo is not None:
                st.session_state['ia_modelo'] = modelo
                st.session_state['ia_acc'] = acc * 100
                st.success(f"IA treinada! Precisao: {acc*100:.1f}%")
                st.balloons()
            else:
                st.error("Dataset nao encontrado")
    if 'ia_acc' in st.session_state:
        st.metric("Precisao IA", f"{st.session_state['ia_acc']:.1f}%")

# ===== ÁREA PRINCIPAL =====
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Entrada")
    
    if modo == "📸 Câmera":
        camera_photo = st.camera_input("Tire uma foto 📷")
        uploaded_file = None
    else:
        camera_photo = None
        uploaded_file = st.file_uploader("Escolha uma foto", type=['jpg', 'jpeg', 'png', 'webp'])

with col2:
    st.header("🎚️ Ajustes")
    
    if modo == "🎨 Manual (Cores)":
        limiar_l = st.slider("Limiar L", 10, 200, 130)
        limiar_verde = st.slider("Sensibilidade Verde", 10, 100, 30)
    elif modo == "📸 Câmera":
        limiar_l = st.slider("Limiar L", 100, 180, 130)
        limiar_verde = 30

# ===== PROCESSAMENTO =====
imagem = None
if uploaded_file is not None:
    imagem = Image.open(uploaded_file)
elif camera_photo is not None:
    imagem = Image.open(camera_photo)

if imagem is not None:
    st.markdown("---")
    st.header("🔍 Resultado")
    
    img_array = np.array(imagem.convert('RGB'))
    escala = EscalaJuliatti()
    detector_vis = DetectorCores()
    
    if modo == "🧠 IA (Auto)" and 'ia_modelo' in st.session_state:
        features = extrair_features(img_array)
        prob = st.session_state['ia_modelo'].predict_proba([features])[0]
        severidade = prob[1] * 100
        st.caption(f"🤖 IA - Confianca: {max(prob)*100:.1f}%")
    else:
        limiar_l = limiar_l if modo != "🧠 IA (Auto)" else 130
        limiar_verde = limiar_verde if modo != "🧠 IA (Auto)" else 30
        mascara_planta, mascara_doenca = detector_vis.segmentar(img_array, limiar_l, limiar_verde)
        area_total = np.sum(mascara_planta > 0)
        area_doente = np.sum(mascara_doenca > 0)
        severidade = (area_doente / area_total * 100) if area_total > 0 else 0
        st.caption("🎨 Modo cores" if modo != "📸 Câmera" else "📸 Câmera")
    
    nota, recomendacao = escala.classificar(severidade)
    
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric("Severidade", f"{severidade:.1f}%")
    with col_r2:
        st.metric("Classificacao", nota)
    with col_r3:
        st.metric("Modo", modo.split()[0])
    
    st.progress(min(severidade/100, 1.0))
    
    if severidade > 35:
        st.error(f"🚨 {recomendacao}")
    elif severidade > 10:
        st.warning(f"⚠️ {recomendacao}")
    else:
        st.success(f"✅ {recomendacao}")
    
    # Máscara visual
    mascara_planta, mascara_doenca = detector_vis.segmentar(img_array, limiar_l if 'limiar_l' in dir() else 130, 30)
    overlay = img_array.copy()
    overlay[mascara_doenca > 0] = [255, 0, 0]
    resultado = cv2.addWeighted(img_array, 0.7, overlay, 0.3, 0)
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.image(img_array, caption="Original", use_container_width=True)
    with col_v2:
        st.image(resultado, caption="Mofo Detectado", use_container_width=True)

else:
    st.info("👆 Escolha uma foto ou tire uma com a câmera")

st.markdown("---")
st.markdown("**🔬 Juliatti, F.C. et al. (2013)** | 🌱 App Completo")