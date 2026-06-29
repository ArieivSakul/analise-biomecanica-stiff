import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile

# 1. Configuração da página
st.set_page_config(page_title="Simulador de Exercício Stiff", layout="wide")

def aplicar_estilo_customizado():
    estilo_css = """
    <style>
        /* 1. Esconder elementos padrão do Streamlit para parecer um App nativo */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* 2. Arredondar botões para um visual mais moderno */
        .stButton>button {
            border-radius: 8px;
            transition: 0.3s;
        }
    </style>
    """
    st.markdown(estilo_css, unsafe_allow_html=True)

# 3. Chama a função para aplicar o visual
aplicar_estilo_customizado()

st.title("Análise Biomecânica - Stiff")

# --- CONFIGURAÇÕES DO STREAMLIT ---
st.set_page_config(page_title="Biomecânica Stiff", layout="wide")

# --- FUNÇÕES MATEMÁTICAS ---

def calcular_angulo(a, b, c):
    """Calcula o ângulo entre três pontos (a, b, c). b é o vértice."""
    a = np.array(a) # Primeiro ponto
    b = np.array(b) # Vértice
    c = np.array(c) # Terceiro ponto
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

def calcular_fisica_stiff(angulo_tronco_vertical, peso_paciente, carga_extra, altura_paciente):
    g = 9.81
    theta = np.radians(angulo_tronco_vertical)
    
    # Constantes de proporção (Winter)
    PROP_MASSA_TRONCO = 0.45 
    PROP_COMP_TRONCO = 0.30   
    
    # Massas e Distâncias
    massa_tronco = peso_paciente * PROP_MASSA_TRONCO
    com_tronco = (altura_paciente * PROP_COMP_TRONCO) * 0.5
    braco_tronco = com_tronco * np.sin(theta)
    braco_barra = (altura_paciente * PROP_COMP_TRONCO) * np.sin(theta)
    
    # Torque
    torque_total = ((massa_tronco * g) * braco_tronco) + ((carga_extra * g) * braco_barra)
    
    # Cisalhamento (Shear)
    shear_force = ((massa_tronco + carga_extra) * g) * np.sin(theta)
    
    return torque_total, shear_force

# --- INTERFACE DO USUÁRIO ---

st.title("🏋️ Simulador Biomecânico: Análise do Stiff")
st.markdown("Ferramenta de análise cinemática e cinética para Fisioterapia.")

col_input, col_video = st.columns([1, 2])

with col_input:
    st.header("1. Dados do Paciente")
    peso = st.number_input("Peso Corporal (kg)", value=70.0)
    altura = st.number_input("Altura (m)", value=1.75)
    carga = st.number_input("Carga da Barra (kg)", value=20.0)
    
    st.divider()
    st.header("2. Upload")
    uploaded_file = st.file_uploader("Envie o vídeo (vista lateral)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    dados = []
    
    # colunas ANTES do loop
    col_video, col_graficos = st.columns([1.5, 1], gap="large")
    
    with col_video:
        st.markdown("### 🎥 Rastreamento Cinemático")
        stframe = st.empty()
        
       with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Convertendo para RGB para processar
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # --- OTIMIZAÇÃO: Pular frames para evitar lag ---
                if frame_count % 5 != 0:
                    stframe.image(image_rgb, width=600)
                    continue 

                # Processamento apenas nos frames selecionados
                results = pose.process(image_rgb)
                
                try:
                    landmarks = results.pose_landmarks.landmark
                    shoulder = [landmarks[11].x, landmarks[11].y]
                    hip = [landmarks[23].x, landmarks[23].y]
                    knee = [landmarks[25].x, landmarks[25].y]
                    
                    hip_angle_raw = calcular_angulo(shoulder, hip, knee)
                    flexao_tronco = abs(180 - hip_angle_raw)
                    torque, shear = calcular_fisica_stiff(flexao_tronco, peso, carga, altura)
                    
                    dados.append([frame_count, flexao_tronco, torque, shear])
                    
                    # Desenhar no frame RGB
                    mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    # Adicionar texto no frame
                    cv2.putText(image_rgb, f"Flexao: {int(flexao_tronco)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
                except Exception:
                    pass
                
                stframe.image(image_rgb, width=600)
        cap.release()

    # --- RELATÓRIO FINAL ---
    # Só roda uma vez, quando o vídeo termina
    if len(dados) > 0:
        df = pd.DataFrame(dados, columns=["Frame", "Flexao", "Torque", "Cisalhamento"])
        
        with col_graficos:
            st.markdown("### 📊 Análise Dinâmica Final")
            st.write("**Evolução do Torque (L5/S1)**")
            st.line_chart(df.set_index("Frame")["Torque"], color="#FF4B4B")
            st.write("**Força de Cisalhamento**")
            st.line_chart(df.set_index("Frame")["Cisalhamento"], color="#0068C9")
            
        st.success("Análise Biomecânica Finalizada!")
        
        # Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Baixar Relatório (CSV)", data=csv, file_name='biomecanica_stiff.csv', mime='text/csv')
    else:
        st.error("Não foi possível detectar o movimento no vídeo.")
