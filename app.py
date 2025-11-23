import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile

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
    # Salvar arquivo temporário
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    # Configurar MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # Listas para dados
    dados = []
    
    # Placeholder para o vídeo
    with col_video:
        st.text("Processando vídeo frame a frame...")
        stframe = st.empty()
        
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Processamento Visual
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Pontos: 11=Ombro, 23=Quadril, 25=Joelho
                    shoulder = [landmarks[11].x, landmarks[11].y]
                    hip = [landmarks[23].x, landmarks[23].y]
                    knee = [landmarks[25].x, landmarks[25].y]
                    
                    # Calcular Ângulos
                    hip_angle_raw = calcular_angulo(shoulder, hip, knee)
                    
                    # Converter para flexão de tronco (0 = em pé)
                    flexao_tronco = abs(180 - hip_angle_raw)
                    
                    # Calcular Física
                    torque, shear = calcular_fisica_stiff(flexao_tronco, peso, carga, altura)
                    
                    # Salvar dados
                    dados.append([frame_count, flexao_tronco, torque, shear])
                    
                    # Desenhar Informações na Tela
                    cv2.putText(image, f"Flexao: {int(flexao_tronco)}graus", (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    cv2.putText(image, f"Torque: {int(torque)} Nm", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                except Exception as e:
                    pass
                
                # Mostrar vídeo atualizado
                stframe.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

        cap.release()
        
    # --- EXIBIÇÃO DOS RESULTADOS ---
    st.success("Análise Finalizada!")
    
    df = pd.DataFrame(dados, columns=["Frame", "Angulo_Flexao", "Torque_Lombar", "Shear_Force"])
    
    # Gráficos
    st.divider()
    st.header("3. Relatório Biomecânico")
    
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Torque na Lombar (Nm)")
        st.line_chart(df.set_index("Frame")["Torque_Lombar"])
        
    with g2:
        st.subheader("Força de Cisalhamento (N)")
        st.line_chart(df.set_index("Frame")["Shear_Force"])
        
    # Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Baixar Dados para Excel", data=csv, file_name='biomecanica_stiff.csv', mime='text/csv')