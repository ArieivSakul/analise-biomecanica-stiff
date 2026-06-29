import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile

# 1. Configuração única da página (Removi a duplicada)
st.set_page_config(page_title="Simulador de Exercício Stiff", layout="wide")

def aplicar_estilo_customizado():
    estilo_css = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stButton>button {border-radius: 8px; transition: 0.3s;}
    </style>
    """
    st.markdown(estilo_css, unsafe_allow_html=True)

aplicar_estilo_customizado()

# --- FUNÇÕES ---
def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return angle if angle <= 180.0 else 360-angle

def calcular_fisica_stiff(angulo, peso, carga, altura):
    g, theta = 9.81, np.radians(angulo)
    m_tronco = peso * 0.45
    com_tronco = (altura * 0.30) * 0.5
    torque = ((m_tronco * g) * (com_tronco * np.sin(theta))) + ((carga * g) * ((altura * 0.30) * np.sin(theta)))
    shear = ((m_tronco + carga) * g) * np.sin(theta)
    return torque, shear

# --- INTERFACE ---
st.title("🏋️ Simulador Biomecânico: Análise do Stiff")
col_input, col_video = st.columns([1, 2])

with col_input:
    peso = st.number_input("Peso (kg)", value=70.0)
    altura = st.number_input("Altura (m)", value=1.75)
    carga = st.number_input("Carga (kg)", value=20.0)
    uploaded_file = st.file_uploader("Envie o vídeo", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    dados = []
    
    # Opção de Gráfico
    st.sidebar.markdown("---")
    tipo_grafico = st.sidebar.radio("Visualização dos Dados:", ["Gráficos Separados", "Gráfico Comparativo (Único)"])

    col_vid, col_graf = st.columns([1.5, 1], gap="large")
    with col_vid:
        stframe = st.empty()
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame_count += 1
                
                # Sempre converte para exibir no Streamlit
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Se não for frame de processamento, apenas exibe o vídeo limpo
                if frame_count % 5 != 0:
                    stframe.image(img_rgb, width=600)
                    continue
                
                # Processamento e Desenho (apenas nos frames selecionados)
                results = pose.process(img_rgb)
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    ang = calcular_angulo([lm[11].x, lm[11].y], [lm[23].x, lm[23].y], [lm[25].x, lm[25].y])
                    t, s = calcular_fisica_stiff(abs(180-ang), peso, carga, altura)
                    dados.append([frame_count, abs(180-ang), t, s])
                    
                    # Desenho na imagem
                    mp_drawing.draw_landmarks(img_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    cv2.putText(img_rgb, f"Flexao: {int(abs(180-ang))} deg", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    cv2.putText(img_rgb, f"Torque: {int(t)} Nm", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                stframe.image(img_rgb, width=600)
        cap.release()

    if dados:
        df = pd.DataFrame(dados, columns=["Frame", "Flexao", "Torque", "Cisalhamento"])
        with col_graf:
            st.markdown("### 📊 Análise Dinâmica")
            if tipo_grafico == "Gráfico Comparativo (Único)":
                st.line_chart(df.set_index("Frame")[["Torque", "Cisalhamento"]])
            else:
                st.write("**Torque (Nm)**")
                st.line_chart(df.set_index("Frame")["Torque"], color="#FF4B4B")
                st.write("**Cisalhamento (N)**")
                st.line_chart(df.set_index("Frame")["Cisalhamento"], color="#0068C9")
        st.success("Análise Finalizada!")
        st.download_button("📥 Baixar CSV", df.to_csv(index=False).encode('utf-8'), 'biomecanica.csv', 'text/csv')
