import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import pygame  # Importar pygame para reproducir sonido

# Inicializar pygame
pygame.mixer.init()

# Función para calcular el Ratio de Aspecto del Ojo (EAR)
def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])  # Distancia vertical 1
    B = distance.euclidean(eye[2], eye[4])  # Distancia vertical 2
    C = distance.euclidean(eye[0], eye[3])  # Distancia horizontal
    ear = (A + B) / (2.0 * C)
    return ear

# Umbral de EAR para detectar ojos cerrados
EYE_AR_THRESH = 0.2
ALERT_COUNT = 15  # Número de frames consecutivos con ojos cerrados para alerta

# Inicializar Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Índices de los puntos clave de los ojos
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# Captura de video en tiempo real
cap = cv2.VideoCapture(0)
alert_frames = 0  # Contador para activar la alerta

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar caras
    faces = detector(gray)

    for face in faces:
        # Detectar landmarks faciales
        landmarks = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Dibujar puntos clave en el rostro
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Extraer los ojos
        left_eye = landmarks[LEFT_EYE]
        right_eye = landmarks[RIGHT_EYE]

        # Calcular EAR para ambos ojos
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Dibujar contornos de los ojos
        cv2.polylines(frame, [left_eye], True, (255, 0, 0), 1)
        cv2.polylines(frame, [right_eye], True, (255, 0, 0), 1)

        # Comprobar si los ojos están cerrados
        if avg_ear < EYE_AR_THRESH:
            alert_frames += 1
            if alert_frames >= ALERT_COUNT:
                cv2.putText(frame, "ALERTA: OJOS CERRADOS!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                pygame.mixer.music.load("alarma.mp3")  # Cargar el sonido de alarma
                pygame.mixer.music.play()  # Reproducir el sonido de alarma
        else:
            alert_frames = 0

    # Mostrar el video con los resultados
    cv2.imshow("Reconocimiento Facial en Tiempo Real", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
