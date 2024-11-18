import cv2
import mediapipe as mp
import time

# Initialiser la capture vidéo et les modules Mediapipe
cap = cv2.VideoCapture(0)  # Index 0 pour la caméra intégrée
ptime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

# Configurer la résolution de la capture
frameWidth = 640  # Choisir une résolution plus faible pour de meilleures performances
frameHeight = 480
cap.set(3, frameWidth)
cap.set(4, frameHeight)

while True:
    # Lire l'image de la caméra
    success, img = cap.read()
    if not success:
        break

    # Convertir l'image en RGB pour Mediapipe
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    # Dessiner les points de repère du visage
    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks:
            # Utiliser FACEMESH_TESSELATION comme alternative à FACE_CONNECTIONS
            mpDraw.draw_landmarks(img, facelms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)

    # Calculer et afficher les FPS
    cTime = time.time()
    fps = 1 / (cTime - ptime)
    ptime = cTime
    cv2.putText(img, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    # Afficher l'image avec les points de repère du visage
    cv2.imshow("Reconnaissance Faciale", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Appuyer sur "Esc" pour quitter
        break

cap.release()
cv2.destroyAllWindows()
