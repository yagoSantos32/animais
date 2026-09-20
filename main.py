import cv2
from ultralytics import YOLO
from collections import defaultdict

# =====================================================
# MODELOS
# =====================================================

model = YOLO('yolov8m.pt')

gato_model = YOLO(
    'runs/detect/campusguard_gato/weights/best.pt'
)

# =====================================================
# CLASSES
# =====================================================

PERSON_CLASS_ID = 0
CAR_CLASS_ID = 2
MOTORBIKE_CLASS_ID = 3
CAT_CLASS_ID = 15
DOG_CLASS_ID = 16

CLASS_INFO = {
    PERSON_CLASS_ID: {
        'name': 'Humano',
        'color': (255, 255, 255),
        'threshold': 0.70
    },

    CAR_CLASS_ID: {
        'name': 'Carro',
        'color': (0, 165, 255),
        'threshold': 0.45
    },

    MOTORBIKE_CLASS_ID: {
        'name': 'Moto',
        'color': (0, 0, 255),
        'threshold': 0.45
    },

    CAT_CLASS_ID: {
        'name': 'Gato',
        'color': (255, 0, 0),
        'threshold': 0.40
    },

    DOG_CLASS_ID: {
        'name': 'Cachorro',
        'color': (0, 255, 0),
        'threshold': 0.40
    }
}

# =====================================================
# VÍDEO
# =====================================================

video_path = 'input/rodovia.mp4'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

# =====================================================
# FPS E TAMANHO
# =====================================================

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

# =====================================================
# VARIÁVEIS
# =====================================================

hasAnimal = False

animal = False
veiculo = False
areaRisco = False
abandono = False
pessoa = False

# =====================================================
# ABANDONO
# =====================================================

frames_animal_sozinho = 0

frames_para_abandono = int(fps * 2)

# =====================================================
# GRAVAÇÃO
# =====================================================

recording = False

frames_to_save = 0

video_writer = None

frames_for_5_seconds = int(fps * 5)

# =====================================================
# LOOP PRINCIPAL
# =====================================================

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    # =================================================
    # REINICIA STATUS DO FRAME
    # =================================================

    animal = False
    veiculo = False
    areaRisco = False
    pessoa = False

    # =================================================
    # YOLO ORIGINAL
    # =================================================

    results = model(
        frame,
        imgsz=416,
        conf=0.20,
        classes=[0, 2, 3, 15, 16],
        verbose=False
    )

    current_frame_detections = defaultdict(int)

    # Guarda se o YOLO original encontrou gato
    encontrou_gato_yolo = False

    # =================================================
    # PROCESSA YOLO ORIGINAL
    # =================================================

    for result in results:

        for box in result.boxes:

            cls_id = int(
                box.cls[0]
            )

            conf = float(
                box.conf[0]
            )

            if cls_id not in CLASS_INFO:
                continue

            info = CLASS_INFO[cls_id]

            # Threshold específico da classe
            if conf < info['threshold']:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # =================================================
            # DESENHA DETECÇÃO ORIGINAL
            # =================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                info['color'],
                2
            )

            label = (
                f"{info['name']} {conf:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 5, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                info['color'],
                2
            )

            current_frame_detections[
                info['name']
            ] += 1

            # =================================================
            # GATO
            # =================================================

            if cls_id == CAT_CLASS_ID:

                encontrou_gato_yolo = True

                animal = True

            # =================================================
            # CACHORRO
            # =================================================

            if cls_id == DOG_CLASS_ID:

                animal = True

            # =================================================
            # CARRO OU MOTO
            # =================================================

            if cls_id in [
                CAR_CLASS_ID,
                MOTORBIKE_CLASS_ID
            ]:

                veiculo = True

            # =================================================
            # PESSOA
            # =================================================

            if cls_id == PERSON_CLASS_ID:

                pessoa = True

            # =================================================
            # PRIMEIRA DETECÇÃO DE ANIMAL
            # =================================================

            if (
                cls_id in [
                    CAT_CLASS_ID,
                    DOG_CLASS_ID
                ]
                and not hasAnimal
            ):

                hasAnimal = True

                recording = True

                frames_to_save = (
                    frames_for_5_seconds
                )

                print(
                    "Animal detectado!"
                )

                print(
                    "hasAnimal =",
                    hasAnimal
                )

                fourcc = (
                    cv2.VideoWriter_fourcc(
                        *'mp4v'
                    )
                )

                video_writer = cv2.VideoWriter(
                    'output/animal_detectado.mp4',
                    fourcc,
                    fps,
                    (width, height)
                )

    # =====================================================
    # MODELO DE GATO TREINADO
    #
    # Só executa se o YOLO original NÃO encontrou gato.
    # =====================================================

    if not encontrou_gato_yolo:

        gato_results = gato_model(
            frame,
            imgsz=640,
            conf=0.20,
            verbose=False
        )

        for result in gato_results:

            for box in result.boxes:

                conf = float(
                    box.conf[0]
                )

                # Threshold do modelo treinado
                if conf < 0.25:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # =================================================
                # DESENHA GATO DO MODELO TREINADO
                # =================================================

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    CLASS_INFO[CAT_CLASS_ID]['color'],
                    2
                )

                label = (
                    f"Gato {conf:.2f}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 5, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    CLASS_INFO[CAT_CLASS_ID]['color'],
                    2
                )

                # =================================================
                # MODELO TREINADO ENCONTROU GATO
                # =================================================

                animal = True

                print(
                    "Gato detectado pelo modelo treinado!"
                )

                # =================================================
                # PRIMEIRA DETECÇÃO DE ANIMAL
                # =================================================

                if not hasAnimal:

                    hasAnimal = True

                    recording = True

                    frames_to_save = (
                        frames_for_5_seconds
                    )

                    print(
                        "Animal detectado!"
                    )

                    print(
                        "hasAnimal =",
                        hasAnimal
                    )

                    fourcc = (
                        cv2.VideoWriter_fourcc(
                            *'mp4v'
                        )
                    )

                    video_writer = cv2.VideoWriter(
                        'output/animal_detectado.mp4',
                        fourcc,
                        fps,
                        (width, height)
                    )

    # =====================================================
    # ÁREA DE RISCO
    #
    # Animal + veículo
    # =====================================================

    if animal and veiculo:

        areaRisco = True

        print(
            "ÁREA DE RISCO!"
        )

        print(
            "animal =",
            animal
        )

        print(
            "veiculo =",
            veiculo
        )

        print(
            "areaRisco =",
            areaRisco
        )

    # =====================================================
    # ABANDONO
    #
    # Animal sem pessoa por mais de 2 segundos
    # =====================================================

    if animal and not pessoa:

        frames_animal_sozinho += 1

        if (
            frames_animal_sozinho
            > frames_para_abandono
        ):

            if not abandono:

                abandono = True

                print(
                    "ABANDONO DETECTADO!"
                )

                print(
                    "abandono =",
                    abandono
                )

    else:

        frames_animal_sozinho = 0

        abandono = False

    # =====================================================
    # GRAVAÇÃO DOS 5 SEGUNDOS
    # =====================================================

    if (
        recording
        and frames_to_save > 0
    ):

        video_writer.write(
            frame
        )

        frames_to_save -= 1

        if frames_to_save == 0:

            recording = False

            video_writer.release()

            video_writer = None

            print(
                "Vídeo de 5 segundos salvo como "
                "animal_detectado.mp4"
            )

    # =====================================================
    # MOSTRA ÁREA DE RISCO
    # =====================================================

    if areaRisco:

        cv2.putText(
            frame,
            "AREA DE RISCO",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # =====================================================
    # MOSTRA ABANDONO
    # =====================================================

    if abandono:

        cv2.putText(
            frame,
            "ABANDONO",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # =====================================================
    # MOSTRA FRAME
    # =====================================================

    cv2.imshow(
        "CampusGuard Vision",
        frame
    )

    if (
        cv2.waitKey(1) & 0xFF
        == ord('q')
    ):
        break

# =====================================================
# FINALIZAÇÃO
# =====================================================

if video_writer is not None:

    video_writer.release()

cap.release()

cv2.destroyAllWindows()