import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

# 1. Carrega o modelo pré-treinado do YOLOv8
model = YOLO('yolov8n.pt')

# IDs das classes no COCO Dataset 
CAR_CLASS_ID = 2        # Carros
MOTORBIKE_CLASS_ID = 3  # Motos
CAT_CLASS_ID = 15       # Gatos
DOG_CLASS_ID = 16       # Cães

# Dicionário com informações das classes
CLASS_INFO = {
    CAR_CLASS_ID: {
        'name': 'Carro',
        'color': (0, 165, 255),  # Laranja (BGR)
        'confidence_threshold': 0.5,
        'count': 0
    },
    MOTORBIKE_CLASS_ID: {
        'name': 'Moto',
        'color': (0, 0, 255),  # Vermelho (BGR)
        'confidence_threshold': 0.5,
        'count': 0
    },
    CAT_CLASS_ID: {
        'name': 'Gato',
        'color': (255, 0, 0),  # Azul (BGR)
        'confidence_threshold': 0.5,
        'count': 0
    },
    DOG_CLASS_ID: {
        'name': 'Cachorro',
        'color': (0, 255, 0),  # Verde (BGR)
        'confidence_threshold': 0.55,  # Sempre mais rigoroso para cães
        'count': 0
    }
}

# Configurações para zoom de cachorro
DOG_ZOOM_FACTOR = 2.5  # Ampliar a região do cachorro 2.5x para detectar melhor
DOG_ZOOM_CONFIDENCE = 0.15  # Confiança mínima para ampliar (muito permissivo)

def ampliar_e_detectar_cachorro(frame, x1, y1, x2, y2):
    """
    Amplia a região do cachorro em múltiplas escalas e executa detecção.
    Retorna a melhor detecção encontrada.
    
    Args:
        frame: Frame original
        x1, y1, x2, y2: Coordenadas da caixa original
    
    Returns:
        Melhor detecção (box) ou None se nenhuma encontrada
    """
    try:
        h, w = frame.shape[:2]
        
        # Calcular o centro e tamanho da caixa original
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        box_width = x2 - x1
        box_height = y2 - y1
        
        best_box = None
        best_conf = 0
        
        # Tentar múltiplos fatores de zoom
        zoom_factors = [1.5, 2.0, 2.5, 3.0]
        
        for zoom_factor in zoom_factors:
            # Calcular novo tamanho com zoom
            new_width = int(box_width * zoom_factor)
            new_height = int(box_height * zoom_factor)
            
            # Calcular novas coordenadas mantendo o centro
            new_x1 = max(0, center_x - new_width // 2)
            new_y1 = max(0, center_y - new_height // 2)
            new_x2 = min(w, center_x + new_width // 2)
            new_y2 = min(h, center_y + new_height // 2)
            
            # Extrair ROI ampliada
            roi = frame[new_y1:new_y2, new_x1:new_x2]
            
            if roi.size == 0:
                continue
            
            # Executar detecção na ROI ampliada
            results = model(roi, verbose=False)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Procurar por cachorro na ROI
                    if cls_id == DOG_CLASS_ID and conf > best_conf:
                        best_conf = conf
                        best_box = box
                        
                        # Se encontrou uma boa confiança, parar
                        if conf > 0.5:
                            return best_box, best_conf, (new_x1, new_y1)
        
        if best_box is not None:
            return best_box, best_conf, (new_x1, new_y1)
        
        return None, 0, (0, 0)
        
    except Exception as e:
        print(f"Erro no zoom: {e}")
        return None, 0, (0, 0)


video_path = 'video1.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erro: Não foi possível abrir o vídeo. Usando webcam...")
    cap = cv2.VideoCapture(0)

print("Pressione 'q' na janela do vídeo para encerrar.")
print(f"Detectando: {', '.join([info['name'] for info in CLASS_INFO.values()])}")

frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    
    # Se o vídeo acabar ou houver falha de leitura, encerra o loop
    if not success:
        print("Fim do vídeo ou arquivo não encontrado.")
        break

    frame_count += 1
    current_frame_detections = defaultdict(int)

    # 3. Roda a inferência do YOLO
    results = model(frame, verbose=False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Verificar se a classe está no dicionário de interesse
            if cls_id in CLASS_INFO:
                class_info = CLASS_INFO[cls_id]
                
                # Se for cachorro com confiança baixa, tentar ampliar para melhor detecção
                if cls_id == DOG_CLASS_ID and DOG_ZOOM_CONFIDENCE <= conf < class_info['confidence_threshold']:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    best_box, best_conf, (roi_x, roi_y) = ampliar_e_detectar_cachorro(frame, x1, y1, x2, y2)
                    
                    # Se encontrou um cachorro melhor na ROI ampliada
                    if best_box is not None and best_conf > class_info['confidence_threshold']:
                        # Converter coordenadas da ROI para o frame original
                        x1 = roi_x + int(best_box.xyxy[0][0])
                        y1 = roi_y + int(best_box.xyxy[0][1])
                        x2 = roi_x + int(best_box.xyxy[0][2])
                        y2 = roi_y + int(best_box.xyxy[0][3])
                        conf = best_conf
                        print(f"✓ Cachorro confirmado por zoom: {conf:.2f}")
                    else:
                        # Não confirmou, pular para a próxima detecção
                        continue
                
                # Aplicar limiar de confiança específico da classe
                if conf > class_info['confidence_threshold']:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Desenhar retângulo
                    cv2.rectangle(frame, (x1, y1), (x2, y2), class_info['color'], 2)
                    
                    # Preparar label com nome da classe e confiança
                    label = f"{class_info['name']} {conf:.2f}"
                    
                    # Desenhar fundo do texto para melhor legibilidade
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, 
                                (x1, y1 - 25), 
                                (x1 + text_size[0], y1), 
                                class_info['color'], -1)
                    
                    # Desenhar texto
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Contar detecção neste frame
                    current_frame_detections[class_info['name']] += 1
                    class_info['count'] += 1

    # Exibir estatísticas no frame
    stats_y = 30
    cv2.putText(frame, f"Frame: {frame_count}", (10, stats_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    stats_y += 35
    for cls_id, info in CLASS_INFO.items():
        color = info['color']
        text = f"{info['name']}: {info['count']} (frame: {current_frame_detections[info['name']]})"
        cv2.putText(frame, text, (10, stats_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        stats_y += 30

    # 4. Exibe o vídeo processado
    cv2.imshow("CampusGuard Vision - Deteccao de Objetos (Merado e Carros)", frame)

    # cv2.waitKey(25) ajusta o tempo entre frames para simular a velocidade real do vídeo (~30 FPS)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        print("\nFinalizando...")
        break

print("\n=== RESUMO FINAL ===")
for info in CLASS_INFO.values():
    print(f"{info['name']}: {info['count']} detecções")

cap.release()
cv2.destroyAllWindows()