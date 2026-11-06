import os
from collections import deque, Counter
from ultralytics import YOLO

# Mapeamento oficial de classes COCO para as 4 categorias solicitadas
CLASS_MAP = {
    1: "Moto",       # bicycle
    2: "Carro",      # car
    3: "Moto",       # motorcycle
    5: "Ônibus",     # bus
    7: "Caminhão",   # truck
}

TARGET_CLASSES = list(CLASS_MAP.keys())

# Cores para desenho (BGR)
CATEGORY_COLORS = {
    "Carro": (0, 255, 0),        # Verde
    "Moto": (255, 165, 0),       # Laranja/Azul claro em BGR: (0, 165, 255)
    "Caminhão": (0, 0, 255),     # Vermelho
    "Ônibus": (255, 0, 255),     # Magenta
}


class VehicleDetectorTracker:
    def __init__(self, model_path="yolo11s.pt", tracker_cfg="custom_tracker.yaml", conf_thresh=0.25):
        self.model_path = model_path
        self.tracker_cfg = tracker_cfg if os.path.exists(tracker_cfg) else "bytetrack.yaml"
        self.conf_thresh = conf_thresh

        # Se yolo11s.pt falhar ou não estiver disponível, faz fallback para yolo11n.pt
        try:
            self.model = YOLO(self.model_path)
        except Exception:
            self.model_path = "yolo11n.pt"
            self.model = YOLO(self.model_path)

        # Histórico de posições por ID para visualização de trajetória
        self.track_trails = {}
        # Histórico de predições de classe por ID para votação de estabilidade
        self.track_class_votes = {}
        # Última posição registrada por ID
        self.last_positions = {}

    def reset(self):
        """Reinicia rastros e estados."""
        self.track_trails.clear()
        self.track_class_votes.clear()
        self.last_positions.clear()

    def process_frame(self, frame):
        """
        Executa inferência e rastreamento no frame fornecido.
        Retorna:
            tracked_vehicles: lista de dicionários com os veículos rastreados ativos
        """
        # Executa detecção com rastreamento persistente (ByteTrack)
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_cfg,
            classes=TARGET_CLASSES,
            conf=self.conf_thresh,
            verbose=False,
        )

        tracked_vehicles = []
        current_ids = set()

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                # Se não possuir ID atribuído pelo rastreador, pula
                if box.id is None:
                    continue

                track_id = int(box.id[0])
                current_ids.add(track_id)

                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                bottom_center = (cx, y2)  # Ponto de contato no solo para maior precisão

                raw_category = CLASS_MAP.get(cls_id, "Carro")

                # Votação de classe para evitar oscilações por oclusão
                if track_id not in self.track_class_votes:
                    self.track_class_votes[track_id] = Counter()
                self.track_class_votes[track_id][raw_category] += 1
                stable_category = self.track_class_votes[track_id].most_common(1)[0][0]

                # Atualiza rastro
                if track_id not in self.track_trails:
                    self.track_trails[track_id] = deque(maxlen=45)
                self.track_trails[track_id].append(bottom_center)

                prev_point = self.last_positions.get(track_id, bottom_center)
                self.last_positions[track_id] = bottom_center

                tracked_vehicles.append({
                    "id": track_id,
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "ground_point": bottom_center,
                    "prev_point": prev_point,
                    "category": stable_category,
                    "confidence": conf,
                    "trail": list(self.track_trails[track_id]),
                })

        # Limpeza periódica de IDs muito antigos que já saíram de cena
        if len(self.track_trails) > 200:
            active_ids = set(self.last_positions.keys())
            for old_id in list(self.track_trails.keys()):
                if old_id not in current_ids and old_id not in active_ids:
                    del self.track_trails[old_id]
                    if old_id in self.track_class_votes:
                        del self.track_class_votes[old_id]

        return tracked_vehicles
