import math
import os
from collections import deque
import numpy as np

# Desativa auto-instalação do Ultralytics para preservar onnxruntime-directml (GPU)
os.environ["YOLO_AUTOINSTALL"] = "False"

import torch

try:
    import onnxruntime as ort
    _ORIG_ORT_INIT = ort.InferenceSession.__init__

    def _dml_ort_init(self, path_or_bytes, sess_options=None, providers=None, **kwargs):
        avail = ort.get_available_providers()
        if "DmlExecutionProvider" in avail:
            # device_id=0 corresponde à GPU dedicada NVIDIA GeForce RTX 3060 neste ambiente
            providers = [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        return _ORIG_ORT_INIT(self, path_or_bytes, sess_options=sess_options, providers=providers, **kwargs)

    ort.InferenceSession.__init__ = _dml_ort_init
    HAS_DML = "DmlExecutionProvider" in ort.get_available_providers()
except Exception:
    HAS_DML = False

from ultralytics import YOLO

try:
    import ultralytics.nn.backends.onnx as _u_onnx
    _u_onnx.check_requirements = lambda *args, **kwargs: True
except Exception:
    pass

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
    "Moto": (255, 165, 0),       # Laranja/Azul claro em BGR
    "Carro": (0, 255, 0),        # Verde
    "Caminhão": (0, 0, 255),     # Vermelho
    "Ônibus": (255, 0, 255),     # Magenta
}


def _bbox_iou(box_a, box_b):
    """Calcula o Intersection over Union (IoU) entre duas bounding boxes (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter_area / float(area_a + area_b - inter_area)


class VehicleDetectorTracker:
    def __init__(
        self,
        model_path="yolo11l.pt",
        tracker_cfg="custom_tracker.yaml",
        conf_thresh=0.06,
        imgsz=(736, 1280),
    ):
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz

        # Determina o dispositivo ideal (GPU NVIDIA CUDA, DirectML via ONNX, ou CPU)
        self.device = 0 if torch.cuda.is_available() else "cpu"

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg_path = os.path.join(base_dir, tracker_cfg) if not os.path.isabs(tracker_cfg) else tracker_cfg
        self.tracker_cfg = cfg_path if os.path.exists(cfg_path) else "bytetrack.yaml"

        # Inicializa o modelo YOLO em resolução nativa 16:9 HD (1280x736) acelerado por GPU DirectML
        self.model_path = model_path
        self._load_model(model_path)

        # Estruturas do Rastreador Cinemático Híbrido (IoU + Projeção de Velocidade Stride-Aware)
        self.track_trails = {}
        self.track_class_votes = {}
        self.last_positions = {}
        self.last_bboxes = {}
        self.frame_count = 0
        self.last_seen_frame = {}
        self.track_kinematics = {}
        self._next_track_id = 1

    def _load_model(self, model_path):
        """Carrega modelo com suporte automático a aceleração GPU DirectML (.onnx 1280x736) e fallback seguro."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        resolved_pt = os.path.join(base_dir, model_path) if not os.path.isabs(model_path) else model_path
        stem, _ = os.path.splitext(resolved_pt)
        onnx_candidate = stem + ".onnx"

        if HAS_DML and not torch.cuda.is_available() and os.path.exists(onnx_candidate):
            try:
                self.model = YOLO(onnx_candidate, task="detect")
                self.model_path = model_path
                self.using_onnx_dml = True
                # Warm-up imediato da GPU DirectML para resposta instantânea no primeiro frame
                dummy = np.zeros((736, 1280, 3), dtype=np.uint8)
                self.model.predict(dummy, imgsz=self.imgsz, device=self.device, verbose=False)
                return
            except Exception:
                pass

        self.using_onnx_dml = False
        try:
            self.model = YOLO(resolved_pt if os.path.exists(resolved_pt) else model_path)
            self.model_path = model_path
        except Exception:
            fallback_onnx = os.path.join(base_dir, "yolo11s.onnx")
            fallback_pt = os.path.join(base_dir, "yolo11s.pt")
            if HAS_DML and os.path.exists(fallback_onnx):
                self.model = YOLO(fallback_onnx, task="detect")
                self.using_onnx_dml = True
            else:
                self.model = YOLO(fallback_pt if os.path.exists(fallback_pt) else "yolo11n.pt")
            self.model_path = "yolo11s.pt"

    def switch_model(self, new_model_path):
        """Permite alternar entre YOLO11l, YOLO11m, YOLO11s e YOLO11n dinamicamente."""
        if new_model_path != self.model_path:
            self._load_model(new_model_path)
            self.reset()

    def reset(self):
        """Reinicia rastros e estados."""
        self.track_trails.clear()
        self.track_class_votes.clear()
        self.last_positions.clear()
        self.last_bboxes.clear()
        self.frame_count = 0
        self.last_seen_frame.clear()
        self.track_kinematics.clear()
        self._next_track_id = 1

    def _associate_detections(self, detections, curr_frame, frame_step):
        """
        Associa as detecções do frame atual aos rastros existentes combinando:
        1. IoU direto e IoU projetado pela velocidade (Kalman/Cinemático)
        2. Distância Euclidiana até a posição projetada (suporta saltos de 2x..100x onde IoU=0!)
        3. Persistência através de oclusões por pilares e semáforos
        """
        if not detections:
            return []

        max_gap_frames = max(50, frame_step * 8)
        candidate_pairs = []

        for d_idx, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = det["ground_point"]
            bw, bh = x2 - x1, y2 - y1
            raw_cat = det["raw_category"]

            for tid, kin in self.track_kinematics.items():
                dt = curr_frame - kin["last_frame"]
                if dt <= 0 or dt > max_gap_frames:
                    continue

                ox, oy = kin["pos"]
                vx, vy = kin["vx"], kin["vy"]
                obx1, oby1, obx2, oby2 = kin["bbox"]

                # Posição e caixa projetadas pela velocidade
                # Amortecimento gradual da velocidade para intervalos longos (oclusão)
                damp = 0.90 if dt <= 12 else 0.70
                dx_pred = vx * dt * damp
                dy_pred = vy * dt * damp
                pred_x = ox + dx_pred
                pred_y = oy + dy_pred
                pred_bbox = (
                    int(obx1 + dx_pred),
                    int(oby1 + dy_pred),
                    int(obx2 + dx_pred),
                    int(oby2 + dy_pred),
                )

                iou_raw = _bbox_iou((x1, y1, x2, y2), (obx1, oby1, obx2, oby2))
                iou_pred = _bbox_iou((x1, y1, x2, y2), pred_bbox)
                best_iou = max(iou_raw, iou_pred)

                dist_pred = math.hypot(cx - pred_x, cy - pred_y)
                dist_raw = math.hypot(cx - ox, cy - oy)
                eff_dist = min(dist_pred, dist_raw * 0.92)

                # Escala de raio máximo conforme perspectiva (y) e salto temporal (dt)
                persp = 0.65 if cy < 400 else 1.0
                max_radius = max(42.0, min(280.0, (32.0 + dt * 24.0) * persp))

                if best_iou < 0.10 and eff_dist > max_radius:
                    continue

                # Validação de coerência direcional para rastros já consolidados em movimento
                speed_old = math.hypot(vx, vy)
                if speed_old > 1.3 and dist_raw > 18.0 and best_iou < 0.25:
                    disp_x = cx - ox
                    disp_y = cy - oy
                    cos_dir = (disp_x * vx + disp_y * vy) / (dist_raw * speed_old + 1e-6)
                    if cos_dir < -0.25:
                        continue

                # Compatibilidade física de escala
                obw, obh = kin["size"]
                size_ratio = max(bw / max(1, obw), obw / max(1, bw))
                if size_ratio > 2.5 and best_iou < 0.20:
                    continue

                cat_penalty = 0.0 if kin["category"] == raw_cat else 14.0
                cost = eff_dist - (best_iou * 65.0) + cat_penalty + (dt * 0.6)
                candidate_pairs.append((cost, d_idx, tid))

        candidate_pairs.sort(key=lambda item: item[0])

        matched_dets = {}
        used_tracks = set()
        for cost, d_idx, tid in candidate_pairs:
            if d_idx not in matched_dets and tid not in used_tracks:
                matched_dets[d_idx] = tid
                used_tracks.add(tid)

        assigned_ids = []
        for d_idx in range(len(detections)):
            if d_idx in matched_dets:
                assigned_ids.append(matched_dets[d_idx])
            else:
                new_id = self._next_track_id
                self._next_track_id += 1
                assigned_ids.append(new_id)

        return assigned_ids

    def process_frame(self, frame, frame_idx=None, frame_step=1):
        """
        Executa inferência em resolução nativa 16:9 HD (1280x736) acelerada por GPU DirectML
        e rastreamento cinemático invariante à velocidade (1x a 100x/Turbo).
        """
        frame_step = max(1, int(frame_step))
        if frame_idx is not None:
            self.frame_count = int(frame_idx)
        else:
            self.frame_count += frame_step

        curr_frame = self.frame_count

        # Usa predict() direto com NMS agnóstico para que o ByteTrack interno nunca descarte
        # veículos rápidos ou distantes quando frame_step > 1 (velocidades 2x..100x)
        results = self.model.predict(
            source=frame,
            classes=TARGET_CLASSES,
            conf=self.conf_thresh,
            iou=0.50,
            agnostic_nms=True,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        candidates = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bw = x2 - x1
                bh = y2 - y1
                b_area = bw * bh

                # Limiar adaptativo por distância de perspectiva (fundo da pista y2 < 420)
                is_distant = y2 < 420
                min_conf = 0.075 if is_distant else 0.15
                min_dim = 5 if is_distant else 12

                if bw < min_dim or bh < min_dim or conf < min_conf:
                    continue

                # Filtra falsos positivos de pedestres em pé na calçada distante classificados como 'bicycle'
                if is_distant and cls_id == 1 and conf < 0.28 and bh > int(bw * 1.75):
                    continue

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                bottom_center = (cx, y2)
                raw_category = CLASS_MAP.get(cls_id, "Carro")

                if raw_category == "Moto" and (bw > 165 or bh > 165 or b_area > 23000):
                    raw_category = "Ônibus" if (bw > 260 or bh > 190) else "Carro"

                candidates.append({
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "ground_point": bottom_center,
                    "cls_id": cls_id,
                    "raw_category": raw_category,
                    "confidence": conf,
                    "area": b_area,
                    "is_distant": is_distant,
                })

        # Desduplicação intra-frame das detecções brutas (IoU > 0.65)
        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        filtered_dets = []
        for cand in candidates:
            if not any(_bbox_iou(cand["bbox"], ex["bbox"]) > 0.65 for ex in filtered_dets):
                filtered_dets.append(cand)

        # Associação cinemática (IoU + Projeção de Velocidade Stride-Aware)
        assigned_ids = self._associate_detections(filtered_dets, curr_frame, frame_step)

        tracked_vehicles = []
        for det, track_id in zip(filtered_dets, assigned_ids):
            x1, y1, x2, y2 = det["bbox"]
            bw, bh = x2 - x1, y2 - y1
            b_area = det["area"]
            cx, cy = det["center"]
            bottom_center = det["ground_point"]
            conf = det["confidence"]
            raw_category = det["raw_category"]
            cls_id = det["cls_id"]
            is_distant = det["is_distant"]

            prev_seen_f = self.last_seen_frame.get(track_id, curr_frame)
            dt_frames = max(1, curr_frame - prev_seen_f)
            self.last_seen_frame[track_id] = curr_frame

            # Sistema de Votação Exponencial Ponderada por Confiança² e Escala (EMA)
            if track_id not in self.track_class_votes:
                self.track_class_votes[track_id] = {
                    "Moto": 0.0,
                    "Carro": 0.0,
                    "Caminhão": 0.0,
                    "Ônibus": 0.0,
                }

            scores = self.track_class_votes[track_id]
            for k in scores:
                scores[k] *= 0.92

            scale_factor = min(2.5, max(0.6, math.sqrt(b_area) / 55.0))
            vote_weight = (conf ** 2) * scale_factor
            scores[raw_category] = scores.get(raw_category, 0.0) + vote_weight

            stable_category = max(scores.items(), key=lambda item: item[1])[0]
            if stable_category == "Moto" and (bw > 165 or bh > 165 or b_area > 23000):
                stable_category = "Ônibus" if (bw > 260 or bh > 190) else "Carro"

            if track_id not in self.track_trails:
                self.track_trails[track_id] = deque(maxlen=1200)

            prev_point = self.last_positions.get(track_id, bottom_center)
            if self.track_trails[track_id]:
                dist_step = math.hypot(bottom_center[0] - prev_point[0], bottom_center[1] - prev_point[1])
                if dt_frames > 1 and 16.0 < dist_step <= 280.0:
                    n_interp = min(10, max(2, int(dist_step // 18)))
                    for s_idx in range(1, n_interp):
                        frac = s_idx / float(n_interp)
                        ix = int(prev_point[0] + (bottom_center[0] - prev_point[0]) * frac)
                        iy = int(prev_point[1] + (bottom_center[1] - prev_point[1]) * frac)
                        self.track_trails[track_id].append((ix, iy))

            self.track_trails[track_id].append(bottom_center)
            self.last_positions[track_id] = bottom_center

            prev_bbox = self.last_bboxes.get(track_id, (x1, y1, x2, y2))
            self.last_bboxes[track_id] = (x1, y1, x2, y2)

            inst_vx = (bottom_center[0] - prev_point[0]) / float(dt_frames)
            inst_vy = (bottom_center[1] - prev_point[1]) / float(dt_frames)
            if track_id in self.track_kinematics and prev_point != bottom_center:
                old_kin = self.track_kinematics[track_id]
                sm_vx = 0.60 * old_kin["vx"] + 0.40 * inst_vx
                sm_vy = 0.60 * old_kin["vy"] + 0.40 * inst_vy
                first_pos = old_kin["first_pos"]
                first_frame = old_kin["first_frame"]
                hits = old_kin["hits"] + 1
            elif track_id in self.track_kinematics:
                old_kin = self.track_kinematics[track_id]
                sm_vx, sm_vy = old_kin["vx"] * 0.8, old_kin["vy"] * 0.8
                first_pos = old_kin["first_pos"]
                first_frame = old_kin["first_frame"]
                hits = old_kin["hits"] + 1
            else:
                sm_vx, sm_vy = 0.0, 0.0
                first_pos = bottom_center
                first_frame = curr_frame
                hits = 1

            self.track_kinematics[track_id] = {
                "pos": bottom_center,
                "bbox": (x1, y1, x2, y2),
                "vx": sm_vx,
                "vy": sm_vy,
                "last_frame": curr_frame,
                "size": (bw, bh),
                "category": stable_category,
                "first_pos": first_pos,
                "first_frame": first_frame,
                "hits": hits,
            }

            # Filtra ruído estático de 1 frame de baixíssima confiança ou objeto estático cls=1 no fundo
            total_disp = math.hypot(bottom_center[0] - first_pos[0], bottom_center[1] - first_pos[1])
            if is_distant and cls_id == 1 and conf < 0.25 and total_disp < 3.5 and hits > 3:
                continue
            if hits < 2 and conf < 0.14:
                continue

            tracked_vehicles.append({
                "id": track_id,
                "bbox": (x1, y1, x2, y2),
                "prev_bbox": prev_bbox,
                "center": (cx, cy),
                "ground_point": bottom_center,
                "prev_point": prev_point,
                "category": stable_category,
                "confidence": conf,
                "trail": list(self.track_trails[track_id]),
            })

        # Limpeza periódica de IDs antigos que saíram de cena há mais de 200 frames
        if len(self.track_trails) > 350:
            for old_id in list(self.track_trails.keys()):
                if (curr_frame - self.last_seen_frame.get(old_id, 0)) > 200:
                    self.track_trails.pop(old_id, None)
                    self.track_class_votes.pop(old_id, None)
                    self.last_bboxes.pop(old_id, None)
                    self.last_positions.pop(old_id, None)
                    self.track_kinematics.pop(old_id, None)

        return tracked_vehicles
