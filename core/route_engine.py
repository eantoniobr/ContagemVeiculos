import math
from collections import deque
import cv2
import numpy as np


def ccw(A, B, C):
    """Verifica orientação no sentido anti-horário de três pontos."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(p1, p2, p3, p4):
    """Determina se o segmento (p1, p2) intercepta o segmento (p3, p4)."""
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def point_in_bbox(pt, bbox):
    """Verifica se um ponto (x, y) está dentro da bounding box (x1, y1, x2, y2)."""
    x, y = pt
    x1, y1, x2, y2 = bbox
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def bbox_edges(bbox):
    """Retorna os segmentos das 4 bordas e 2 diagonais da bounding box."""
    x1, y1, x2, y2 = bbox
    return [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
        ((x1, y1), (x2, y2)),
        ((x1, y2), (x2, y1)),
    ]


def bbox_intersects_line(line, bbox):
    """Verifica se a linha corta qualquer borda da bounding box."""
    p1, p2 = line
    if point_in_bbox(p1, bbox) or point_in_bbox(p2, bbox):
        return True
    for e1, e2 in bbox_edges(bbox):
        if segments_intersect(p1, p2, e1, e2):
            return True
    return False


def swept_bbox_intersects_line(line, prev_bbox, curr_bbox):
    """Verifica se o deslocamento da bounding box cruzou a linha entre quadros."""
    if prev_bbox is None or curr_bbox is None:
        return False
    px1, py1, px2, py2 = prev_bbox
    cx1, cy1, cx2, cy2 = curr_bbox
    corners_prev = [(px1, py1), (px2, py1), (px2, py2), (px1, py2)]
    corners_curr = [(cx1, cy1), (cx2, cy1), (cx2, cy2), (cx1, cy2)]
    for cp, cc in zip(corners_prev, corners_curr):
        if segments_intersect(line[0], line[1], cp, cc):
            return True
    return False


def perspective_half_width(y, custom_scale=1.0):
    """
    Calcula a meia-largura (em pixels) do corredor direcional calibrada na medida exata de uma faixa de rodovia:
    - No horizonte distante (y ~ 260..340): meia-largura ~ 14 a 22 px (largura total 28 a 44 px).
    - No primeiro plano (y ~ 650..700): meia-largura ~ 42 a 58 px (largura total 84 a 116 px).
    - custom_scale permite ao usuário ajustar a espessura da faixa (0.4x = muito estreita até 2.0x = muito larga).
    """
    base = max(13.0, min(65.0, 13.0 + 0.095 * max(0.0, float(y) - 250.0)))
    return base * float(custom_scale)


def resample_polyline(points, step_px=28.0):
    """Reamostra uma polilinha para ter vértices espaçados uniformemente."""
    if not points or len(points) < 2:
        return list(points) if points else []

    cleaned = [tuple(map(float, points[0]))]
    for pt in points[1:]:
        if math.hypot(pt[0] - cleaned[-1][0], pt[1] - cleaned[-1][1]) >= 2.0:
            cleaned.append(tuple(map(float, pt)))
    if len(cleaned) < 2:
        p0 = cleaned[0]
        cleaned.append((p0[0] + 10.0, p0[1] + 10.0))

    out = [(int(round(cleaned[0][0])), int(round(cleaned[0][1])))]
    for i in range(len(cleaned) - 1):
        ax, ay = cleaned[i]
        bx, by = cleaned[i + 1]
        seg_len = math.hypot(bx - ax, by - ay)
        n_steps = max(1, int(seg_len // step_px))
        for s in range(1, n_steps + 1):
            frac = s / float(n_steps)
            px = int(round(ax + (bx - ax) * frac))
            py = int(round(ay + (by - ay) * frac))
            if (px, py) != out[-1]:
                out.append((px, py))
    return out


def resample_path_with_widths(path_points, point_widths=None, default_width=1.0, step_px=25.0):
    """
    Reamostra uma polilinha direcional mantendo e interpolando a largura definida para cada ponto.
    Retorna uma lista de tuplas (px, py, pw).
    """
    if not path_points or len(path_points) < 2:
        if not path_points:
            return []
        w0 = point_widths[0] if (point_widths and len(point_widths) > 0) else default_width
        return [(int(round(path_points[0][0])), int(round(path_points[0][1])), float(w0))]

    n_pts = len(path_points)
    widths = []
    for i in range(n_pts):
        if point_widths is not None and i < len(point_widths):
            widths.append(float(point_widths[i]))
        else:
            widths.append(float(default_width))

    cleaned_pts = [tuple(map(float, path_points[0]))]
    cleaned_w = [widths[0]]
    for i in range(1, n_pts):
        pt = tuple(map(float, path_points[i]))
        last = cleaned_pts[-1]
        dist = math.hypot(pt[0] - last[0], pt[1] - last[1])
        if dist >= 2.0:
            cleaned_pts.append(pt)
            cleaned_w.append(widths[i])
        else:
            cleaned_w[-1] = widths[i]

    if len(cleaned_pts) < 2:
        p0 = cleaned_pts[0]
        cleaned_pts.append((p0[0] + 10.0, p0[1] + 10.0))
        cleaned_w.append(cleaned_w[0])

    sampled = [(int(round(cleaned_pts[0][0])), int(round(cleaned_pts[0][1])), float(cleaned_w[0]))]

    for i in range(len(cleaned_pts) - 1):
        ax, ay = cleaned_pts[i]
        bx, by = cleaned_pts[i + 1]
        wa = cleaned_w[i]
        wb = cleaned_w[i + 1]
        seg_len = math.hypot(bx - ax, by - ay)
        n_steps = max(1, int(seg_len // step_px))

        for s in range(1, n_steps + 1):
            frac = s / float(n_steps)
            px = int(round(ax + (bx - ax) * frac))
            py = int(round(ay + (by - ay) * frac))
            pw = float(wa + (wb - wa) * frac)
            if (px, py) != (sampled[-1][0], sampled[-1][1]):
                sampled.append((px, py, pw))

    return sampled


def build_corridor_polygon(path_points, width_scale=1.0, point_widths=None):
    """
    Constrói o polígono de faixa (corredor) ao redor do percurso direcional
    (Início -> Fim) com calibração de perspectiva e suporte a larguras variáveis por ponto.
    """
    sampled = resample_path_with_widths(
        path_points,
        point_widths=point_widths,
        default_width=width_scale,
        step_px=25.0,
    )
    if len(sampled) < 2:
        return []

    left_side = []
    right_side = []
    n = len(sampled)

    for i in range(n):
        px, py, pw = sampled[i]
        if i == 0:
            dx = sampled[1][0] - sampled[0][0]
            dy = sampled[1][1] - sampled[0][1]
        elif i == n - 1:
            dx = sampled[-1][0] - sampled[-2][0]
            dy = sampled[-1][1] - sampled[-2][1]
        else:
            dx = sampled[i + 1][0] - sampled[i - 1][0]
            dy = sampled[i + 1][1] - sampled[i - 1][1]

        norm = math.hypot(dx, dy)
        if norm < 1e-5:
            tx, ty = 1.0, 0.0
        else:
            tx, ty = dx / norm, dy / norm

        nx, ny = -ty, tx
        hw = perspective_half_width(py, custom_scale=pw)

        lx = int(round(px + nx * hw))
        ly = int(round(py + ny * hw))
        rx = int(round(px - nx * hw))
        ry = int(round(py - ny * hw))

        left_side.append((lx, ly))
        right_side.append((rx, ry))

    return left_side + list(reversed(right_side))


def project_point_to_path(pt, path_points):
    """
    Projeta um ponto (x, y) sobre a polilinha direcional path_points.
    Retorna:
        s_norm (float entre 0.0 e 1.0): progresso longitudinal do Início (0.0) ao Fim (1.0)
        d_perp (float): distância perpendicular mínima até a linha central do percurso
        closest_pt (tuple): ponto (x, y) mais próximo sobre a linha central
        tangent (tuple): vetor unitário (tx, ty) da direção do percurso naquele trecho
        total_len (float): comprimento total do percurso em pixels
    """
    if not path_points or len(path_points) < 2:
        return 0.0, float("inf"), pt, (0.0, 1.0), 1.0

    px, py = float(pt[0]), float(pt[1])
    seg_lengths = []
    cum_lengths = [0.0]
    total_len = 0.0

    for i in range(len(path_points) - 1):
        ax, ay = path_points[i]
        bx, by = path_points[i + 1]
        sl = math.hypot(bx - ax, by - ay)
        seg_lengths.append(sl)
        total_len += sl
        cum_lengths.append(total_len)

    if total_len < 1.0:
        return 0.0, math.hypot(px - path_points[0][0], py - path_points[0][1]), path_points[0], (0.0, 1.0), 1.0

    best_dist = float("inf")
    best_s = 0.0
    best_closest = path_points[0]
    best_tan = (0.0, 1.0)

    for i in range(len(path_points) - 1):
        ax, ay = float(path_points[i][0]), float(path_points[i][1])
        bx, by = float(path_points[i + 1][0]), float(path_points[i + 1][1])
        ux, uy = bx - ax, by - ay
        sl = seg_lengths[i]
        if sl < 1e-5:
            continue

        alpha = ((px - ax) * ux + (py - ay) * uy) / (sl * sl)
        alpha_clamped = max(0.0, min(1.0, alpha))

        cx = ax + alpha_clamped * ux
        cy = ay + alpha_clamped * uy
        dist = math.hypot(px - cx, py - cy)

        if dist < best_dist:
            best_dist = dist
            best_s = (cum_lengths[i] + alpha_clamped * sl) / total_len
            best_closest = (cx, cy)
            best_tan = (ux / sl, uy / sl)

    return best_s, best_dist, best_closest, best_tan, total_len


def vehicle_in_polygon(vehicle, polygon_pts, require_center_or_ground=False):
    """Verifica se um veículo detectado intercepta ou está contido no polígono."""
    if not polygon_pts or len(polygon_pts) < 3:
        return False

    poly_np = np.array(polygon_pts, dtype=np.int32)
    bbox = vehicle.get("bbox")
    if not bbox:
        pt = vehicle.get("ground_point") or vehicle.get("center")
        if pt:
            return cv2.pointPolygonTest(poly_np, (float(pt[0]), float(pt[1])), False) >= 0
        return False

    bx1, by1, bx2, by2 = bbox
    cx = (bx1 + bx2) // 2
    cy = (by1 + by2) // 2
    ground_pt = vehicle.get("ground_point", (cx, by2))

    in_ground = cv2.pointPolygonTest(poly_np, (float(ground_pt[0]), float(ground_pt[1])), False) >= 0
    in_center = cv2.pointPolygonTest(poly_np, (float(cx), float(cy)), False) >= 0

    if in_ground or in_center:
        return True

    if require_center_or_ground:
        prev_pt = vehicle.get("prev_point")
        n_pts = len(polygon_pts)
        if prev_pt and prev_pt != ground_pt:
            for i in range(n_pts):
                p_edge = (polygon_pts[i], polygon_pts[(i + 1) % n_pts])
                if segments_intersect(p_edge[0], p_edge[1], prev_pt, ground_pt):
                    return True
        trail = vehicle.get("trail", [])
        if trail and len(trail) >= 2:
            steps = min(4, len(trail) - 1)
            for s in range(1, steps + 1):
                seg_a, seg_b = trail[-s - 1], trail[-s]
                for i in range(n_pts):
                    p_edge = (polygon_pts[i], polygon_pts[(i + 1) % n_pts])
                    if segments_intersect(p_edge[0], p_edge[1], seg_a, seg_b):
                        return True
        return False

    test_points = [(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2)]
    for pt in test_points:
        if cv2.pointPolygonTest(poly_np, (float(pt[0]), float(pt[1])), False) >= 0:
            return True

    for px, py in polygon_pts:
        if bx1 <= px <= bx2 and by1 <= py <= by2:
            return True

    b_edges = [
        ((bx1, by1), (bx2, by1)),
        ((bx2, by1), (bx2, by2)),
        ((bx2, by2), (bx1, by2)),
        ((bx1, by2), (bx1, by1)),
    ]
    n = len(polygon_pts)
    for i in range(n):
        p_edge = (polygon_pts[i], polygon_pts[(i + 1) % n])
        for be in b_edges:
            if segments_intersect(p_edge[0], p_edge[1], be[0], be[1]):
                return True

    trail = vehicle.get("trail", [])
    if trail:
        for pt in trail[-10:]:
            if cv2.pointPolygonTest(poly_np, (float(pt[0]), float(pt[1])), False) >= 0:
                return True

    return False


def point_in_zone(pt, zone):
    """Verifica se o ponto (x, y) está dentro do retângulo da zona (x1, y1, x2, y2)."""
    if not pt or not zone:
        return False
    x, y = pt
    zx1, zy1, zx2, zy2 = zone
    return min(zx1, zx2) <= x <= max(zx1, zx2) and min(zy1, zy2) <= y <= max(zy1, zy2)


def bbox_in_zone(bbox, zone):
    """Verifica se a bounding box intercepta o retângulo da zona."""
    if not bbox or not zone:
        return False
    bx1, by1, bx2, by2 = bbox
    zx1, zy1, zx2, zy2 = zone
    rx1, rx2 = min(zx1, zx2), max(zx1, zx2)
    ry1, ry2 = min(zy1, zy2), max(zy1, zy2)
    return not (bx2 < rx1 or bx1 > rx2 or by2 < ry1 or by1 > ry2)


def check_gate_crossing(gate, trail, prev_pos, curr_pos, bbox=None, prev_bbox=None):
    """Verifica se o veículo cruzou a linha."""
    if not gate:
        return False
    g1, g2 = gate
    if trail and len(trail) >= 2:
        steps = min(15, len(trail) - 1)
        for i in range(1, steps + 1):
            if segments_intersect(g1, g2, trail[-i - 1], trail[-i]):
                return True
    elif prev_pos and curr_pos and segments_intersect(g1, g2, prev_pos, curr_pos):
        return True

    if bbox is not None and bbox_intersects_line(gate, bbox):
        return True

    if prev_bbox is not None and bbox is not None:
        if swept_bbox_intersects_line(gate, prev_bbox, bbox):
            return True

    return False


class Route:
    def __init__(
        self,
        route_id,
        name,
        path_points=None,
        corridor_width=1.0,
        point_widths=None,
        polygon_a=None,
        polygon_b=None,
        gate_a=None,
        gate_b=None,
        zone_a=None,
        zone_b=None,
        color=(0, 255, 255),
    ):
        self.route_id = route_id
        self.name = name
        self.color = color
        self.corridor_width = float(corridor_width) if corridor_width else 1.0

        # Percurso Único Direcional (Início -> Fim)
        self.path_points = [tuple(map(int, p)) for p in path_points] if path_points and len(path_points) >= 2 else None

        # Larguras individuais de cada ponto do percurso (calibração por ponto)
        if point_widths:
            self.point_widths = [float(w) for w in point_widths]
        elif self.path_points:
            self.point_widths = [self.corridor_width] * len(self.path_points)
        else:
            self.point_widths = []

        if self.path_points:
            if len(self.point_widths) < len(self.path_points):
                self.point_widths += [self.corridor_width] * (len(self.path_points) - len(self.point_widths))
            elif len(self.point_widths) > len(self.path_points):
                self.point_widths = self.point_widths[:len(self.path_points)]

        # Geometrias cacheadas para renderização em tempo real ultra-rápida (sem recálculos por quadro)
        self.cum_lengths = [0.0]
        self.corridor_polygon = None
        self.corridor_poly_np = None
        self.resampled_arrows = None
        self.path_total_len = 1.0
        self._rebuild_cache()

        # Compatibilidade com zonas e portais legados (testes unitários e retrocompatibilidade)
        if polygon_a is None and zone_a is not None:
            zx1, zy1, zx2, zy2 = zone_a
            polygon_a = [(zx1, zy1), (zx2, zy1), (zx2, zy2), (zx1, zy2)]
        if polygon_b is None and zone_b is not None:
            zx1, zy1, zx2, zy2 = zone_b
            polygon_b = [(zx1, zy1), (zx2, zy1), (zx2, zy2), (zx1, zy2)]

        self.polygon_a = [tuple(p) for p in polygon_a] if polygon_a else None
        self.polygon_b = [tuple(p) for p in polygon_b] if polygon_b else None
        self.gate_a = gate_a
        self.gate_b = gate_b
        self.zone_a = zone_a
        self.zone_b = zone_b

        # Contadores desta rota (Ordem: Moto, Carro, Caminhão, Ônibus)
        self.counts = {
            "Moto": 0,
            "Carro": 0,
            "Caminhão": 0,
            "Ônibus": 0,
        }

        self.vehicle_states = {}
        self.completed_ids = set()
        self.recent_completions = deque(maxlen=60)

    def _rebuild_cache(self):
        """Reconstrói as geometrias pré-calculadas da rota."""
        if self.path_points and len(self.path_points) >= 2:
            self.cum_lengths = [0.0]
            t_len = 0.0
            for i in range(len(self.path_points) - 1):
                p1, p2 = self.path_points[i], self.path_points[i + 1]
                seg_l = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                t_len += seg_l
                self.cum_lengths.append(t_len)
            self.path_total_len = max(1.0, t_len)

            if len(self.point_widths) != len(self.path_points):
                if len(self.point_widths) < len(self.path_points):
                    self.point_widths += [self.corridor_width] * (len(self.path_points) - len(self.point_widths))
                else:
                    self.point_widths = self.point_widths[:len(self.path_points)]

            self.corridor_polygon = build_corridor_polygon(
                self.path_points,
                width_scale=self.corridor_width,
                point_widths=self.point_widths,
            )
            if self.corridor_polygon and len(self.corridor_polygon) >= 3:
                self.corridor_poly_np = np.array(self.corridor_polygon, dtype=np.int32)
            else:
                self.corridor_poly_np = None

            # Reamostra setas proporcionais ao comprimento do percurso
            arrow_spacing = max(35.0, min(75.0, self.path_total_len / 5.0))
            self.resampled_arrows = resample_polyline(self.path_points, step_px=arrow_spacing)
        else:
            self.cum_lengths = [0.0]
            self.corridor_polygon = None
            self.corridor_poly_np = None
            self.resampled_arrows = None
            self.path_total_len = 1.0

    def get_width_at_s(self, s_norm):
        """Retorna a escala de largura calibrada na fração de progresso s (0.0=Início, 1.0=Fim)."""
        if not self.point_widths:
            return self.corridor_width
        if len(self.point_widths) == 1:
            return float(self.point_widths[0])

        target_d = max(0.0, min(self.path_total_len, float(s_norm) * self.path_total_len))
        if hasattr(self, 'cum_lengths') and len(self.cum_lengths) == len(self.point_widths):
            for i in range(len(self.point_widths) - 1):
                d0 = self.cum_lengths[i]
                d1 = self.cum_lengths[i + 1]
                if d0 <= target_d <= d1 or i == len(self.point_widths) - 2:
                    sl = max(1e-5, d1 - d0)
                    frac = max(0.0, min(1.0, (target_d - d0) / sl))
                    return float(self.point_widths[i] + frac * (self.point_widths[i + 1] - self.point_widths[i]))
            return float(self.point_widths[-1])
        else:
            n = len(self.point_widths)
            val = float(s_norm) * (n - 1)
            idx = int(math.floor(val))
            if idx >= n - 1:
                return float(self.point_widths[-1])
            frac = val - idx
            return float(self.point_widths[idx] + frac * (self.point_widths[idx + 1] - self.point_widths[idx]))

    def set_point_width(self, point_idx, new_width):
        """Ajusta a largura de um ponto específico do percurso."""
        if 0 <= point_idx < len(self.point_widths):
            self.point_widths[point_idx] = max(0.20, min(3.50, float(new_width)))
            self.corridor_width = float(np.mean(self.point_widths))
            self._rebuild_cache()

    def set_corridor_width(self, new_width):
        """Ajusta a escala geral da rota mantendo a proporção entre os pontos."""
        new_w = max(0.20, min(3.50, float(new_width)))
        ratio = new_w / max(0.01, self.corridor_width)
        self.corridor_width = new_w
        if self.point_widths:
            self.point_widths = [max(0.20, min(3.50, round(w * ratio, 2))) for w in self.point_widths]
        self._rebuild_cache()

    def to_dict(self):
        d = {
            "route_id": self.route_id,
            "name": self.name,
            "color": list(self.color),
        }
        if self.path_points:
            d["path_points"] = [list(p) for p in self.path_points]
            d["corridor_width"] = round(self.corridor_width, 2)
            if self.point_widths:
                d["point_widths"] = [round(w, 2) for w in self.point_widths]
        if self.polygon_a:
            d["polygon_a"] = [list(p) for p in self.polygon_a]
        if self.polygon_b:
            d["polygon_b"] = [list(p) for p in self.polygon_b]
        if self.gate_a:
            d["gate_a"] = self.gate_a
        if self.gate_b:
            d["gate_b"] = self.gate_b
        if self.zone_a:
            d["zone_a"] = list(self.zone_a)
        if self.zone_b:
            d["zone_b"] = list(self.zone_b)
        return d

    @classmethod
    def from_dict(cls, data):
        path_pts = [tuple(p) for p in data["path_points"]] if data.get("path_points") else None
        poly_a = [tuple(p) for p in data["polygon_a"]] if data.get("polygon_a") else None
        poly_b = [tuple(p) for p in data["polygon_b"]] if data.get("polygon_b") else None

        return cls(
            route_id=data["route_id"],
            name=data["name"],
            path_points=path_pts,
            corridor_width=data.get("corridor_width", 1.0),
            point_widths=data.get("point_widths"),
            polygon_a=poly_a,
            polygon_b=poly_b,
            gate_a=tuple(tuple(p) for p in data["gate_a"]) if data.get("gate_a") else None,
            gate_b=tuple(tuple(p) for p in data["gate_b"]) if data.get("gate_b") else None,
            zone_a=tuple(data["zone_a"]) if data.get("zone_a") else None,
            zone_b=tuple(data["zone_b"]) if data.get("zone_b") else None,
            color=tuple(data.get("color", (0, 255, 255))),
        )

    def reset_counts(self):
        for k in self.counts:
            self.counts[k] = 0
        self.vehicle_states.clear()
        self.completed_ids.clear()
        self.recent_completions.clear()

    def rollback_to_frame(self, target_frame):
        """Descarta estados e conclusões recentes que ocorreram após target_frame."""
        self.vehicle_states = {
            tid: s for tid, s in self.vehicle_states.items()
            if s.get("frame", 0) <= target_frame
        }
        self.recent_completions = deque(
            [rc for rc in self.recent_completions if rc[0] <= target_frame],
            maxlen=60
        )


class RouteEngine:
    def __init__(self):
        self.routes = []
        self.total_counts = {
            "Moto": 0,
            "Carro": 0,
            "Caminhão": 0,
            "Ônibus": 0,
        }
        self.event_log = []

    def add_route(self, route):
        self.routes.append(route)

    def remove_route(self, route_id):
        self.routes = [r for r in self.routes if r.route_id != route_id]
        self.recalculate_totals()

    def clear_routes(self):
        self.routes.clear()
        self.reset_counts()

    def reset_counts(self):
        for r in self.routes:
            r.reset_counts()
        for k in self.total_counts:
            self.total_counts[k] = 0
        self.event_log.clear()

    def recalculate_totals(self):
        for k in self.total_counts:
            self.total_counts[k] = sum(r.counts[k] for r in self.routes)

    def rollback_to_frame(self, target_frame):
        """Rola o estado das contagens de volta para target_frame."""
        prev_total = sum(self.total_counts.values())
        self.event_log = [e for e in self.event_log if e.get("frame", 0) <= target_frame]

        for route in self.routes:
            for k in route.counts:
                route.counts[k] = 0
            route.completed_ids.clear()
            route.rollback_to_frame(target_frame)

        for k in self.total_counts:
            self.total_counts[k] = 0

        for event in self.event_log:
            r_id = event.get("route_id")
            cat = event.get("category")
            tid = event.get("track_id")

            if cat in self.total_counts:
                self.total_counts[cat] += 1

            for route in self.routes:
                if route.route_id == r_id:
                    if cat in route.counts:
                        route.counts[cat] += 1
                    if tid is not None:
                        route.completed_ids.add(tid)
                    break

        new_total = sum(self.total_counts.values())
        return max(0, prev_total - new_total)

    def _update_single_path_route(self, route, vehicle, frame_idx, active_current_ids, events):
        """
        Avalia um veículo sobre um Percurso Único Direcional (Início -> Fim):
        - Adaptável para percursos curtos ou longos.
        - Exige entrada na primeira metade do percurso.
        - Exige avanço longitudinal positivo no sentido do percurso (rejeita 100% contramão e transversal).
        """
        track_id = vehicle["id"]
        if track_id in route.completed_ids:
            return

        ground_pt = vehicle["ground_point"]
        center_pt = vehicle["center"]
        prev_pt = vehicle.get("prev_point", ground_pt)
        trail = vehicle.get("trail", [])
        category = vehicle["category"]
        bbox = vehicle.get("bbox")

        s_g, d_g, cp_g, tan_g, total_len = project_point_to_path(ground_pt, route.path_points)
        s_c, d_c, cp_c, tan_c, _ = project_point_to_path(center_pt, route.path_points)

        if d_g <= d_c:
            s_curr, d_perp, closest_pt, tan_vec = s_g, d_g, cp_g, tan_g
            ref_pt = ground_pt
        else:
            s_curr, d_perp, closest_pt, tan_vec = s_c, d_c, cp_c, tan_c
            ref_pt = center_pt

        w_curr = route.get_width_at_s(s_curr)
        hw = perspective_half_width(closest_pt[1], w_curr)
        in_corridor = d_perp <= (hw * 1.25)

        # Segmento varrido entre frames
        if not in_corridor and prev_pt and prev_pt != ground_pt:
            mid_pt = ((prev_pt[0] + ground_pt[0]) * 0.5, (prev_pt[1] + ground_pt[1]) * 0.5)
            s_m, d_m, cp_m, tan_m, _ = project_point_to_path(mid_pt, route.path_points)
            w_m = route.get_width_at_s(s_m)
            hw_m = perspective_half_width(cp_m[1], w_m)
            if d_m <= hw_m * 1.20:
                in_corridor = True
                s_curr, d_perp, closest_pt, tan_vec = s_m, d_m, cp_m, tan_m

        if not in_corridor:
            if track_id in route.vehicle_states:
                route.vehicle_states[track_id]["category"] = category
            return

        # Limiares estritos: O veículo DEVE vir da região inicial (Origem) e percorrer o caminho até o fim
        is_short_route = (total_len < 200.0)
        s_entry_threshold = 0.40 if is_short_route else 0.35
        s_exit_threshold = 0.70 if is_short_route else 0.75
        min_progress = 0.35 if is_short_route else 0.40

        if track_id not in route.vehicle_states:
            s_earliest = s_curr
            earliest_pos = ref_pt
            if trail:
                for tp in trail[:10]:
                    s_t, d_t, cp_t, _, _ = project_point_to_path(tp, route.path_points)
                    w_t = route.get_width_at_s(s_t)
                    hw_t = perspective_half_width(cp_t[1], w_t)
                    if d_t <= hw_t * 1.25 and s_t < s_earliest:
                        s_earliest = s_t
                        earliest_pos = tp

            route.vehicle_states[track_id] = {
                "status": "IN_CORRIDOR",
                "frame": frame_idx,
                "last_frame": frame_idx,
                "category": category,
                "s_min": s_earliest,
                "s_max": s_curr,
                "s_last": s_curr,
                "start_pos": earliest_pos,
                "last_pos": ref_pt,
                "frames_in_corridor": 1,
            }
        else:
            st = route.vehicle_states[track_id]
            st["last_frame"] = frame_idx
            st["category"] = category
            st["frames_in_corridor"] = st.get("frames_in_corridor", 1) + 1
            if s_curr < st["s_min"] and (st["s_last"] - s_curr) < 0.08:
                st["s_min"] = s_curr
            if s_curr > st["s_max"]:
                st["s_max"] = s_curr
            st["s_last"] = s_curr
            st["last_pos"] = ref_pt

        st = route.vehicle_states[track_id]

        # Herança estrita de estado apenas para brevíssimas oclusões (ex: postes/placas no mesmo sentido)
        if st["s_min"] > s_entry_threshold and s_curr < 0.65:
            best_orphan_id = None
            best_orphan_s = -1.0
            for cand_id, cand_st in list(route.vehicle_states.items()):
                if cand_id == track_id or cand_id in active_current_ids:
                    continue
                if cand_st.get("s_min", 1.0) > s_entry_threshold:
                    continue
                # Órfão só pode ser herdado se foi visto recentemente (máximo 35 quadros ~ 1.1s)
                dt_f = frame_idx - cand_st.get("last_frame", 0)
                if not (1 <= dt_f <= 35):
                    continue
                # Órfão deve ter categoria compatível
                cand_cat = cand_st.get("category")
                if cand_cat != category:
                    if not (cand_cat in ("Carro", "Ônibus", "Caminhão") and category in ("Carro", "Ônibus", "Caminhão")):
                        continue
                # Deve estar próximo longitudinalmente ao ponto onde o órfão foi visto
                s_diff = s_curr - cand_st.get("s_last", 0.0)
                if not (0.01 <= s_diff <= 0.18):
                    continue

                ox, oy = cand_st["last_pos"]
                vx, vy = ref_pt[0] - ox, ref_pt[1] - oy
                v_norm = math.hypot(vx, vy)
                if 6.0 <= v_norm <= 140.0:
                    cos_align = (vx * tan_vec[0] + vy * tan_vec[1]) / v_norm
                    if cos_align >= 0.70 and cand_st["s_last"] > best_orphan_s:
                        best_orphan_s = cand_st["s_last"]
                        best_orphan_id = cand_id

            if best_orphan_id is not None:
                orphan = route.vehicle_states.pop(best_orphan_id)
                st["s_min"] = orphan["s_min"]
                st["start_pos"] = orphan["start_pos"]
                st["frame"] = orphan["frame"]
                st["frames_in_corridor"] = st.get("frames_in_corridor", 1) + orphan.get("frames_in_corridor", 1)

        # ======================================================================
        # VALIDAÇÃO DE CONCLUSÃO DO PERCURSO NA DIREÇÃO CORRETA (INÍCIO -> FIM)
        # ======================================================================
        # 1. Deve ter alcançado a zona final do percurso (Destino)
        if s_curr < s_exit_threshold or d_perp > (hw * 1.15):
            return

        # 2. OBRIGATÓRIO: Deve ter vindo do início do percurso (s_min <= s_entry_threshold)
        # Impede 100% que veículos que entraram da lateral ou do meio contem!
        if st["s_min"] > s_entry_threshold:
            return

        # 3. Deve ter percorrido avanço longitudinal real ao longo da rota
        delta_s = s_curr - st["s_min"]
        if delta_s < min_progress:
            return

        # 4. Deve ter permanecido tempo suficiente no percurso (rejeita cortes transversais rápidos)
        if st.get("frames_in_corridor", 1) < 3:
            return

        # 5. Validação rigorosa do sentido de deslocamento (impede tráfego transversal e contramão)
        sx, sy = st["start_pos"]
        jx, jy = ref_pt[0] - sx, ref_pt[1] - sy
        j_len = math.hypot(jx, jy)
        min_journey_px = max(20.0, total_len * 0.35)
        if j_len < min_journey_px:
            return

        # Decomposição em avanço longitudinal (ao longo da via) e desvio transversal (atravessando a via)
        j_longitudinal = jx * tan_vec[0] + jy * tan_vec[1]
        j_transversal = abs(-jx * tan_vec[1] + jy * tan_vec[0])

        # O veículo deve se mover substancialmente no sentido do percurso (cos >= 0.65)
        cos_journey = j_longitudinal / j_len
        if cos_journey < 0.65:
            return

        # O avanço longitudinal deve ser predominante sobre o movimento lateral (impede cortar de lado)
        if j_longitudinal < (1.6 * j_transversal):
            return

        # 6. Alinhamento vetorial da trajetória recente (últimos frames do rastro)
        if trail and len(trail) >= 2:
            k_back = min(len(trail), 8)
            lx = trail[-1][0] - trail[-k_back][0]
            ly = trail[-1][1] - trail[-k_back][1]
            l_len = math.hypot(lx, ly)
            if l_len > 5.0:
                cos_recent = (lx * tan_vec[0] + ly * tan_vec[1]) / l_len
                if cos_recent < 0.60:
                    return

        # 7. Desduplicação espacial para não duplicar contagem no mesmo instante
        cx, cy = center_pt
        for old_frame, old_pos, old_cat in route.recent_completions:
            dt = frame_idx - old_frame
            if dt <= 45 and old_cat == category:
                dist = math.hypot(cx - old_pos[0], cy - old_pos[1])
                if dist < min(140.0, max(45.0, hw * 1.3)):
                    return

        final_cat = category
        route.vehicle_states.pop(track_id, None)

        if bbox is not None:
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            b_area = bw * bh
            if final_cat == "Moto" and (bw > 160 or bh > 160 or b_area > 22000):
                final_cat = "Ônibus" if (bw > 260 or bh > 190) else "Carro"

        route.counts[final_cat] += 1
        self.total_counts[final_cat] += 1
        route.completed_ids.add(track_id)
        route.recent_completions.append((frame_idx, (cx, cy), final_cat))

        event = {
            "frame": frame_idx,
            "track_id": track_id,
            "route_id": route.route_id,
            "route_name": route.name,
            "category": final_cat,
        }
        events.append(event)
        self.event_log.append(event)

    def update(self, tracked_vehicles, frame_idx):
        """Processa os veículos rastreados no frame atual."""
        events = []
        active_current_ids = {v["id"] for v in tracked_vehicles}

        for route in self.routes:
            expired_keys = [
                tid for tid, s in route.vehicle_states.items()
                if (frame_idx - s.get("last_frame", s.get("last_seen_in_a", s.get("frame", 0)))) > 45
            ]
            for ek in expired_keys:
                del route.vehicle_states[ek]

        for vehicle in tracked_vehicles:
            track_id = vehicle["id"]
            curr_pos = vehicle["ground_point"]
            prev_pos = vehicle["prev_point"]
            category = vehicle["category"]
            trail = vehicle["trail"]
            bbox = vehicle.get("bbox")
            prev_bbox = vehicle.get("prev_bbox")

            for route in self.routes:
                if track_id in route.completed_ids:
                    continue

                if route.path_points and len(route.path_points) >= 2:
                    self._update_single_path_route(route, vehicle, frame_idx, active_current_ids, events)
                    continue

                # Modo legado (polígonos A e B)
                entered_origin = False
                if route.polygon_a:
                    if vehicle_in_polygon(vehicle, route.polygon_a, require_center_or_ground=False):
                        entered_origin = True
                elif route.zone_a:
                    if point_in_zone(curr_pos, route.zone_a) or bbox_in_zone(bbox, route.zone_a):
                        entered_origin = True
                elif route.gate_a:
                    if check_gate_crossing(route.gate_a, trail, prev_pos, curr_pos, bbox, prev_bbox):
                        entered_origin = True

                if entered_origin:
                    if track_id not in route.vehicle_states:
                        route.vehicle_states[track_id] = {
                            "status": "ENTERED",
                            "frame": frame_idx,
                            "category": category,
                            "start_pos": curr_pos,
                        }
                    else:
                        route.vehicle_states[track_id]["last_seen_in_a"] = frame_idx
                        route.vehicle_states[track_id]["category"] = category
                elif track_id in route.vehicle_states:
                    route.vehicle_states[track_id]["category"] = category

                has_destination = (
                    (route.polygon_b is not None)
                    or (route.zone_b is not None)
                    or (route.gate_b is not None)
                )

                if has_destination:
                    reached_dest = False
                    if route.polygon_b:
                        if vehicle_in_polygon(vehicle, route.polygon_b, require_center_or_ground=True):
                            reached_dest = True
                    elif route.zone_b:
                        if point_in_zone(curr_pos, route.zone_b):
                            reached_dest = True
                    elif route.gate_b:
                        if check_gate_crossing(route.gate_b, trail, prev_pos, curr_pos, bbox, prev_bbox):
                            reached_dest = True

                    if reached_dest:
                        name_lower = route.name.lower()
                        is_valid_direction = True

                        if trail and len(trail) >= 2:
                            steps = min(len(trail), 12)
                            v_dx = trail[-1][0] - trail[-steps][0]
                            v_dy = trail[-1][1] - trail[-steps][1]
                            total_dx = trail[-1][0] - trail[0][0]
                            total_dy = trail[-1][1] - trail[0][1]

                            if "direita" in name_lower or "cam1" in name_lower:
                                if v_dx < -15 or v_dy < -15 or total_dx < -25:
                                    is_valid_direction = False
                            elif "reto" in name_lower or "frente" in name_lower:
                                if v_dy < -20 or v_dx > 120 or total_dy < 40:
                                    is_valid_direction = False

                        if not is_valid_direction:
                            continue

                        matched_state_id = None
                        if track_id in route.vehicle_states:
                            matched_state_id = track_id
                        else:
                            trail_passed_a = False
                            if route.polygon_a and trail:
                                poly_a_np = np.array(route.polygon_a, dtype=np.int32)
                                for tp in trail:
                                    if cv2.pointPolygonTest(poly_a_np, (float(tp[0]), float(tp[1])), False) >= 0:
                                        trail_passed_a = True
                                        break

                            if trail_passed_a:
                                for st_id, st_val in list(route.vehicle_states.items()):
                                    if st_val.get("category") == category and st_id not in active_current_ids:
                                        matched_state_id = st_id
                                        break
                                if matched_state_id is None:
                                    matched_state_id = track_id

                            if matched_state_id is None and route.polygon_a and trail and route.vehicle_states:
                                best_candidate = None
                                best_time = float("inf")
                                for st_id, st_val in list(route.vehicle_states.items()):
                                    if st_id in active_current_ids:
                                        continue
                                    st_frame = st_val.get("frame", 0)
                                    delta_f = frame_idx - st_frame
                                    st_cat = st_val.get("category")
                                    if 5 <= delta_f <= 3000 and st_cat == category:
                                        if delta_f < best_time:
                                            best_time = delta_f
                                            best_candidate = st_id
                                if best_candidate is not None:
                                    matched_state_id = best_candidate

                        if matched_state_id is not None:
                            cx, cy = vehicle["center"]
                            is_duplicate = False
                            for old_frame, old_pos, old_cat in route.recent_completions:
                                dt = frame_idx - old_frame
                                if dt <= 45 and old_cat == category:
                                    dist = math.hypot(cx - old_pos[0], cy - old_pos[1])
                                    if dist < 160:
                                        is_duplicate = True
                                        break

                            if not is_duplicate:
                                final_cat = category
                                if matched_state_id in route.vehicle_states:
                                    del route.vehicle_states[matched_state_id]

                                if bbox is not None:
                                    bw = bbox[2] - bbox[0]
                                    bh = bbox[3] - bbox[1]
                                    b_area = bw * bh
                                    if final_cat == "Moto" and (bw > 160 or bh > 160 or b_area > 22000):
                                        final_cat = "Ônibus" if (bw > 260 or bh > 190) else "Carro"

                                route.counts[final_cat] += 1
                                self.total_counts[final_cat] += 1
                                route.completed_ids.add(track_id)
                                route.recent_completions.append((frame_idx, (cx, cy), final_cat))

                                event = {
                                    "frame": frame_idx,
                                    "track_id": track_id,
                                    "route_id": route.route_id,
                                    "route_name": route.name,
                                    "category": final_cat,
                                }
                                events.append(event)
                                self.event_log.append(event)

                elif entered_origin and track_id not in route.completed_ids:
                    is_forward = True
                    if len(trail) >= 2:
                        dy = trail[-1][1] - trail[0][1]
                        if dy < -10:
                            is_forward = False

                    if is_forward:
                        route.counts[category] += 1
                        self.total_counts[category] += 1
                        route.completed_ids.add(track_id)

                        event = {
                            "frame": frame_idx,
                            "track_id": track_id,
                            "route_id": route.route_id,
                            "route_name": route.name,
                            "category": category,
                        }
                        events.append(event)
                        self.event_log.append(event)

        return events

    def get_summary_text(self):
        """Retorna o texto de resumo formatado na ordem: Moto, Carro, Caminhão, Ônibus."""
        lines = []
        for route in self.routes:
            c = route.counts
            lines.append(f"{route.name}: Motos: {c['Moto']}, Carros: {c['Carro']}, Caminhões: {c['Caminhão']}, Ônibus: {c['Ônibus']}")
        t = self.total_counts
        lines.append(f"Total: Motos: {t['Moto']}, Carros: {t['Carro']}, Caminhões: {t['Caminhão']}, Ônibus: {t['Ônibus']}")
        return "\n".join(lines)
