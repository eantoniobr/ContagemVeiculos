import math
from collections import defaultdict


def ccw(A, B, C):
    """Verifica orientação no sentido anti-horário de três pontos."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(p1, p2, p3, p4):
    """Determina se o segmento (p1, p2) intercepta o segmento (p3, p4)."""
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def check_gate_crossing(gate, trail, prev_pos, curr_pos, bbox=None):
    """Verifica se o veículo cruzou a linha do portal considerando trajetória e largura da bounding box."""
    g1, g2 = gate
    # 1. Trajetória recente (últimos até 3 passos para evitar perda em quadros rápidos ou pulos)
    if len(trail) >= 2:
        steps = min(3, len(trail) - 1)
        for i in range(1, steps + 1):
            if segments_intersect(g1, g2, trail[-i - 1], trail[-i]):
                return True
    elif segments_intersect(g1, g2, prev_pos, curr_pos):
        return True

    # 2. Base da bounding box (largura do veículo tocando ou cruzando a linha)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        if segments_intersect(g1, g2, (x1, y2), (x2, y2)):
            return True

    return False


class Route:
    def __init__(self, route_id, name, gate_a, gate_b=None, color=(0, 255, 255)):
        """
        route_id: int ou str único
        name: str (ex: "Rota 1 - Viaduto Direita")
        gate_a: ((x1, y1), (x2, y2)) - Linha de Entrada
        gate_b: ((x3, y3), (x4, y4)) - Linha de Saída (opcional, para modo portal duplo)
        color: tupla BGR
        """
        self.route_id = route_id
        self.name = name
        self.gate_a = gate_a
        self.gate_b = gate_b
        self.color = color

        # Contadores desta rota por categoria
        self.counts = {
            "Carro": 0,
            "Moto": 0,
            "Caminhão": 0,
            "Ônibus": 0,
        }

        # Estado dos veículos rastreados: {track_id: {"status": "ENTERED", "frame": frame_idx, "category": cat}}
        self.vehicle_states = {}
        # IDs já contabilizados nesta rota para evitar contagem duplicada
        self.completed_ids = set()

    def to_dict(self):
        return {
            "route_id": self.route_id,
            "name": self.name,
            "gate_a": self.gate_a,
            "gate_b": self.gate_b,
            "color": list(self.color),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            route_id=data["route_id"],
            name=data["name"],
            gate_a=tuple(tuple(p) for p in data["gate_a"]),
            gate_b=tuple(tuple(p) for p in data["gate_b"]) if data.get("gate_b") else None,
            color=tuple(data.get("color", (0, 255, 255))),
        )

    def reset_counts(self):
        for k in self.counts:
            self.counts[k] = 0
        self.vehicle_states.clear()
        self.completed_ids.clear()


class RouteEngine:
    def __init__(self):
        self.routes = []
        self.total_counts = {
            "Carro": 0,
            "Moto": 0,
            "Caminhão": 0,
            "Ônibus": 0,
        }
        # Log histórico de eventos de contagem: list of dicts
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

    def update(self, tracked_vehicles, frame_idx):
        """
        Processa os veículos rastreados no frame atual contra todas as rotas configuradas.
        """
        events = []

        for vehicle in tracked_vehicles:
            track_id = vehicle["id"]
            curr_pos = vehicle["ground_point"]
            prev_pos = vehicle["prev_point"]
            category = vehicle["category"]
            trail = vehicle["trail"]

            # Se o veículo mal se moveu, não precisa testar interseção de mesmo ponto
            if curr_pos == prev_pos and len(trail) < 2:
                continue

            bbox = vehicle.get("bbox")

            for route in self.routes:
                if track_id in route.completed_ids:
                    continue

                # 1. Verifica cruzamento da Linha de Entrada (Gate A)
                crossed_a = check_gate_crossing(route.gate_a, trail, prev_pos, curr_pos, bbox)

                if crossed_a and track_id not in route.vehicle_states:
                    route.vehicle_states[track_id] = {
                        "status": "ENTERED",
                        "frame": frame_idx,
                        "category": category,
                    }

                # 2. Caso a rota possua Linha de Saída (Gate B) -> Portal Duplo
                if route.gate_b is not None:
                    crossed_b = check_gate_crossing(route.gate_b, trail, prev_pos, curr_pos, bbox)

                    if crossed_b and track_id in route.vehicle_states:
                        state = route.vehicle_states[track_id]
                        # Garante que cruzou A antes de B e dentro de uma janela temporal coerente (ex: até 900 frames / 30s)
                        frame_delta = frame_idx - state["frame"]
                        if 1 <= frame_delta <= 900:
                            final_cat = category or state["category"]
                            route.counts[final_cat] += 1
                            self.total_counts[final_cat] += 1
                            route.completed_ids.add(track_id)
                            del route.vehicle_states[track_id]

                            event = {
                                "frame": frame_idx,
                                "track_id": track_id,
                                "route_id": route.route_id,
                                "route_name": route.name,
                                "category": final_cat,
                            }
                            events.append(event)
                            self.event_log.append(event)

                # 3. Caso a rota possua apenas Gate A (portal único de sentido direto)
                elif crossed_a and track_id not in route.completed_ids:
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
        """Retorna o texto de resumo formatado de acordo com o padrão do usuário."""
        lines = []
        for route in self.routes:
            c = route.counts
            lines.append(f"{route.name}: Carros: {c['Carro']}, Caminhão: {c['Caminhão']}, Ônibus: {c['Ônibus']}, Moto: {c['Moto']}")
        t = self.total_counts
        lines.append(f"Total: Carros: {t['Carro']}, Caminhão: {t['Caminhão']}, Ônibus: {t['Ônibus']}, Motos: {t['Moto']}")
        return "\n".join(lines)
