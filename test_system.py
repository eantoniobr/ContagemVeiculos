import os
import cv2
from core.detector_tracker import VehicleDetectorTracker
from core.route_engine import Route, RouteEngine
from core.config_manager import ConfigManager


def test_config_manager():
    print("[TEST] Testando ConfigManager...")
    r1 = Route(1, "Teste 1", ((100, 100), (200, 100)), ((100, 300), (200, 300)))
    r2 = Route(2, "Teste 2", ((300, 100), (400, 100)), ((300, 300), (400, 300)))
    test_file = "test_routes.json"
    ConfigManager.save_routes([r1, r2], test_file)
    assert os.path.exists(test_file), "Arquivo de rotas não foi criado!"

    loaded = ConfigManager.load_routes(test_file)
    assert len(loaded) == 2, f"Esperado 2 rotas carregadas, obteve {len(loaded)}"
    assert loaded[0].name == "Teste 1", "Nome da rota incorreto"
    if os.path.exists(test_file):
        os.remove(test_file)
    print(" -> ConfigManager OK!")


def test_route_crossing_logic():
    print("[TEST] Testando Lógica de Cruzamento de Portais (Gate A -> Gate B)...")
    engine = RouteEngine()
    # Rota com Linha A em y=100 e Linha B em y=200
    r = Route(1, "Rota Vertical", ((50, 100), (150, 100)), ((50, 200), (150, 200)))
    engine.add_route(r)

    # Simula veículo ID 1 descendo verticalmente de y=80 até y=220
    # Frame 1: acima de A
    v1_f1 = [{
        "id": 1,
        "bbox": (90, 70, 110, 90),
        "ground_point": (100, 90),
        "prev_point": (100, 80),
        "category": "Carro",
        "confidence": 0.9,
        "trail": [(100, 80), (100, 90)]
    }]
    events_1 = engine.update(v1_f1, frame_idx=1)
    assert len(events_1) == 0, "Não deveria contar no frame 1"
    assert 1 not in r.vehicle_states, "Não deveria entrar no frame 1"

    # Frame 2: cruza Linha A (de y=90 para y=110)
    v1_f2 = [{
        "id": 1,
        "bbox": (90, 90, 110, 110),
        "ground_point": (100, 110),
        "prev_point": (100, 90),
        "category": "Carro",
        "confidence": 0.9,
        "trail": [(100, 80), (100, 90), (100, 110)]
    }]
    events_2 = engine.update(v1_f2, frame_idx=2)
    assert len(events_2) == 0, "Ainda não concluiu a rota"
    assert 1 in r.vehicle_states, "Deveria ter registrado estado ENTERED"
    assert r.vehicle_states[1]["status"] == "ENTERED"

    # Frame 3: veículo passa por trás de um poste (simulado entre 110 e 190)
    v1_f3 = [{
        "id": 1,
        "bbox": (90, 170, 110, 190),
        "ground_point": (100, 190),
        "prev_point": (100, 110),
        "category": "Carro",
        "confidence": 0.85,
        "trail": [(100, 80), (100, 90), (100, 110), (100, 190)]
    }]
    events_3 = engine.update(v1_f3, frame_idx=10)
    assert len(events_3) == 0, "Ainda não cruzou B"

    # Frame 4: cruza Linha B (de y=190 para y=210)
    v1_f4 = [{
        "id": 1,
        "bbox": (90, 190, 110, 210),
        "ground_point": (100, 210),
        "prev_point": (100, 190),
        "category": "Carro",
        "confidence": 0.92,
        "trail": [(100, 110), (100, 190), (100, 210)]
    }]
    events_4 = engine.update(v1_f4, frame_idx=15)
    assert len(events_4) == 1, "Deveria ter contabilizado o evento de conclusão da rota!"
    assert r.counts["Carro"] == 1, f"Contagem de Carro deveria ser 1, obtido {r.counts['Carro']}"
    assert engine.total_counts["Carro"] == 1, "Total geral deveria ser 1"

    # Frame 5: veículo continua se movendo após a Linha B (não pode duplicar a contagem)
    v1_f5 = [{
        "id": 1,
        "bbox": (90, 210, 110, 230),
        "ground_point": (100, 230),
        "prev_point": (100, 210),
        "category": "Carro",
        "confidence": 0.95,
        "trail": [(100, 190), (100, 210), (100, 230)]
    }]
    events_5 = engine.update(v1_f5, frame_idx=16)
    assert len(events_5) == 0, "Não pode haver contagem duplicada do mesmo veículo!"
    assert r.counts["Carro"] == 1, "Contagem não deve ter aumentado"
    print(" -> Lógica de Cruzamento OK!")


def test_video_dav_processing():
    print("[TEST] Testando processamento real em 'video.dav'...")
    assert os.path.exists("video.dav"), "Arquivo video.dav não encontrado!"
    cap = cv2.VideoCapture("video.dav")
    assert cap.isOpened(), "Falha ao abrir video.dav!"

    detector = VehicleDetectorTracker(model_path="yolo11n.pt", tracker_cfg="custom_tracker.yaml")
    engine = RouteEngine()

    # Rota pré-calibrada
    r1 = Route(
        1, "Fundo -> Direita Viaduto",
        gate_a=((420, 440), (520, 460)),
        gate_b=((720, 560), (980, 680))
    )
    engine.add_route(r1)

    processed_frames = 0
    all_tracked = []

    for i in range(40):
        ret, frame = cap.read()
        if not ret:
            break
        tracked = detector.process_frame(frame)
        engine.update(tracked, i + 1)
        if tracked:
            all_tracked.extend(tracked)
        processed_frames += 1

    cap.release()
    print(f" -> Processou {processed_frames} quadros com sucesso!")
    print(f" -> Veículos rastreados detectados: {len(all_tracked)} instâncias.")
    print(" -> Resumo atual:\n" + engine.get_summary_text())


if __name__ == "__main__":
    test_config_manager()
    test_route_crossing_logic()
    test_video_dav_processing()
    print("\nTODOS OS TESTES FORAM APROVADOS COM SUCESSO!")
