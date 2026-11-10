import os
import cv2
from core.detector_tracker import VehicleDetectorTracker
from core.route_engine import Route, RouteEngine, vehicle_in_polygon, build_corridor_polygon
from core.config_manager import ConfigManager


def test_config_manager():
    print("[TEST] Testando ConfigManager...")
    r1 = Route(
        route_id=1,
        name="Teste 1",
        polygon_a=[(100, 100), (200, 100), (200, 200), (100, 200)],
        polygon_b=[(100, 300), (200, 300), (200, 400), (100, 400)]
    )
    r2 = Route(
        route_id=2,
        name="Teste 2",
        polygon_a=[(300, 100), (400, 100), (400, 200), (300, 200)],
        polygon_b=[(300, 300), (400, 300), (400, 400), (300, 400)]
    )
    test_file = "test_routes.json"
    ConfigManager.save_routes([r1, r2], test_file)
    assert os.path.exists(test_file), "Arquivo de rotas não foi criado!"

    loaded = ConfigManager.load_routes(test_file)
    assert len(loaded) == 2, f"Esperado 2 rotas carregadas, obteve {len(loaded)}"
    assert loaded[0].name == "Teste 1", "Nome da rota incorreto"
    assert loaded[0].polygon_a is not None, "Polígono A não foi carregado corretamente"
    if os.path.exists(test_file):
        os.remove(test_file)
    print(" -> ConfigManager OK!")


def test_polygon_route_logic():
    print("[TEST] Testando Lógica de Rotas por Polígonos Livres (Origem Vermelha -> Destino Amarelo)...")
    engine = RouteEngine()

    # Rota: Polígono A (Entrada) no topo, Polígono B (Saída) no rodapé
    poly_a = [(50, 50), (150, 50), (150, 120), (50, 120)]
    poly_b = [(50, 250), (150, 250), (150, 320), (50, 320)]

    r = Route(1, "Rota Poligonal", polygon_a=poly_a, polygon_b=poly_b)
    engine.add_route(r)

    # 1. Simula Carro 10 que entra na Origem A
    v1_f1 = [{
        "id": 10,
        "bbox": (80, 60, 120, 100),
        "prev_bbox": (80, 60, 120, 100),
        "center": (100, 80),
        "ground_point": (100, 100),
        "prev_point": (100, 80),
        "category": "Carro",
        "confidence": 0.95,
        "trail": [(100, 80), (100, 100)]
    }]
    events_1 = engine.update(v1_f1, frame_idx=1)
    assert len(events_1) == 0, "Carro acabou de entrar na Origem, não deve contar ainda"
    assert 10 in r.vehicle_states, "Carro 10 deveria estar registrado em vehicle_states"

    # 2. Simula Carro 99 (tráfego cruzado de outra rua) que surge direto no Destino B sem nunca passar por A!
    v_cross = [{
        "id": 99,
        "bbox": (80, 260, 120, 300),
        "prev_bbox": (80, 260, 120, 300),
        "center": (100, 280),
        "ground_point": (100, 300),
        "prev_point": (100, 280),
        "category": "Moto",
        "confidence": 0.92,
        "trail": [(250, 280), (100, 280)]
    }]
    events_cross = engine.update(v_cross, frame_idx=5)
    assert len(events_cross) == 0, "Veículo de tráfego cruzado NUNCA deve ser contado!"
    assert 99 not in r.vehicle_states, "Veículo de outra rua não deve entrar no estado da rota"

    # 3. Carro 10 avança pelo meio da rua
    v1_f2 = [{
        "id": 10,
        "bbox": (80, 150, 120, 190),
        "prev_bbox": (80, 60, 120, 100),
        "center": (100, 170),
        "ground_point": (100, 190),
        "prev_point": (100, 100),
        "category": "Carro",
        "confidence": 0.95,
        "trail": [(100, 80), (100, 100), (100, 190)]
    }]
    events_2 = engine.update(v1_f2, frame_idx=10)
    assert len(events_2) == 0, "Carro no meio do percurso não deve contar ainda"

    # 4. Carro 10 atinge a Área de Destino B!
    v1_f3 = [{
        "id": 10,
        "bbox": (80, 260, 120, 300),
        "prev_bbox": (80, 150, 120, 190),
        "center": (100, 280),
        "ground_point": (100, 300),
        "prev_point": (100, 190),
        "category": "Carro",
        "confidence": 0.95,
        "trail": [(100, 80), (100, 100), (100, 190), (100, 300)]
    }]
    events_3 = engine.update(v1_f3, frame_idx=15)
    assert len(events_3) == 1, "Carro 10 deveria ser contabilizado com sucesso ao atingir B!"
    assert r.counts["Carro"] == 1, "Contagem de Carro na rota deve ser 1"
    assert engine.total_counts["Carro"] == 1, "Total geral de Carros deve ser 1"

    # 5. Carro 10 continua andando após B (não pode duplicar!)
    v1_f4 = [{
        "id": 10,
        "bbox": (80, 330, 120, 370),
        "prev_bbox": (80, 260, 120, 300),
        "center": (100, 350),
        "ground_point": (100, 370),
        "prev_point": (100, 300),
        "category": "Carro",
        "confidence": 0.95,
        "trail": [(100, 190), (100, 300), (100, 370)]
    }]
    events_4 = engine.update(v1_f4, frame_idx=20)
    assert len(events_4) == 0, "Carro 10 não pode ser contado novamente!"
    assert r.counts["Carro"] == 1, "Contagem não deve mudar"

    print(" -> Lógica de Rotas por Polígonos Livres OK (Rejeição de outras ruas e contagem precisa comprovadas)!")


def test_single_directional_path():
    print("[TEST] Testando Percurso Único Direcional (Início ➔ Fim)...")
    engine = RouteEngine()
    r = Route(
        route_id=1,
        name="Percurso Único",
        path_points=[(100, 300), (300, 450), (500, 600)],
        corridor_width=1.0,
    )
    engine.add_route(r)

    # 1. Veículo na contramão (Fim -> Início: de 500,600 para 100,300) NUNCA deve ser contado!
    for idx, (px, py) in enumerate([(490, 590), (300, 450), (110, 310)]):
        evs = engine.update([{
            "id": 77,
            "bbox": (px - 20, py - 20, px + 20, py + 20),
            "prev_bbox": (px - 20, py - 20, px + 20, py + 20),
            "center": (px, py),
            "ground_point": (px, py),
            "prev_point": (px, py),
            "category": "Carro",
            "confidence": 0.92,
            "trail": [(490, 590), (px, py)],
        }], frame_idx=idx + 1)
        assert len(evs) == 0, "Veículo indo de B para A (contramão) jamais pode ser contado!"

    # 2. Veículo no sentido correto (Início -> Fim: de 100,300 para 500,600) DEVE ser contado!
    trail_ok = []
    counted = 0
    for idx, (px, py) in enumerate([(110, 310), (250, 410), (380, 510), (480, 585)]):
        trail_ok.append((px, py))
        evs = engine.update([{
            "id": 88,
            "bbox": (px - 20, py - 20, px + 20, py + 20),
            "prev_bbox": (px - 20, py - 20, px + 20, py + 20),
            "center": (px, py),
            "ground_point": (px, py),
            "prev_point": trail_ok[-2] if len(trail_ok) > 1 else (px, py),
            "category": "Moto",
            "confidence": 0.94,
            "trail": list(trail_ok),
        }], frame_idx=10 + idx)
        counted += len(evs)

    # 3. Veículo da lateral cortando quase no ponto final (NUNCA DEVE CONTAR!)
    # Ponto final está em (500, 600). Caminhão corta transversalmente de (600, 520) para (420, 640)
    for idx, (px, py) in enumerate([(580, 530), (530, 570), (480, 600), (440, 630)]):
        evs = engine.update([{
            "id": 99,
            "bbox": (px - 30, py - 30, px + 30, py + 30),
            "prev_bbox": (px - 30, py - 30, px + 30, py + 30),
            "center": (px, py),
            "ground_point": (px, py),
            "prev_point": (px, py),
            "category": "Caminhão",
            "confidence": 0.95,
            "trail": [(580, 530), (px, py)],
        }], frame_idx=25 + idx)
        assert len(evs) == 0, "Caminhão vindo da lateral cortando o final NÃO pode ser contabilizado!"

    # 4. Veículo que surge no meio do percurso (s = 0.60) sem vir do início (NUNCA DEVE CONTAR!)
    for idx, (px, py) in enumerate([(340, 480), (400, 525), (480, 585)]):
        evs = engine.update([{
            "id": 105,
            "bbox": (px - 20, py - 20, px + 20, py + 20),
            "prev_bbox": (px - 20, py - 20, px + 20, py + 20),
            "center": (px, py),
            "ground_point": (px, py),
            "prev_point": (px, py),
            "category": "Carro",
            "confidence": 0.93,
            "trail": [(340, 480), (px, py)],
        }], frame_idx=35 + idx)
        assert len(evs) == 0, "Veículo que entrou pelo meio da rota sem vir da origem NÃO pode ser contado!"

    assert r.counts["Moto"] == 1, "Apenas a Moto legítima deve ter sido contada!"
    assert r.counts["Caminhão"] == 0, "Caminhão lateral não deve ser contado!"
    assert r.counts["Carro"] == 0, "Carro que entrou no meio não deve ser contado!"
    print(" -> Percurso Único Direcional OK (Sentido A -> B contado, contramão bloqueada e tráfego lateral 100% rejeitado)!")


def test_timeline_rollback():
    print("[TEST] Testando Mecanismo de Rollback / Dedução Temporal no Timeline...")
    engine = RouteEngine()
    poly_a = [(50, 50), (150, 50), (150, 120), (50, 120)]
    poly_b = [(50, 250), (150, 250), (150, 320), (50, 320)]
    r = Route(1, "Rota Timeline", polygon_a=poly_a, polygon_b=poly_b)
    engine.add_route(r)

    # 1. Carro 101 entra em A (frame 10) e conclui em B (frame 30)
    engine.update([{
        "id": 101, "bbox": (80, 60, 120, 100), "prev_bbox": (80, 60, 120, 100),
        "center": (100, 80), "ground_point": (100, 100), "prev_point": (100, 80),
        "category": "Carro", "confidence": 0.95, "trail": [(100, 80), (100, 100)]
    }], frame_idx=10)

    engine.update([{
        "id": 101, "bbox": (80, 260, 120, 300), "prev_bbox": (80, 150, 120, 190),
        "center": (100, 280), "ground_point": (100, 300), "prev_point": (100, 190),
        "category": "Carro", "confidence": 0.95, "trail": [(100, 80), (100, 100), (100, 300)]
    }], frame_idx=30)
    assert r.counts["Carro"] == 1 and engine.total_counts["Carro"] == 1, "Carro 101 deveria estar contado"

    # 2. Moto 202 entra em A (frame 40) e conclui em B (frame 60)
    engine.update([{
        "id": 202, "bbox": (80, 60, 100, 90), "prev_bbox": (80, 60, 100, 90),
        "center": (90, 75), "ground_point": (90, 90), "prev_point": (90, 75),
        "category": "Moto", "confidence": 0.90, "trail": [(90, 75), (90, 90)]
    }], frame_idx=40)

    engine.update([{
        "id": 202, "bbox": (80, 260, 100, 290), "prev_bbox": (80, 150, 100, 180),
        "center": (90, 275), "ground_point": (90, 290), "prev_point": (90, 180),
        "category": "Moto", "confidence": 0.90, "trail": [(90, 75), (90, 90), (90, 290)]
    }], frame_idx=60)
    assert r.counts["Moto"] == 1 and engine.total_counts["Moto"] == 1, "Moto 202 deveria estar contada"
    assert sum(engine.total_counts.values()) == 2, "Total deve ser 2 (1 Carro + 1 Moto)"

    # 3. Caminhão 303 entra em A (frame 70) e conclui em B (frame 90)
    engine.update([{
        "id": 303, "bbox": (70, 50, 130, 110), "prev_bbox": (70, 50, 130, 110),
        "center": (100, 80), "ground_point": (100, 110), "prev_point": (100, 80),
        "category": "Caminhão", "confidence": 0.96, "trail": [(100, 80), (100, 110)]
    }], frame_idx=70)

    engine.update([{
        "id": 303, "bbox": (70, 250, 130, 310), "prev_bbox": (70, 150, 130, 210),
        "center": (100, 280), "ground_point": (100, 310), "prev_point": (100, 210),
        "category": "Caminhão", "confidence": 0.96, "trail": [(100, 80), (100, 110), (100, 310)]
    }], frame_idx=90)
    assert sum(engine.total_counts.values()) == 3, "Total deve ser 3 (1 Carro, 1 Moto, 1 Caminhão)"

    # --- TESTE DE ROLLBACK: Retrocede o timeline para frame 75 ---
    # Deve descontar o Caminhão 303 (contado em frame 90) e manter Carro 101 e Moto 202
    deducted = engine.rollback_to_frame(75)
    assert deducted == 1, f"Esperado 1 veículo descontado no rollback para frame 75, obteve {deducted}"
    assert r.counts["Caminhão"] == 0, "Caminhão não deve mais estar contado"
    assert r.counts["Carro"] == 1 and r.counts["Moto"] == 1, "Carro e Moto continuam válidos"
    assert sum(engine.total_counts.values()) == 2, "Total de veículos deve ser 2 após dedução"
    assert 303 not in r.completed_ids, "ID 303 do caminhão deve ser liberado de completed_ids"

    # --- TESTE DE RE-CONTAGEM: Quando o vídeo avança novamente pelo frame 90, o caminhão deve ser contado! ---
    engine.update([{
        "id": 303, "bbox": (70, 250, 130, 310), "prev_bbox": (70, 150, 130, 210),
        "center": (100, 280), "ground_point": (100, 310), "prev_point": (100, 210),
        "category": "Caminhão", "confidence": 0.96, "trail": [(100, 80), (100, 110), (100, 310)]
    }], frame_idx=90)
    assert r.counts["Caminhão"] == 1, "Caminhão deve ser contabilizado novamente após novo avanço!"
    assert sum(engine.total_counts.values()) == 3, "Total voltou para 3!"

    # --- TESTE DE ROLLBACK PROFUNDO: Retrocede para frame 20 ---
    # Deve descontar a Moto (frame 60) e o Caminhão (frame 90), restando 0 veículos (Carro foi frame 30)
    deducted_2 = engine.rollback_to_frame(20)
    assert deducted_2 == 3, f"Esperado descontar os 3 veículos, descontou {deducted_2}"
    assert sum(engine.total_counts.values()) == 0, "Total deve ser 0"
    assert len(r.completed_ids) == 0, "Todos os IDs devem estar livres"

    print(" -> Rollback Temporal e Dedução de Veículos APROVADOS COM SUCESSO!")


def test_per_point_widths():
    print("[TEST] Testando calibração de larguras individuais por ponto (point_widths)...")
    pts = [(100, 200), (200, 300), (400, 500)]
    widths = [0.50, 1.00, 1.80]
    r = Route(1, "Rota com Larguras Individuais", path_points=pts, point_widths=widths)
    assert len(r.point_widths) == 3
    assert r.point_widths[0] == 0.50
    assert r.point_widths[1] == 1.00
    assert r.point_widths[2] == 1.80

    # Verifica interpolação ao longo do percurso
    assert abs(r.get_width_at_s(0.0) - 0.50) < 1e-3
    assert abs(r.get_width_at_s(1.0) - 1.80) < 1e-3

    # Testa modificação de ponto individual
    r.set_point_width(0, 0.75)
    assert r.point_widths[0] == 0.75
    assert abs(r.get_width_at_s(0.0) - 0.75) < 1e-3

    # Testa serialização / desserialização
    d = r.to_dict()
    assert "point_widths" in d
    assert d["point_widths"] == [0.75, 1.0, 1.8]
    r_loaded = Route.from_dict(d)
    assert r_loaded.point_widths == [0.75, 1.0, 1.8]

    # Verifica construção do polígono de corredor com larguras variáveis
    poly = build_corridor_polygon(pts, point_widths=r.point_widths)
    assert len(poly) >= 6, "Polígono de corredor variável deve conter vértices"
    print(" -> Larguras individuais por ponto (point_widths) APROVADAS COM SUCESSO!")


def test_video_dav_processing():
    print("[TEST] Testando processamento real em 'video.dav' com a calibração de rotas...")
    assert os.path.exists("video.dav"), "Arquivo video.dav não encontrado!"
    cap = cv2.VideoCapture("video.dav")
    assert cap.isOpened(), "Falha ao abrir video.dav!"

    detector = VehicleDetectorTracker(model_path="yolo11n.pt", tracker_cfg="custom_tracker.yaml")
    engine = RouteEngine()

    routes = ConfigManager.load_routes("routes_config.json")
    for r in routes:
        engine.add_route(r)

    processed_frames = 0
    all_tracked = []
    events_captured = []

    # Inicia no frame 1210 para capturar o fluxo de veículos vindo da Origem até o Destino
    start_f = 1210
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    for i in range(110):
        ret, frame = cap.read()
        if not ret:
            break
        curr_frame = start_f + i
        tracked = detector.process_frame(frame)
        evs = engine.update(tracked, curr_frame)
        if evs:
            events_captured.extend(evs)
        if tracked:
            all_tracked.extend(tracked)
        processed_frames += 1

    cap.release()
    print(f" -> Processou {processed_frames} quadros com sucesso!")
    print(f" -> Veículos rastreados detectados: {len(all_tracked)} instâncias.")
    print(f" -> Eventos de contagem registrados: {len(events_captured)}")
    print(" -> Resumo atual:\n" + engine.get_summary_text())

    assert len(events_captured) >= 1, "Veículos legítimos da Rota 1 deveriam ter sido contabilizados!"
    print(" -> Validação em video.dav APROVADA COM SUCESSO!")


if __name__ == "__main__":
    test_config_manager()
    test_polygon_route_logic()
    test_single_directional_path()
    test_timeline_rollback()
    test_per_point_widths()
    test_video_dav_processing()
    print("\nTODOS OS TESTES FORAM APROVADOS COM SUCESSO!")
