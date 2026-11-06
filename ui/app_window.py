import os
import cv2
import time
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk

from core.detector_tracker import VehicleDetectorTracker, CATEGORY_COLORS
from core.route_engine import Route, RouteEngine
from core.config_manager import ConfigManager

ROUTE_PALETTE = [
    (0, 255, 255),   # Amarelo/Ciano
    (255, 0, 128),   # Roxo/Rosa
    (0, 200, 255),   # Laranja claro
    (255, 128, 0),   # Azul
    (128, 255, 0),   # Verde lima
]


class AppWindow:
    def __init__(self, root, video_path="video.dav"):
        self.root = root
        self.root.title("Sistema Inteligente de Contagem de Veículos por Rota")
        self.root.geometry("1420x860")
        self.root.minsize(1100, 700)
        self.root.configure(bg="#1e1e24")

        self.video_path = video_path

        # Inicializa motores
        self.detector = VehicleDetectorTracker(model_path="yolo11s.pt", tracker_cfg="custom_tracker.yaml")
        self.route_engine = RouteEngine()

        # Carrega rotas salvas
        loaded_routes = ConfigManager.load_routes()
        for r in loaded_routes:
            self.route_engine.add_route(r)

        # Estado da reprodução
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Erro de Leitura", f"Não foi possível abrir o arquivo de vídeo: {self.video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        self.current_frame_idx = 0
        self.is_playing = False
        self.play_speed = 1.0  # 1.0 = normal, 2.0 = rapido, 0 = sem delay (max)

        # Estado do modo de desenho de rotas
        self.drawing_mode = False
        self.draw_step = 0  # 0: idle, 1: Linha A pto 1, 2: Linha A pto 2, 3: Linha B pto 1, 4: Linha B pto 2
        self.temp_points = []
        self.temp_route_name = ""

        # Armazenamento do frame atual em RGB para redesenho
        self.last_frame_bgr = None
        self.canvas_scale = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0

        # Fila de comunicação entre a thread de processamento e a UI
        self.frame_queue = queue.Queue(maxsize=3)
        self.worker_thread = None
        self.stop_worker = threading.Event()

        self._build_ui()
        self._load_first_frame()

        # Inicia loop de leitura de frames da fila
        self.root.after(20, self._check_frame_queue)

    def _build_ui(self):
        # Layout principal com 2 painéis: Esquerda (Vídeo + Controles) e Direita (Rotas + Contagens)
        self.main_container = tk.Frame(self.root, bg="#1e1e24")
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Painel Esquerdo
        self.left_panel = tk.Frame(self.main_container, bg="#1e1e24")
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Canvas para exibição do vídeo
        self.canvas_frame = tk.Frame(self.left_panel, bg="#111116", relief=tk.RIDGE, bd=2)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#0d0d11", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Motion>", self._on_canvas_motion)

        # Faixa de Instrução de Desenho (visível apenas ao desenhar)
        self.draw_banner = tk.Label(
            self.left_panel,
            text="",
            bg="#ffaa00",
            fg="#000000",
            font=("Segoe UI", 10, "bold"),
            pady=4
        )

        # Barra de Controles Inferior
        self.ctrl_frame = tk.Frame(self.left_panel, bg="#272730", pady=8, padx=10)
        self.ctrl_frame.pack(fill=tk.X, pady=(8, 0))

        # Botão Play / Pause
        self.btn_play = tk.Button(
            self.ctrl_frame,
            text="▶ Iniciar Análise",
            command=self._toggle_play,
            bg="#00b4d8",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            padx=14,
            pady=4,
            cursor="hand2"
        )
        self.btn_play.pack(side=tk.LEFT, padx=6)

        # Botão Reiniciar
        self.btn_reset = tk.Button(
            self.ctrl_frame,
            text="↺ Reiniciar Contagens",
            command=self._reset_counters,
            bg="#4a4e69",
            fg="#ffffff",
            font=("Segoe UI", 9),
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2"
        )
        self.btn_reset.pack(side=tk.LEFT, padx=6)

        # Seletor de Velocidade
        tk.Label(self.ctrl_frame, text="Velocidade:", bg="#272730", fg="#dddddd", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(12, 4))
        self.speed_var = tk.StringVar(value="1.0x")
        self.speed_combo = ttk.Combobox(
            self.ctrl_frame,
            textvariable=self.speed_var,
            values=["0.5x", "1.0x", "2.0x", "4.0x", "8.0x", "10.0x", "20.0x", "Máxima (Turbo)"],
            state="readonly",
            width=14
        )
        self.speed_combo.pack(side=tk.LEFT, padx=4)
        self.speed_combo.bind("<<ComboboxSelected>>", self._on_speed_change)

        # Status do Frame / Tempo
        self.lbl_time = tk.Label(
            self.ctrl_frame,
            text="Frame: 0 / 0 | 00:00 / 00:00",
            bg="#272730",
            fg="#a0a0b0",
            font=("Segoe UI", 9)
        )
        self.lbl_time.pack(side=tk.RIGHT, padx=8)

        # Painel Direito (Estatísticas e Configuração de Rotas)
        self.right_panel = tk.Frame(self.main_container, bg="#272730", width=420, padx=12, pady=12)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))
        self.right_panel.pack_propagate(False)

        self._build_sidebar()

    def _build_sidebar(self):
        # Título da Seção de Rotas
        lbl_rotas_title = tk.Label(
            self.right_panel,
            text="GESTÃO DE ROTAS",
            bg="#272730",
            fg="#00b4d8",
            font=("Segoe UI", 11, "bold")
        )
        lbl_rotas_title.pack(anchor=tk.W, pady=(0, 6))

        # Botões de Rotas
        btn_box = tk.Frame(self.right_panel, bg="#272730")
        btn_box.pack(fill=tk.X, pady=(0, 8))

        self.btn_new_route = tk.Button(
            btn_box,
            text="+ Desenhar Nova Rota",
            command=self._start_drawing_route,
            bg="#2a9d8f",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            pady=4,
            cursor="hand2"
        )
        self.btn_new_route.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_preset_routes = tk.Button(
            btn_box,
            text="Rotas Padrão",
            command=self._load_preset_routes,
            bg="#3a5a40",
            fg="#ffffff",
            font=("Segoe UI", 9),
            relief=tk.FLAT,
            pady=4,
            cursor="hand2"
        )
        self.btn_preset_routes.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        self.btn_clear_routes = tk.Button(
            btn_box,
            text="Limpar",
            command=self._clear_all_routes,
            bg="#e76f51",
            fg="#ffffff",
            font=("Segoe UI", 9),
            relief=tk.FLAT,
            pady=4,
            cursor="hand2"
        )
        self.btn_clear_routes.pack(side=tk.LEFT, padx=(4, 0))

        # Separador
        ttk.Separator(self.right_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # Título da Seção de Contagens
        lbl_count_title = tk.Label(
            self.right_panel,
            text="CONTAGEM EM TEMPO REAL",
            bg="#272730",
            fg="#00b4d8",
            font=("Segoe UI", 11, "bold")
        )
        lbl_count_title.pack(anchor=tk.W, pady=(0, 6))

        # Container scrollável para os cards de rotas
        self.cards_container = tk.Frame(self.right_panel, bg="#272730")
        self.cards_container.pack(fill=tk.BOTH, expand=True, pady=4)

        # Card do Totalizador Geral (sempre no rodapé do painel lateral)
        self.total_card = tk.Frame(self.right_panel, bg="#111116", bd=2, relief=tk.GROOVE, padx=10, pady=10)
        self.total_card.pack(fill=tk.X, side=tk.BOTTOM, pady=(8, 0))

        self.lbl_total_title = tk.Label(
            self.total_card,
            text="TOTAL CONSOLIDADO",
            bg="#111116",
            fg="#f4a261",
            font=("Segoe UI", 11, "bold")
        )
        self.lbl_total_title.pack(anchor=tk.W)

        self.lbl_total_counts = tk.Label(
            self.total_card,
            text="Carros: 0  |  Caminhões: 0\nÔnibus: 0  |  Motos: 0\nTotal Geral: 0 veículos",
            bg="#111116",
            fg="#ffffff",
            font=("Segoe UI", 10),
            justify=tk.LEFT,
            pady=4
        )
        self.lbl_total_counts.pack(anchor=tk.W)

        self._refresh_route_cards()

    def _refresh_route_cards(self):
        """Reconstrói os cards visuais de contagem para cada rota existente."""
        for widget in self.cards_container.winfo_children():
            widget.destroy()

        if not self.route_engine.routes:
            lbl_empty = tk.Label(
                self.cards_container,
                text="Nenhuma rota configurada.\nClique em '+ Desenhar Nova Rota'\nou 'Rotas Padrão' para começar.",
                bg="#272730",
                fg="#888899",
                font=("Segoe UI", 10, "italic"),
                pady=20
            )
            lbl_empty.pack(fill=tk.BOTH, expand=True)
            self._update_total_card()
            return

        for route in self.route_engine.routes:
            card = tk.Frame(self.cards_container, bg="#1d1d24", bd=1, relief=tk.SOLID, padx=8, pady=8)
            card.pack(fill=tk.X, pady=4)

            # Cabeçalho da Rota com botão de exclusão
            header = tk.Frame(card, bg="#1d1d24")
            header.pack(fill=tk.X)

            lbl_name = tk.Label(
                header,
                text=route.name,
                bg="#1d1d24",
                fg="#e0e1dd",
                font=("Segoe UI", 10, "bold")
            )
            lbl_name.pack(side=tk.LEFT)

            btn_del = tk.Button(
                header,
                text="✕",
                command=lambda r_id=route.route_id: self._delete_route(r_id),
                bg="#1d1d24",
                fg="#e76f51",
                bd=0,
                font=("Segoe UI", 9, "bold"),
                cursor="hand2"
            )
            btn_del.pack(side=tk.RIGHT)

            c = route.counts
            subtotal = sum(c.values())
            lbl_counts = tk.Label(
                card,
                text=f"Carros: {c['Carro']}  |  Caminhão: {c['Caminhão']}\nÔnibus: {c['Ônibus']}  |  Moto: {c['Moto']}\nSubtotal: {subtotal}",
                bg="#1d1d24",
                fg="#a8dadc",
                font=("Segoe UI", 9),
                justify=tk.LEFT,
                pady=4
            )
            lbl_counts.pack(anchor=tk.W)

        self._update_total_card()

    def _update_total_card(self):
        t = self.route_engine.total_counts
        total_all = sum(t.values())
        self.lbl_total_counts.config(
            text=f"Carros: {t['Carro']}  |  Caminhões: {t['Caminhão']}\nÔnibus: {t['Ônibus']}  |  Motos: {t['Moto']}\nTotal Geral: {total_all} veículos"
        )

    def _load_first_frame(self):
        """Carrega e exibe o primeiro frame do vídeo com linhas de rota desenhadas."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if ret:
            self.last_frame_bgr = frame
            self.current_frame_idx = 0
            self._render_frame(frame, [])

    def _on_canvas_resize(self, event):
        """Redimensiona o frame proporcionalmente quando o canvas muda de tamanho."""
        if self.last_frame_bgr is not None:
            self._render_frame(self.last_frame_bgr, [])

    def _render_frame(self, frame_bgr, tracked_vehicles):
        """Renderiza o frame do OpenCV com todas as anotações visuais no Canvas do Tkinter."""
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 50 or canvas_h < 50:
            return

        display_frame = frame_bgr.copy()

        # 1. Desenha rotas configuradas
        for route in self.route_engine.routes:
            color = route.color
            # Linha de Entrada A (verde/ciano brilhante)
            p1_a, p2_a = route.gate_a
            cv2.line(display_frame, p1_a, p2_a, (0, 255, 0), 3)
            mid_a = ((p1_a[0] + p2_a[0]) // 2, (p1_a[1] + p2_a[1]) // 2)
            cv2.putText(display_frame, f"[A] {route.name}", (mid_a[0] - 40, mid_a[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

            # Linha de Saída B (se existir)
            if route.gate_b:
                p1_b, p2_b = route.gate_b
                cv2.line(display_frame, p1_b, p2_b, (0, 0, 255), 3)
                mid_b = ((p1_b[0] + p2_b[0]) // 2, (p1_b[1] + p2_b[1]) // 2)
                cv2.putText(display_frame, f"[B] {route.name}", (mid_b[0] - 40, mid_b[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

        # 2. Desenha pontos temporários do modo de desenho
        if self.drawing_mode:
            if len(self.temp_points) >= 1:
                cv2.circle(display_frame, self.temp_points[0], 6, (0, 255, 255), -1)
            if len(self.temp_points) >= 2:
                cv2.line(display_frame, self.temp_points[0], self.temp_points[1], (0, 255, 0), 3)
            if len(self.temp_points) >= 3:
                cv2.circle(display_frame, self.temp_points[2], 6, (0, 165, 255), -1)

        # 3. Desenha caixas, IDs e trilhas dos veículos rastreados
        for v in tracked_vehicles:
            x1, y1, x2, y2 = v["bbox"]
            cat = v["category"]
            tid = v["id"]
            color = CATEGORY_COLORS.get(cat, (0, 255, 0))

            # Caixa delimitadora
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            # Etiqueta com fundo sólido para legibilidade máxima
            label = f"#{tid} {cat}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display_frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(display_frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Rastro histórico
            trail = v["trail"]
            if len(trail) > 1:
                for i in range(1, len(trail)):
                    cv2.line(display_frame, trail[i - 1], trail[i], color, 2)

        # Redimensiona mantendo proporção de 16:9
        scale = min(canvas_w / self.video_width, canvas_h / self.video_height)
        new_w = int(self.video_width * scale)
        new_h = int(self.video_height * scale)
        self.canvas_scale = scale
        self.canvas_offset_x = (canvas_w - new_w) // 2
        self.canvas_offset_y = (canvas_h - new_h) // 2

        resized = cv2.resize(display_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        self.tk_img = ImageTk.PhotoImage(image=pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(
            self.canvas_offset_x,
            self.canvas_offset_y,
            anchor=tk.NW,
            image=self.tk_img
        )

        # Atualiza rodapé de tempo
        curr_sec = int(self.current_frame_idx / self.video_fps)
        total_sec = int(self.total_frames / self.video_fps)
        self.lbl_time.config(
            text=f"Frame: {self.current_frame_idx} / {self.total_frames} | {curr_sec // 60:02d}:{curr_sec % 60:02d} / {total_sec // 60:02d}:{total_sec % 60:02d}"
        )

    # ------------------ Processamento em Background ------------------

    def _toggle_play(self):
        if self.is_playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        if self.drawing_mode:
            self._cancel_drawing()

        self.is_playing = True
        self.btn_play.config(text="⏸ Pausar", bg="#e76f51")
        self.stop_worker.clear()

        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()

    def _pause(self):
        self.is_playing = False
        self.btn_play.config(text="▶ Iniciar Análise", bg="#00b4d8")
        self.stop_worker.set()

    def _reset_counters(self):
        self.route_engine.reset_counts()
        self.detector.reset()
        self._refresh_route_cards()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_idx = 0
        ret, frame = self.cap.read()
        if ret:
            self.last_frame_bgr = frame
            self._render_frame(frame, [])

    def _on_speed_change(self, event):
        val = self.speed_var.get()
        if "0.5x" in val:
            self.play_speed = 0.5
        elif "1.0x" in val:
            self.play_speed = 1.0
        elif "2.0x" in val:
            self.play_speed = 2.0
        elif "4.0x" in val:
            self.play_speed = 4.0
        elif "8.0x" in val:
            self.play_speed = 8.0
        elif "10.0x" in val:
            self.play_speed = 10.0
        elif "20.0x" in val:
            self.play_speed = 20.0
        elif "Turbo" in val:
            self.play_speed = 0.0

    def _worker_loop(self):
        """Thread que faz a leitura sequencial do vídeo e inferência no modelo."""
        while self.is_playing and not self.stop_worker.is_set():
            t_start = time.time()

            # Em velocidades elevadas (>= 4x), realiza avanço adaptativo de frames para atingir a taxa desejada
            frame_step = 1
            if self.play_speed >= 20.0:
                frame_step = 6
            elif self.play_speed >= 10.0:
                frame_step = 4
            elif self.play_speed >= 4.0:
                frame_step = 2

            if frame_step > 1:
                for _ in range(frame_step - 1):
                    if not self.cap.grab():
                        break
                    self.current_frame_idx += 1

            ret, frame = self.cap.read()
            if not ret:
                # Fim do vídeo
                self.is_playing = False
                self.root.after(0, lambda: self.btn_play.config(text="▶ Reiniciar Análise", bg="#00b4d8"))
                break

            self.current_frame_idx += 1

            # Rastreia veículos
            tracked = self.detector.process_frame(frame)

            # Atualiza rotas e contadores
            new_events = self.route_engine.update(tracked, self.current_frame_idx)

            # Envia frame e dados para UI
            if not self.frame_queue.full():
                self.frame_queue.put((frame, tracked, new_events, self.current_frame_idx))

            # Controle de taxa de quadros (respeitando velocidade selecionada)
            if self.play_speed > 0:
                target_frame_time = (float(frame_step) / self.video_fps) / self.play_speed
                elapsed = time.time() - t_start
                sleep_time = target_frame_time - elapsed
                if sleep_time > 0.002:
                    time.sleep(sleep_time)

    def _check_frame_queue(self):
        """Consome frames processados da fila para atualizar o canvas do Tkinter com máxima fluidez."""
        try:
            while not self.frame_queue.empty():
                frame, tracked, events, f_idx = self.frame_queue.get_nowait()
                self.last_frame_bgr = frame
                self.current_frame_idx = f_idx
                self._render_frame(frame, tracked)

                # Se houve contagem no frame, atualiza a barra lateral
                if events:
                    self._refresh_route_cards()
        except Exception:
            pass

        self.root.after(15, self._check_frame_queue)

    # ------------------ Modo Desenho Interativo de Rotas ------------------

    def _start_drawing_route(self):
        self._pause()
        name = simpledialog.askstring("Nova Rota", "Digite o nome da rota (ex: 'Rota 1 - Viaduto Direita'):", parent=self.root)
        if not name:
            return

        self.temp_route_name = name.strip()
        self.drawing_mode = True
        self.draw_step = 1
        self.temp_points = []

        self.draw_banner.pack(fill=tk.X, before=self.canvas_frame)
        self.draw_banner.config(text=f"Desenhando '{self.temp_route_name}': Clique no VÍDEO para marcar o 1º PONTO da LINHA DE ENTRADA (A).")

    def _cancel_drawing(self):
        self.drawing_mode = False
        self.draw_step = 0
        self.temp_points = []
        self.draw_banner.pack_forget()
        if self.last_frame_bgr is not None:
            self._render_frame(self.last_frame_bgr, [])

    def _canvas_to_video_coords(self, cx, cy):
        """Converte coordenadas de clique do Canvas para coordenadas reais da imagem (1280x720)."""
        if self.canvas_scale <= 0:
            return (0, 0)
        vx = int((cx - self.canvas_offset_x) / self.canvas_scale)
        vy = int((cy - self.canvas_offset_y) / self.canvas_scale)
        vx = max(0, min(self.video_width - 1, vx))
        vy = max(0, min(self.video_height - 1, vy))
        return (vx, vy)

    def _on_canvas_click(self, event):
        if not self.drawing_mode:
            return

        pt = self._canvas_to_video_coords(event.x, event.y)
        self.temp_points.append(pt)

        if self.draw_step == 1:
            self.draw_step = 2
            self.draw_banner.config(text=f"Desenhando '{self.temp_route_name}': Clique para marcar o 2º PONTO da LINHA DE ENTRADA (A).")
        elif self.draw_step == 2:
            self.draw_step = 3
            self.draw_banner.config(text=f"Desenhando '{self.temp_route_name}': Clique para marcar o 1º PONTO da LINHA DE SAÍDA (B).")
        elif self.draw_step == 3:
            self.draw_step = 4
            self.draw_banner.config(text=f"Desenhando '{self.temp_route_name}': Clique para marcar o 2º PONTO da LINHA DE SAÍDA (B).")
        elif self.draw_step == 4:
            # Rota completa com Portal A e Portal B
            gate_a = (self.temp_points[0], self.temp_points[1])
            gate_b = (self.temp_points[2], self.temp_points[3])
            r_id = len(self.route_engine.routes) + 1
            color = ROUTE_PALETTE[(r_id - 1) % len(ROUTE_PALETTE)]

            new_route = Route(
                route_id=r_id,
                name=self.temp_route_name,
                gate_a=gate_a,
                gate_b=gate_b,
                color=color
            )
            self.route_engine.add_route(new_route)
            ConfigManager.save_routes(self.route_engine.routes)

            self._cancel_drawing()
            self._refresh_route_cards()
            if self.last_frame_bgr is not None:
                self._render_frame(self.last_frame_bgr, [])
            messagebox.showinfo("Rota Adicionada", f"Rota '{new_route.name}' configurada com sucesso!")

    def _on_canvas_motion(self, event):
        pass

    def _delete_route(self, route_id):
        if messagebox.askyesno("Excluir Rota", "Tem certeza que deseja remover esta rota?"):
            self.route_engine.remove_route(route_id)
            ConfigManager.save_routes(self.route_engine.routes)
            self._refresh_route_cards()
            if self.last_frame_bgr is not None:
                self._render_frame(self.last_frame_bgr, [])

    def _clear_all_routes(self):
        if messagebox.askyesno("Limpar Rotas", "Deseja remover todas as rotas configuradas?"):
            self.route_engine.clear_routes()
            ConfigManager.save_routes(self.route_engine.routes)
            self._refresh_route_cards()
            if self.last_frame_bgr is not None:
                self._render_frame(self.last_frame_bgr, [])

    def _load_preset_routes(self):
        """Carrega as 2 rotas principais calibradas para este cenário de viaduto."""
        if messagebox.askyesno("Carregar Rotas Padrão", "Carregar as rotas pré-calibradas para este vídeo (Viaduto Direita e Frente)?"):
            self.route_engine.clear_routes()

            # Rota 1: Vindo do fundo e virando à direita sob o viaduto
            r1 = Route(
                route_id=1,
                name="Rota 1: Fundo -> Viaduto Direita",
                gate_a=((340, 420), (710, 470)),
                gate_b=((670, 520), (1279, 660)),
                color=(0, 255, 255)
            )

            # Rota 2: Vindo do fundo e seguindo em frente/contorno
            r2 = Route(
                route_id=2,
                name="Rota 2: Fundo -> Seguir Reto",
                gate_a=((340, 420), (710, 470)),
                gate_b=((180, 680), (660, 715)),
                color=(255, 0, 128)
            )

            self.route_engine.add_route(r1)
            self.route_engine.add_route(r2)
            ConfigManager.save_routes(self.route_engine.routes)

            self._refresh_route_cards()
            if self.last_frame_bgr is not None:
                self._render_frame(self.last_frame_bgr, [])
