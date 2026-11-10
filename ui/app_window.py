import math
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk

from core.detector_tracker import VehicleDetectorTracker, CATEGORY_COLORS
from core.route_engine import Route, RouteEngine, build_corridor_polygon, resample_polyline, project_point_to_path, perspective_half_width
from core.config_manager import ConfigManager

ROUTE_PALETTE = [
    (0, 255, 255),    # Amarelo
    (255, 160, 0),    # Azul Ciano
    (0, 255, 128),    # Verde Claro
    (255, 0, 255),    # Magenta
    (0, 165, 255),    # Laranja
    (180, 105, 255),  # Rosa
]


class AppWindow:
    def __init__(self, root, video_path="video.dav"):
        self.root = root
        self.root.title("Sistema de Contagem e Classificação de Veículos por Rota")
        self.root.geometry("1480x880")
        self.root.configure(bg="#1e1e24")

        self.video_path = video_path

        # Inicializa motores de rastreamento com YOLO11s acelerado por GPU DirectML em resolução HD nativa (1280x736)
        # YOLO11s roda a ~40 FPS reais na RTX 3060, garantindo fluidez absoluta sem nenhum lag
        self.detector = VehicleDetectorTracker(
            model_path="yolo11s.pt",
            tracker_cfg="custom_tracker.yaml",
            conf_thresh=0.06,
            imgsz=(736, 1280),
        )
        self.route_engine = RouteEngine()

        # Carrega rotas salvas
        loaded_routes = ConfigManager.load_routes()
        for r in loaded_routes:
            self.route_engine.add_route(r)

        # Rota ativa/selecionada para controle de espessura (scroll ou +/-)
        self.selected_route_id = loaded_routes[0].route_id if loaded_routes else None

        # Feedback visual flutuante (HUD) na tela do vídeo
        self.hud_message = ""
        self.hud_time = 0.0

        # Dicionários de widgets para atualização rápida sem recriação do painel
        self.scale_vars = {}
        self.scale_labels = {}

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
        self.play_speed = 1.0
        self.last_proc_time = time.time()
        self.proc_fps = 0.0

        # Estado do Timeline e Scrubbing
        self.is_scrubbing = False
        self.was_playing_before_scrub = False
        self._last_scrub_time = 0

        # Estado do modo de desenho de Percurso Único Direcional (Início ➔ Fim)
        self.drawing_mode = False
        self.temp_route_name = ""
        self.temp_corridor_width = 1.0
        self.current_draw_points = []
        self.current_draw_widths = []
        self.active_draw_width = 1.0
        self.hovered_draw_pt_idx = None
        self.hovered_vertex_info = None
        self.mouse_video_pos = None

        # Armazenamento do frame atual em BGR para redesenho
        self.last_frame_bgr = None
        self.canvas_scale = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0

        # Fila de comunicação desacoplada não-bloqueante
        self.frame_queue = queue.Queue(maxsize=2)
        self.worker_thread = None
        self.stop_worker = threading.Event()
        self.cap_lock = threading.Lock()

        self._build_ui()
        self._load_first_frame()

        # Inicia loop de leitura de frames da fila
        self.root.after(15, self._check_frame_queue)

    def _build_ui(self):
        # Layout principal com 2 painéis: Esquerda (Vídeo + Controles) e Direita (Rotas + Contagens)
        self.main_container = tk.Frame(self.root, bg="#1e1e24")
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Painel Esquerdo
        self.left_panel = tk.Frame(self.main_container, bg="#1e1e24")
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Faixa Interativa Superior de Desenho (visível ao desenhar percurso único)
        self.draw_banner_frame = tk.Frame(self.left_panel, bg="#141e28", pady=6, padx=12, relief=tk.SOLID, bd=1)

        self.draw_banner_lbl = tk.Label(
            self.draw_banner_frame,
            text="",
            bg="#141e28",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            justify=tk.LEFT
        )
        self.draw_banner_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.btn_finish_step = tk.Button(
            self.draw_banner_frame,
            text="✓ Concluir Percurso (Enter)",
            command=self._finish_current_draw_step,
            bg="#2a9d8f",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=12,
            pady=3,
            cursor="hand2"
        )
        self.btn_finish_step.pack(side=tk.RIGHT, padx=(8, 0))

        self.btn_cancel_step = tk.Button(
            self.draw_banner_frame,
            text="✖ Cancelar (Esc)",
            command=self._cancel_drawing,
            bg="#3a3a44",
            fg="#dddddd",
            font=("Segoe UI", 9),
            relief=tk.FLAT,
            padx=8,
            pady=3,
            cursor="hand2"
        )
        self.btn_cancel_step.pack(side=tk.RIGHT, padx=4)

        # Canvas para exibição do vídeo
        self.canvas_frame = tk.Frame(self.left_panel, bg="#111116", relief=tk.RIDGE, bd=2)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#0d0d11", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Eventos do Mouse e Teclado no Canvas
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Double-Button-1>", lambda e: self._finish_current_draw_step())
        self.canvas.bind("<Button-3>", lambda e: self._finish_current_draw_step())

        # Scroll do mouse no Canvas para ajustar a espessura da via (Windows e Linux)
        self.canvas.bind("<MouseWheel>", self._on_canvas_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._on_canvas_wheel_dir(+1, e))
        self.canvas.bind("<Button-5>", lambda e: self._on_canvas_wheel_dir(-1, e))

        # Teclas de atalho para aumentar (+) ou diminuir (-) a espessura da faixa
        for key in ["<plus>", "<KP_Add>", "<equal>", "<bracketright>"]:
            self.root.bind(key, lambda e: self._on_shortcut_adjust_width(+0.10))
        for key in ["<minus>", "<KP_Subtract>", "<underscore>", "<bracketleft>"]:
            self.root.bind(key, lambda e: self._on_shortcut_adjust_width(-0.10))

        # Listener global para caracteres '+' e '-' em qualquer teclado
        def _on_global_key(e):
            if self.drawing_mode and e.keysym in ("Return", "Escape"):
                return
            if e.char in ("+", "="):
                self._on_shortcut_adjust_width(+0.10)
            elif e.char in ("-", "_"):
                self._on_shortcut_adjust_width(-0.10)
            elif e.char in ("]", "}"):
                self._on_shortcut_adjust_width(+0.10)
            elif e.char in ("[", "{"):
                self._on_shortcut_adjust_width(-0.10)

        self.root.bind("<Key>", _on_global_key, add="+")

        self.root.bind("<Return>", lambda e: self._finish_current_draw_step() if self.drawing_mode else None)
        self.root.bind("<Escape>", lambda e: self._cancel_drawing() if self.drawing_mode else None)
        self.root.bind("<Left>", lambda e: self._jump_relative(-5) if not self.drawing_mode else None)
        self.root.bind("<Right>", lambda e: self._jump_relative(5) if not self.drawing_mode else None)
        self.root.bind("<space>", lambda e: self._toggle_play() if not self.drawing_mode else None)
        self.root.bind("<ButtonRelease-1>", self._on_global_release, add="+")

        # Barra da Linha do Tempo Interativa (Timeline / Scrubber com Rollback)
        self.timeline_frame = tk.Frame(self.left_panel, bg="#1a1a24", pady=6, padx=10, relief=tk.SOLID, bd=1)
        self.timeline_frame.pack(fill=tk.X, pady=(6, 0))

        # Linha 1 do Timeline: Slider / Régua de Tempo
        slider_row = tk.Frame(self.timeline_frame, bg="#1a1a24")
        slider_row.pack(fill=tk.X, pady=(2, 4))

        self.lbl_timeline_current = tk.Label(
            slider_row,
            text="⏱ 00:00",
            bg="#1a1a24",
            fg="#00b4d8",
            font=("Segoe UI", 10, "bold"),
            width=8,
            anchor=tk.W
        )
        self.lbl_timeline_current.pack(side=tk.LEFT, padx=(0, 6))

        self.timeline_var = tk.DoubleVar(value=0.0)
        self.timeline_slider = ttk.Scale(
            slider_row,
            from_=0,
            to=max(1, self.total_frames - 1),
            orient=tk.HORIZONTAL,
            variable=self.timeline_var
        )
        self.timeline_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        total_sec = int(self.total_frames / self.video_fps)
        total_time_str = f"{total_sec // 60:02d}:{total_sec % 60:02d}"
        self.lbl_timeline_total = tk.Label(
            slider_row,
            text=f"{total_time_str} ({self.total_frames} frames)",
            bg="#1a1a24",
            fg="#888899",
            font=("Segoe UI", 9),
            anchor=tk.E
        )
        self.lbl_timeline_total.pack(side=tk.RIGHT, padx=(6, 0))

        # Eventos do Slider para Scrubbing suave com pausa e busca direta
        self.timeline_slider.bind("<ButtonPress-1>", self._on_slider_press)
        self.timeline_slider.bind("<B1-Motion>", self._on_slider_motion)
        self.timeline_slider.bind("<ButtonRelease-1>", self._on_slider_release)

        # Linha 2 do Timeline: Botões de Salto Rápido e Feedback de Rollback
        nav_row = tk.Frame(self.timeline_frame, bg="#1a1a24")
        nav_row.pack(fill=tk.X, pady=(2, 2))

        btn_style_sec = {"bg": "#2d2d3a", "fg": "#e0e1dd", "font": ("Segoe UI", 8, "bold"), "relief": tk.FLAT, "padx": 6, "pady": 2, "cursor": "hand2"}
        btn_style_pri = {"bg": "#264653", "fg": "#2a9d8f", "font": ("Segoe UI", 8, "bold"), "relief": tk.FLAT, "padx": 8, "pady": 2, "cursor": "hand2"}

        self.btn_jump_m30 = tk.Button(nav_row, text="⏪ -30s", command=lambda: self._jump_relative(-30), **btn_style_sec)
        self.btn_jump_m30.pack(side=tk.LEFT, padx=2)

        self.btn_jump_m10 = tk.Button(nav_row, text="◀◀ -10s", command=lambda: self._jump_relative(-10), **btn_style_sec)
        self.btn_jump_m10.pack(side=tk.LEFT, padx=2)

        self.btn_jump_m5 = tk.Button(nav_row, text="◀ -5s", command=lambda: self._jump_relative(-5), **btn_style_pri)
        self.btn_jump_m5.pack(side=tk.LEFT, padx=2)

        self.btn_jump_m1 = tk.Button(nav_row, text="◂ -1s", command=lambda: self._jump_relative(-1), **btn_style_sec)
        self.btn_jump_m1.pack(side=tk.LEFT, padx=2)

        self.btn_jump_p1 = tk.Button(nav_row, text="+1s ▸", command=lambda: self._jump_relative(1), **btn_style_sec)
        self.btn_jump_p1.pack(side=tk.LEFT, padx=2)

        self.btn_jump_p5 = tk.Button(nav_row, text="+5s ▶", command=lambda: self._jump_relative(5), **btn_style_pri)
        self.btn_jump_p5.pack(side=tk.LEFT, padx=2)

        self.btn_jump_p10 = tk.Button(nav_row, text="+10s ▶▶", command=lambda: self._jump_relative(10), **btn_style_sec)
        self.btn_jump_p10.pack(side=tk.LEFT, padx=2)

        self.btn_jump_p30 = tk.Button(nav_row, text="+30s ⏩", command=lambda: self._jump_relative(30), **btn_style_sec)
        self.btn_jump_p30.pack(side=tk.LEFT, padx=2)

        self.lbl_rollback_notice = tk.Label(
            nav_row,
            text="Dica: Role a rodinha do mouse (scroll) ou aperte +/- para afinar ou alargar a via no vídeo.",
            bg="#1a1a24",
            fg="#72efdd",
            font=("Segoe UI", 8),
            anchor=tk.W
        )
        self.lbl_rollback_notice.pack(side=tk.LEFT, padx=(12, 0), fill=tk.X, expand=True)

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

        # Seletor de Modelo YOLO
        tk.Label(self.ctrl_frame, text="Modelo:", bg="#272730", fg="#dddddd", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(10, 4))
        self.model_var = tk.StringVar(value="YOLO11s (Recomendado - Fluido 40 FPS)")
        self.model_combo = ttk.Combobox(
            self.ctrl_frame,
            textvariable=self.model_var,
            values=[
                "YOLO11s (Recomendado - Fluido 40 FPS)",
                "YOLO11m (Alta Precisão - 24 FPS)",
                "YOLO11l (Máxima Precisão - 19 FPS)",
                "YOLO11n (Ultra Rápido - 70 FPS)",
            ],
            state="readonly",
            width=32
        )
        self.model_combo.pack(side=tk.LEFT, padx=4)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        # Seletor de Velocidade
        tk.Label(self.ctrl_frame, text="Velocidade:", bg="#272730", fg="#dddddd", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(12, 4))
        self.speed_var = tk.StringVar(value="1.0x")
        self.speed_combo = ttk.Combobox(
            self.ctrl_frame,
            textvariable=self.speed_var,
            values=[
                "0.5x", "1.0x", "2.0x", "3.0x", "4.0x", "8.0x",
                "10.0x", "20.0x", "30.0x", "50.0x", "75.0x", "100.0x", "Máxima (Turbo)"
            ],
            state="readonly",
            width=15
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
        lbl_rotas_title = tk.Label(
            self.right_panel,
            text="GESTÃO DE PERCURSOS (DIRECIONAIS)",
            bg="#272730",
            fg="#00b4d8",
            font=("Segoe UI", 11, "bold")
        )
        lbl_rotas_title.pack(anchor=tk.W, pady=(0, 6))

        # Botão ÚNICO de Novo Percurso em 1 Etapa
        self.btn_new_route = tk.Button(
            self.right_panel,
            text="★ + Novo Percurso (Início ➔ Fim)",
            command=self._start_drawing_route,
            bg="#0077b6",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            pady=7,
            cursor="hand2"
        )
        self.btn_new_route.pack(fill=tk.X, pady=(0, 6))

        # Linha de utilitários: Percursos Padrão e Limpar
        btn_box_sub = tk.Frame(self.right_panel, bg="#272730")
        btn_box_sub.pack(fill=tk.X, pady=(0, 8))

        self.btn_preset_routes = tk.Button(
            btn_box_sub,
            text="Percursos Padrão",
            command=self._load_preset_routes,
            bg="#3a5a40",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            pady=4,
            cursor="hand2"
        )
        self.btn_preset_routes.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_clear_routes = tk.Button(
            btn_box_sub,
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

        ttk.Separator(self.right_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        lbl_count_title = tk.Label(
            self.right_panel,
            text="CONTAGEM EM TEMPO REAL",
            bg="#272730",
            fg="#00b4d8",
            font=("Segoe UI", 11, "bold")
        )
        lbl_count_title.pack(anchor=tk.W, pady=(0, 6))

        self.cards_container = tk.Frame(self.right_panel, bg="#272730")
        self.cards_container.pack(fill=tk.BOTH, expand=True, pady=4)

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
            text="Motos: 0  |  Carros: 0\nCaminhões: 0  |  Ônibus: 0\nTotal Geral: 0 veículos",
            bg="#111116",
            fg="#ffffff",
            font=("Segoe UI", 10),
            justify=tk.LEFT,
            pady=4
        )
        self.lbl_total_counts.pack(anchor=tk.W)

        self._refresh_route_cards()

    def _refresh_route_cards(self):
        """Reconstrói os cards com controles interativos de espessura (slider + botões + scroll)."""
        for widget in self.cards_container.winfo_children():
            widget.destroy()

        self.scale_vars.clear()
        self.scale_labels.clear()

        if not self.route_engine.routes:
            lbl_empty = tk.Label(
                self.cards_container,
                text="Nenhum percurso configurado.\nClique em '★ + Novo Percurso (Início ➔ Fim)'\nou 'Percursos Padrão' para começar.",
                bg="#272730",
                fg="#888899",
                font=("Segoe UI", 10, "italic"),
                pady=20
            )
            lbl_empty.pack(fill=tk.BOTH, expand=True)
            self._update_total_card()
            return

        for route in self.route_engine.routes:
            is_selected = (route.route_id == self.selected_route_id)
            card_bd = 2 if is_selected else 1
            card_highlight = "#00b4d8" if is_selected else "#3a3a46"

            card = tk.Frame(self.cards_container, bg="#1d1d24", bd=card_bd, relief=tk.SOLID, padx=8, pady=6, highlightbackground=card_highlight, highlightthickness=card_bd)
            card.pack(fill=tk.X, pady=4)

            # Clicar no card seleciona a rota ativa
            card.bind("<Button-1>", lambda e, r=route: self._select_route(r.route_id))

            header = tk.Frame(card, bg="#1d1d24")
            header.pack(fill=tk.X)
            header.bind("<Button-1>", lambda e, r=route: self._select_route(r.route_id))

            # Indicador de cor da rota
            hex_color = f"#{route.color[2]:02x}{route.color[1]:02x}{route.color[0]:02x}"
            dot = tk.Label(header, text="●", fg=hex_color, bg="#1d1d24", font=("Segoe UI", 12, "bold"))
            dot.pack(side=tk.LEFT, padx=(0, 4))
            dot.bind("<Button-1>", lambda e, r=route: self._select_route(r.route_id))

            lbl_name = tk.Label(
                header,
                text=route.name + ("  [ATIVA]" if is_selected else ""),
                bg="#1d1d24",
                fg="#72efdd" if is_selected else "#e0e1dd",
                font=("Segoe UI", 10, "bold" if is_selected else "normal")
            )
            lbl_name.pack(side=tk.LEFT)
            lbl_name.bind("<Button-1>", lambda e, r=route: self._select_route(r.route_id))

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

            # Controles de Largura/Espessura da Faixa com Slider e Scroll
            if route.path_points:
                width_row = tk.Frame(card, bg="#1d1d24")
                width_row.pack(fill=tk.X, pady=(4, 2))

                tk.Label(
                    width_row,
                    text="Faixa:",
                    bg="#1d1d24",
                    fg="#a0a0b0",
                    font=("Segoe UI", 8, "bold")
                ).pack(side=tk.LEFT, padx=(0, 2))

                btn_w_minus = tk.Button(
                    width_row,
                    text="➖",
                    command=lambda r=route: self._adjust_route_width(r, -0.10),
                    bg="#2a2a35",
                    fg="#ffffff",
                    font=("Segoe UI", 8, "bold"),
                    relief=tk.FLAT,
                    padx=5,
                    pady=0,
                    cursor="hand2"
                )
                btn_w_minus.pack(side=tk.LEFT)

                # Slider Interativo de Arrasto
                w_var = tk.DoubleVar(value=route.corridor_width)
                self.scale_vars[route.route_id] = w_var
                scale_w = ttk.Scale(
                    width_row,
                    from_=0.20,
                    to=2.80,
                    orient=tk.HORIZONTAL,
                    variable=w_var,
                    command=lambda val, r=route: self._on_scale_route_width(r, val)
                )
                scale_w.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

                btn_w_plus = tk.Button(
                    width_row,
                    text="➕",
                    command=lambda r=route: self._adjust_route_width(r, +0.10),
                    bg="#2a2a35",
                    fg="#ffffff",
                    font=("Segoe UI", 8, "bold"),
                    relief=tk.FLAT,
                    padx=5,
                    pady=0,
                    cursor="hand2"
                )
                btn_w_plus.pack(side=tk.LEFT)

                lbl_w_val = tk.Label(
                    width_row,
                    text=f"{route.corridor_width:.2f}x",
                    bg="#1d1d24",
                    fg="#00b4d8",
                    font=("Segoe UI", 8, "bold"),
                    width=6
                )
                lbl_w_val.pack(side=tk.LEFT, padx=(2, 0))
                self.scale_labels[route.route_id] = lbl_w_val

                # Bind de scroll diretamente sobre o card do percurso
                card.bind("<MouseWheel>", lambda e, r=route: self._on_card_mouse_wheel(r, e))
                width_row.bind("<MouseWheel>", lambda e, r=route: self._on_card_mouse_wheel(r, e))
                scale_w.bind("<MouseWheel>", lambda e, r=route: self._on_card_mouse_wheel(r, e))

                tip_lbl = tk.Label(
                    card,
                    text="🖱️ Aponte na rota e gire o Scroll do mouse ou aperte +/-",
                    bg="#1d1d24",
                    fg="#778da9",
                    font=("Segoe UI", 7, "italic")
                )
                tip_lbl.pack(anchor=tk.W, pady=(0, 2))

            c = route.counts
            subtotal = sum(c.values())
            lbl_counts = tk.Label(
                card,
                text=f"Motos: {c['Moto']}  |  Carros: {c['Carro']}\nCaminhões: {c['Caminhão']}  |  Ônibus: {c['Ônibus']}\nSubtotal: {subtotal}",
                bg="#1d1d24",
                fg="#a8dadc",
                font=("Segoe UI", 9),
                justify=tk.LEFT,
                pady=1
            )
            lbl_counts.pack(anchor=tk.W)

        self._update_total_card()

    def _select_route(self, route_id):
        """Seleciona a rota ativa para ajuste visual e atualiza a interface."""
        if self.selected_route_id != route_id:
            self.selected_route_id = route_id
            for r in self.route_engine.routes:
                if r.route_id == route_id:
                    self.hud_message = f"🎯 [{r.name}] selecionada. Role o Scroll ↕ ou use +/- para ajustar a largura."
                    self.hud_time = time.time()
                    break
            self._refresh_route_cards()
            if self.last_frame_bgr is not None:
                self._render_frame(self.last_frame_bgr, [])

    def _on_card_mouse_wheel(self, route, event):
        """Ajusta a largura ao girar a rodinha do mouse sobre o card da rota."""
        delta = 0.10 if event.delta > 0 else -0.10
        self._adjust_route_width(route, delta)
        return "break"

    def _on_scale_route_width(self, route, val):
        """Chamado continuamente enquanto o usuário arrasta a barra deslizante (slider) da faixa."""
        new_w = max(0.20, min(2.80, round(float(val), 2)))
        route.set_corridor_width(new_w)
        if route.route_id in self.scale_labels:
            self.scale_labels[route.route_id].config(text=f"{new_w:.2f}x")
        self.selected_route_id = route.route_id
        self.hud_message = f"📏 [{route.name}] Largura: {new_w:.2f}x (Scroll ↕ ou +/-)"
        self.hud_time = time.time()
        ConfigManager.save_routes(self.route_engine.routes)
        if self.last_frame_bgr is not None:
            self._render_frame(self.last_frame_bgr, [])

    def _adjust_route_width(self, route, delta):
        """Ajusta a espessura da faixa da rota e atualiza a tela instantaneamente."""
        new_w = max(0.20, min(2.80, round(route.corridor_width + delta, 2)))
        route.set_corridor_width(new_w)
        ConfigManager.save_routes(self.route_engine.routes)
        self.selected_route_id = route.route_id
        self.hud_message = f"📏 [{route.name}] Largura: {new_w:.2f}x (Scroll ↕ ou +/-)"
        self.hud_time = time.time()

        if route.route_id in self.scale_vars:
            self.scale_vars[route.route_id].set(new_w)
        if route.route_id in self.scale_labels:
            self.scale_labels[route.route_id].config(text=f"{new_w:.2f}x")

        if self.last_frame_bgr is not None:
            self._render_frame(self.last_frame_bgr, [])

    def _find_target_route_at_pos(self, vx, vy):
        """Identifica de forma inteligente a rota sob o cursor do mouse no vídeo."""
        if not self.route_engine.routes:
            return None

        if len(self.route_engine.routes) == 1:
            return self.route_engine.routes[0]

        best_route = None
        best_dist = float("inf")

        for route in self.route_engine.routes:
            if route.path_points and len(route.path_points) >= 2:
                s_val, d_perp, closest_pt, _, _ = project_point_to_path((vx, vy), route.path_points)
                w_scale = route.get_width_at_s(s_val) if hasattr(route, 'get_width_at_s') else route.corridor_width
                hw = perspective_half_width(closest_pt[1], w_scale)
                eff_dist = max(0.0, d_perp - hw)
                if eff_dist < best_dist:
                    best_dist = eff_dist
                    best_route = route

        if best_route is not None and best_dist < 180.0:
            return best_route

        # Fallback: rota ativa selecionada ou primeira rota
        if self.selected_route_id is not None:
            for r in self.route_engine.routes:
                if r.route_id == self.selected_route_id:
                    return r
        return self.route_engine.routes[0] if self.route_engine.routes else None

    def _find_vertex_at_pos(self, vx, vy, max_dist=24):
        """Identifica se o cursor do mouse está sobre algum vértice de rota."""
        # Se houver rota selecionada, prioriza seus pontos
        if self.selected_route_id is not None:
            for r in self.route_engine.routes:
                if r.route_id == self.selected_route_id and r.path_points:
                    for idx, p in enumerate(r.path_points):
                        if math.hypot(vx - p[0], vy - p[1]) <= max_dist:
                            return r, idx

        # Senão, busca em todas as rotas
        for r in self.route_engine.routes:
            if r.path_points:
                for idx, p in enumerate(r.path_points):
                    if math.hypot(vx - p[0], vy - p[1]) <= max_dist:
                        return r, idx
        return None, None

    def _on_canvas_mouse_wheel(self, event):
        """Ajusta a espessura da faixa diretamente com a rodinha (scroll) do mouse sobre o vídeo."""
        direction = 1 if event.delta > 0 else -1
        vx, vy = self._canvas_to_video_coords(event.x, event.y)
        self._step_route_width_at_pos(vx, vy, direction * 0.10)
        return "break"

    def _on_canvas_wheel_dir(self, direction, event):
        """Suporte para botões Button-4 e Button-5 (Linux/X11)."""
        vx, vy = self._canvas_to_video_coords(event.x, event.y)
        self._step_route_width_at_pos(vx, vy, direction * 0.10)
        return "break"

    def _step_route_width_at_pos(self, vx, vy, delta):
        """Modifica a largura da rota sob o cursor do mouse ou em modo de desenho."""
        if self.drawing_mode:
            # Em modo de desenho:
            if self.hovered_draw_pt_idx is not None and self.hovered_draw_pt_idx < len(self.current_draw_widths):
                idx = self.hovered_draw_pt_idx
                new_w = max(0.20, min(3.00, round(self.current_draw_widths[idx] + delta, 2)))
                self.current_draw_widths[idx] = new_w
                self.hud_message = f"📏 Ponto #{idx + 1}: Largura ajustada para {new_w:.2f}x (Scroll ↕)"
                self.hud_time = time.time()
            else:
                self.active_draw_width = max(0.20, min(3.00, round(self.active_draw_width + delta, 2)))
                pt_num = len(self.current_draw_points) + 1
                self.hud_message = f"📏 Ponto #{pt_num} (no cursor): Largura {self.active_draw_width:.2f}x (Scroll ↕)"
                self.hud_time = time.time()

            if self.last_frame_bgr is not None:
                self._render_frame(self.last_frame_bgr, [])
            return

        # Modo normal:
        target_route, vertex_idx = self._find_vertex_at_pos(vx, vy)
        if target_route is not None and vertex_idx is not None:
            curr_w = target_route.point_widths[vertex_idx] if (target_route.point_widths and vertex_idx < len(target_route.point_widths)) else target_route.corridor_width
            new_w = max(0.20, min(3.00, round(curr_w + delta, 2)))
            target_route.set_point_width(vertex_idx, new_w)
            self.selected_route_id = target_route.route_id
            ConfigManager.save_routes(self.route_engine.routes)
            self.hud_message = f"📏 [{target_route.name}] Ponto #{vertex_idx + 1}: Largura {new_w:.2f}x (Scroll ↕)"
            self.hud_time = time.time()
            if target_route.route_id in self.scale_vars:
                self.scale_vars[target_route.route_id].set(target_route.corridor_width)
            if target_route.route_id in self.scale_labels:
                self.scale_labels[target_route.route_id].config(text=f"{target_route.corridor_width:.2f}x")
            if self.last_frame_bgr is not None:
                self._render_frame(self.last_frame_bgr, [])
            return

        target_route = self._find_target_route_at_pos(vx, vy)
        if target_route is not None:
            self._adjust_route_width(target_route, delta)

    def _on_shortcut_adjust_width(self, delta):
        """Atalho de teclado (+ ou -) para aumentar ou diminuir a largura."""
        if self.drawing_mode:
            if self.hovered_draw_pt_idx is not None and self.hovered_draw_pt_idx < len(self.current_draw_widths):
                idx = self.hovered_draw_pt_idx
                new_w = max(0.20, min(3.00, round(self.current_draw_widths[idx] + delta, 2)))
                self.current_draw_widths[idx] = new_w
                self.hud_message = f"📏 Ponto #{idx + 1}: Largura ajustada para {new_w:.2f}x (+/-)"
                self.hud_time = time.time()
            else:
                self.active_draw_width = max(0.20, min(3.00, round(self.active_draw_width + delta, 2)))
                pt_num = len(self.current_draw_points) + 1
                self.hud_message = f"📏 Ponto #{pt_num} (no cursor): Largura {self.active_draw_width:.2f}x (+/-)"
                self.hud_time = time.time()

            if self.last_frame_bgr is not None:
                self._render_frame(self.last_frame_bgr, [])
            return

        target_route = None
        if self.mouse_video_pos:
            vx, vy = self.mouse_video_pos
            target_route, vertex_idx = self._find_vertex_at_pos(vx, vy)
            if target_route is not None and vertex_idx is not None:
                curr_w = target_route.point_widths[vertex_idx] if (target_route.point_widths and vertex_idx < len(target_route.point_widths)) else target_route.corridor_width
                new_w = max(0.20, min(3.00, round(curr_w + delta, 2)))
                target_route.set_point_width(vertex_idx, new_w)
                self.selected_route_id = target_route.route_id
                ConfigManager.save_routes(self.route_engine.routes)
                self.hud_message = f"📏 [{target_route.name}] Ponto #{vertex_idx + 1}: Largura {new_w:.2f}x (+/-)"
                self.hud_time = time.time()
                if target_route.route_id in self.scale_vars:
                    self.scale_vars[target_route.route_id].set(target_route.corridor_width)
                if target_route.route_id in self.scale_labels:
                    self.scale_labels[target_route.route_id].config(text=f"{target_route.corridor_width:.2f}x")
                if self.last_frame_bgr is not None:
                    self._render_frame(self.last_frame_bgr, [])
                return

            target_route = self._find_target_route_at_pos(self.mouse_video_pos[0], self.mouse_video_pos[1])

        if target_route is None and self.selected_route_id is not None:
            for r in self.route_engine.routes:
                if r.route_id == self.selected_route_id:
                    target_route = r
                    break

        if target_route is None and self.route_engine.routes:
            target_route = self.route_engine.routes[0]

        if target_route is not None:
            self._adjust_route_width(target_route, delta)

    def _update_total_card(self):
        t = self.route_engine.total_counts
        total_all = sum(t.values())
        self.lbl_total_counts.config(
            text=f"Motos: {t['Moto']}  |  Carros: {t['Carro']}\nCaminhões: {t['Caminhão']}  |  Ônibus: {t['Ônibus']}\nTotal Geral: {total_all} veículos"
        )

    def _load_first_frame(self):
        """Carrega e exibe o primeiro frame do vídeo com percursos desenhados."""
        with self.cap_lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if ret:
                self.last_frame_bgr = frame
                self.current_frame_idx = 0
                self._render_frame(frame, [])

    def _on_canvas_resize(self, event):
        if self.last_frame_bgr is not None:
            self._render_frame(self.last_frame_bgr, [])

    def _draw_directional_corridor(
        self,
        display_frame,
        path_pts,
        color,
        name="",
        width_scale=1.0,
        point_widths=None,
        is_preview=False,
        precomputed_poly_np=None,
        precomputed_arrows=None,
        is_active=False,
        hovered_pt_idx=None,
    ):
        """
        Renderiza um Percurso Único Direcional (Início ➔ Fim) com:
        - Faixa de corredor translúcida nítida com suporte a larguras variáveis por ponto
        - Alças e barras transversais de espessura em cada vértice
        - Destaque reforçado e contorno visual quando a rota estiver ativa
        - Setas direcionais ao longo do eixo central indicando o sentido válido (Início ➔ Fim)
        - Marcadores de [INÍCIO] e [FIM] com larguras
        """
        if not path_pts or len(path_pts) < 2:
            return

        poly_np = precomputed_poly_np
        if poly_np is None:
            corridor_poly = build_corridor_polygon(path_pts, width_scale=width_scale, point_widths=point_widths)
            if corridor_poly and len(corridor_poly) >= 3:
                poly_np = np.array(corridor_poly, dtype=np.int32)

        if poly_np is not None and len(poly_np) >= 3:
            rx, ry, rw, rh = cv2.boundingRect(poly_np)
            rx1 = max(0, rx)
            ry1 = max(0, ry)
            rx2 = min(display_frame.shape[1], rx + rw)
            ry2 = min(display_frame.shape[0], ry + rh)

            if rx2 > rx1 and ry2 > ry1:
                roi = display_frame[ry1:ry2, rx1:rx2]
                mask = np.zeros((ry2 - ry1, rx2 - rx1, 3), dtype=np.uint8)
                poly_offset = poly_np - [rx1, ry1]
                cv2.fillPoly(mask, [poly_offset], color)

                alpha = 0.32 if is_active else (0.24 if not is_preview else 0.30)
                cv2.addWeighted(mask, alpha, roi, 1.0, 0, roi)

                # Contorno externo bem visível
                line_thick = 2 if is_active else 1
                cv2.polylines(display_frame, [poly_np], True, color, line_thick, cv2.LINE_AA)
                if is_active:
                    cv2.polylines(display_frame, [poly_np], True, (255, 255, 255), 1, cv2.LINE_AA)

        # Desenha setas direcionais indicando o sentido (Início -> Fim)
        resampled = precomputed_arrows
        if resampled is None:
            t_len = math.hypot(path_pts[-1][0] - path_pts[0][0], path_pts[-1][1] - path_pts[0][1])
            step = max(35.0, min(65.0, t_len / 4.0))
            resampled = resample_polyline(path_pts, step_px=step)

        if resampled and len(resampled) >= 2:
            arrow_color = (255, 255, 255) if is_active else color
            for i in range(len(resampled) - 1):
                p1 = resampled[i]
                p2 = resampled[i + 1]
                cv2.arrowedLine(
                    display_frame,
                    p1,
                    p2,
                    arrow_color,
                    2,
                    cv2.LINE_AA,
                    tipLength=0.30,
                )

        n_pts = len(path_pts)
        pw_list = point_widths if point_widths else [width_scale] * n_pts
        if len(pw_list) < n_pts:
            pw_list = list(pw_list) + [width_scale] * (n_pts - len(pw_list))

        # Alças e barras transversais de espessura nos vértices
        if is_active or is_preview:
            for i in range(n_pts):
                pt = path_pts[i]
                w_i = pw_list[i]
                is_hovered = (hovered_pt_idx == i)

                if i == 0:
                    dx = path_pts[1][0] - path_pts[0][0]
                    dy = path_pts[1][1] - path_pts[0][1]
                elif i == n_pts - 1:
                    dx = path_pts[-1][0] - path_pts[-2][0]
                    dy = path_pts[-1][1] - path_pts[-2][1]
                else:
                    dx = path_pts[i + 1][0] - path_pts[i - 1][0]
                    dy = path_pts[i + 1][1] - path_pts[i - 1][1]

                norm = math.hypot(dx, dy)
                if norm < 1e-5:
                    nx, ny = 0.0, 1.0
                else:
                    nx, ny = -dy / norm, dx / norm

                hw = perspective_half_width(pt[1], w_i)
                lx = int(round(pt[0] + nx * hw))
                ly = int(round(pt[1] + ny * hw))
                rx = int(round(pt[0] - nx * hw))
                ry = int(round(pt[1] - ny * hw))

                bar_col = (0, 240, 255) if is_hovered else (200, 200, 200)
                bar_th = 2 if is_hovered else 1
                cv2.line(display_frame, (lx, ly), (rx, ry), bar_col, bar_th, cv2.LINE_AA)
                cv2.circle(display_frame, (lx, ly), 3, bar_col, -1, cv2.LINE_AA)
                cv2.circle(display_frame, (rx, ry), 3, bar_col, -1, cv2.LINE_AA)

                # Marcador central
                p_col = (0, 255, 0) if i == 0 else ((0, 215, 255) if (i == n_pts - 1 and not is_preview) else (255, 170, 0))
                radius = 8 if is_hovered else 5
                cv2.circle(display_frame, pt, radius, p_col, -1, cv2.LINE_AA)
                cv2.circle(display_frame, pt, radius + 2, (255, 255, 255), 1, cv2.LINE_AA)

                if is_hovered:
                    cv2.circle(display_frame, pt, radius + 6, (0, 220, 255), 2, cv2.LINE_AA)
                    lbl = f"P{i + 1}: {w_i:.2f}x [Role Scroll ↕]"
                    cv2.putText(display_frame, lbl, (pt[0] + 12, pt[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 240, 255), 2, cv2.LINE_AA)
                elif n_pts <= 10:
                    lbl = f"P{i + 1}: {w_i:.2f}x"
                    cv2.putText(display_frame, lbl, (pt[0] + 8, pt[1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)

        # Marcador de INÍCIO (Origem) com espessura
        p_start = path_pts[0]
        w_start = pw_list[0]
        start_lbl = f"[INÍCIO] {name} ({w_start:.2f}x)" if name else f"[INÍCIO] ({w_start:.2f}x)"
        cv2.putText(
            display_frame,
            start_lbl,
            (max(6, p_start[0] - 35), max(18, p_start[1] - 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 120),
            1,
            cv2.LINE_AA,
        )

        # Marcador de FIM (Destino)
        if not is_preview:
            p_end = path_pts[-1]
            w_end = pw_list[-1]
            end_lbl = f"[FIM] {name} ({w_end:.2f}x)" if name else f"[FIM] ({w_end:.2f}x)"
            cv2.putText(
                display_frame,
                end_lbl,
                (max(6, p_end[0] - 40), min(self.video_height - 8, p_end[1] + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 235, 255),
                1,
                cv2.LINE_AA,
            )

    def _render_frame(self, frame_bgr, tracked_vehicles):
        """Renderiza o frame com cache de polígonos, HUD flutuante e redimensionamento otimizado."""
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 50 or canvas_h < 50:
            return

        display_frame = frame_bgr.copy()

        # 1. Desenha rotas configuradas
        for route in self.route_engine.routes:
            is_active = (route.route_id == self.selected_route_id)
            hovered_idx = None
            if self.hovered_vertex_info and self.hovered_vertex_info[0] == route.route_id:
                hovered_idx = self.hovered_vertex_info[1]

            if route.path_points and len(route.path_points) >= 2:
                self._draw_directional_corridor(
                    display_frame,
                    route.path_points,
                    route.color,
                    name=route.name,
                    width_scale=route.corridor_width,
                    point_widths=route.point_widths,
                    is_preview=False,
                    precomputed_poly_np=route.corridor_poly_np,
                    precomputed_arrows=route.resampled_arrows,
                    is_active=is_active,
                    hovered_pt_idx=hovered_idx,
                )
            else:
                if route.polygon_a and len(route.polygon_a) >= 3:
                    pts_a = np.array(route.polygon_a, dtype=np.int32)
                    overlay = display_frame.copy()
                    cv2.fillPoly(overlay, [pts_a], (0, 0, 180))
                    cv2.addWeighted(overlay, 0.22, display_frame, 0.78, 0, display_frame)
                    cv2.polylines(display_frame, [pts_a], True, (0, 0, 255), 1, cv2.LINE_AA)
                if route.polygon_b and len(route.polygon_b) >= 3:
                    pts_b = np.array(route.polygon_b, dtype=np.int32)
                    overlay = display_frame.copy()
                    cv2.fillPoly(overlay, [pts_b], (0, 210, 255))
                    cv2.addWeighted(overlay, 0.22, display_frame, 0.78, 0, display_frame)
                    cv2.polylines(display_frame, [pts_b], True, (0, 255, 255), 1, cv2.LINE_AA)

        # 2. Desenha pré-visualização em tempo real durante o desenho
        if self.drawing_mode:
            if not self.current_draw_points and self.mouse_video_pos:
                # Nenhum ponto clicado ainda: exibe crossbar no cursor para o usuário calibrar o 1º ponto
                mx, my = self.mouse_video_pos
                hw = perspective_half_width(my, self.active_draw_width)
                cv2.line(display_frame, (int(mx - hw), int(my)), (int(mx + hw), int(my)), (0, 240, 255), 2, cv2.LINE_AA)
                cv2.circle(display_frame, (int(mx), int(my)), 7, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(display_frame, (int(mx), int(my)), 10, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(
                    display_frame,
                    f"[INÍCIO] P1: {self.active_draw_width:.2f}x (Role o Scroll ↕ para ajustar)",
                    (max(8, int(mx) - 60), max(22, int(my) - 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 255, 200),
                    1,
                    cv2.LINE_AA,
                )
            elif self.current_draw_points:
                preview_pts = list(self.current_draw_points)
                preview_widths = list(self.current_draw_widths)
                if self.mouse_video_pos and preview_pts[-1] != self.mouse_video_pos:
                    preview_pts.append(self.mouse_video_pos)
                    preview_widths.append(self.active_draw_width)

                if len(preview_pts) >= 2:
                    self._draw_directional_corridor(
                        display_frame,
                        preview_pts,
                        color=(0, 255, 255),
                        name=self.temp_route_name,
                        width_scale=self.active_draw_width,
                        point_widths=preview_widths,
                        is_preview=True,
                        is_active=True,
                        hovered_pt_idx=self.hovered_draw_pt_idx,
                    )
                else:
                    p0 = preview_pts[0]
                    cv2.circle(display_frame, p0, 7, (0, 255, 0), -1, cv2.LINE_AA)
                    cv2.putText(
                        display_frame,
                        f"[INÍCIO] ({preview_widths[0]:.2f}x)",
                        (max(6, p0[0] - 25), max(18, p0[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

        # 3. Desenha caixas, IDs e trilhas dos veículos rastreados
        for v in tracked_vehicles:
            x1, y1, x2, y2 = v["bbox"]
            cat = v["category"]
            tid = v["id"]
            color = CATEGORY_COLORS.get(cat, (0, 255, 0))

            is_counted = False
            for r in self.route_engine.routes:
                if tid in r.completed_ids:
                    is_counted = True
                    break

            if is_counted:
                box_color = (0, 215, 255)  # Dourado vibrante
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
                label = f"#{tid} {cat} [✓ CONTOU]"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.rectangle(display_frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), box_color, -1)
                cv2.putText(display_frame, label, (x1 + 3, max(10, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                label = f"#{tid} {cat}"
                font_scale = 0.42 if (y2 < 380) else 0.48
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                cv2.rectangle(display_frame, (x1, max(0, y1 - th - 5)), (x1 + tw + 5, y1), color, -1)
                cv2.putText(display_frame, label, (x1 + 3, max(9, y1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

            trail = v["trail"]
            if len(trail) > 1:
                for i in range(1, len(trail)):
                    cv2.line(display_frame, trail[i - 1], trail[i], color, 2)

        # 4. Desenha badge HUD flutuante no topo do vídeo quando o usuário ajustar a largura
        if self.hud_message and (time.time() - self.hud_time < 2.5):
            hud_text = self.hud_message
            font = cv2.FONT_HERSHEY_SIMPLEX
            f_scale = 0.52
            thick = 2
            (tw, th), _ = cv2.getTextSize(hud_text, font, f_scale, thick)
            pad_x, pad_y = 16, 9
            cx = self.video_width // 2
            y_top = 20
            bx1 = max(0, cx - (tw // 2) - pad_x)
            by1 = y_top
            bx2 = min(display_frame.shape[1], cx + (tw // 2) + pad_x)
            by2 = min(display_frame.shape[0], y_top + th + pad_y * 2)

            if bx2 > bx1 and by2 > by1:
                hud_roi = display_frame[by1:by2, bx1:bx2]
                hud_bg = np.zeros_like(hud_roi)
                cv2.rectangle(hud_bg, (0, 0), (bx2 - bx1, by2 - by1), (25, 30, 42), -1)
                cv2.addWeighted(hud_bg, 0.88, hud_roi, 0.12, 0, hud_roi)
                cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (0, 215, 255), 2)
                cv2.putText(display_frame, hud_text, (bx1 + pad_x, by1 + th + pad_y - 2),
                            font, f_scale, (255, 255, 255), thick, cv2.LINE_AA)

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

        curr_sec = int(self.current_frame_idx / self.video_fps)
        total_sec = int(self.total_frames / self.video_fps)
        eff_mult = self.proc_fps / max(1.0, self.video_fps)
        fps_info = f" | Velocidade Real: {eff_mult:.1f}x ({self.proc_fps:.0f} FPS)" if self.is_playing else ""
        self.lbl_time.config(
            text=f"Frame: {self.current_frame_idx} / {self.total_frames} | ⏱ {curr_sec // 60:02d}:{curr_sec % 60:02d} / {total_sec // 60:02d}:{total_sec % 60:02d}{fps_info}"
        )

    # ------------------ Processamento em Background ------------------

    def _toggle_play(self):
        if self.is_playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        self.is_scrubbing = False
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
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)

    def _reset_counters(self):
        was_playing = self.is_playing
        if was_playing:
            self._pause()

        self.route_engine.reset_counts()
        self.detector.reset()
        self._refresh_route_cards()

        with self.cap_lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_idx = 0
            ret, frame = self.cap.read()
            if ret:
                self.last_frame_bgr = frame
                self._render_frame(frame, [])

        self.timeline_var.set(0)
        self.lbl_timeline_current.config(text="⏱ 00:00")
        self.lbl_rollback_notice.config(
            text="↺ Contadores zerados e vídeo reiniciado no início.",
            fg="#72efdd"
        )
        if was_playing:
            self._play()

    def _on_model_change(self, event):
        val = self.model_var.get()
        if "YOLO11l" in val:
            new_model = "yolo11l.pt"
        elif "YOLO11m" in val:
            new_model = "yolo11m.pt"
        elif "YOLO11n" in val:
            new_model = "yolo11n.pt"
        else:
            new_model = "yolo11s.pt"
        self.detector.switch_model(new_model)

    def _on_speed_change(self, event):
        val = self.speed_var.get()
        if "0.5x" in val:
            self.play_speed = 0.5
        elif "1.0x" in val:
            self.play_speed = 1.0
        elif "2.0x" in val:
            self.play_speed = 2.0
        elif "3.0x" in val:
            self.play_speed = 3.0
        elif "4.0x" in val:
            self.play_speed = 4.0
        elif "8.0x" in val:
            self.play_speed = 8.0
        elif "10.0x" in val:
            self.play_speed = 10.0
        elif "20.0x" in val:
            self.play_speed = 20.0
        elif "30.0x" in val:
            self.play_speed = 30.0
        elif "50.0x" in val:
            self.play_speed = 50.0
        elif "75.0x" in val:
            self.play_speed = 75.0
        elif "100.0x" in val:
            self.play_speed = 100.0
        elif "Turbo" in val:
            self.play_speed = 0.0

    def _compute_frame_step(self):
        """Passo de amostragem proporcional à velocidade que garante fluidez e aceleração real."""
        sp = self.play_speed
        if sp == 0.0:  # Máxima (Turbo)
            return 12
        if sp <= 1.0:
            return 1
        if sp <= 2.0:
            return 2
        if sp <= 3.0:
            return 3
        if sp <= 4.0:
            return 4
        if sp <= 8.0:
            return 6
        if sp <= 10.0:
            return 8
        if sp <= 20.0:
            return 12
        if sp <= 30.0:
            return 16
        if sp <= 50.0:
            return 24
        return 30

    def _worker_loop(self):
        """Thread que faz a leitura acelerada do vídeo e inferência no modelo com fila não-bloqueante."""
        while self.is_playing and not self.stop_worker.is_set():
            t_start = time.time()
            frame_step = self._compute_frame_step()

            with self.cap_lock:
                eof_reached = False
                for _ in range(frame_step - 1):
                    if not self.cap.grab():
                        eof_reached = True
                        break

                if eof_reached:
                    self.is_playing = False
                    self.root.after(0, lambda: self.btn_play.config(text="▶ Reiniciar Análise", bg="#00b4d8"))
                    break

                ret, frame = self.cap.read()
                if not ret:
                    self.is_playing = False
                    self.root.after(0, lambda: self.btn_play.config(text="▶ Reiniciar Análise", bg="#00b4d8"))
                    break

                actual_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                self.current_frame_idx = actual_frame

            # Rastreia veículos com compensação cinemática para o frame_step atual
            tracked = self.detector.process_frame(frame, frame_idx=actual_frame, frame_step=frame_step)

            # Atualiza contagens nos percursos direcionais
            new_events = self.route_engine.update(tracked, actual_frame)

            now = time.time()
            dt_proc = now - t_start
            if dt_proc > 0.001:
                inst_fps = float(frame_step) / dt_proc
                self.proc_fps = 0.80 * self.proc_fps + 0.20 * inst_fps if self.proc_fps > 0 else inst_fps

            # Fila não-bloqueante: se a GUI estiver ocupada, descarta o frame antigo e insere o mais recente
            try:
                self.frame_queue.put_nowait((frame, tracked, new_events, actual_frame))
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait((frame, tracked, new_events, actual_frame))
                except Exception:
                    pass

            if self.play_speed > 0:
                target_frame_time = (float(frame_step) / self.video_fps) / self.play_speed
                elapsed = time.time() - t_start
                sleep_time = target_frame_time - elapsed
                if sleep_time > 0.001:
                    time.sleep(sleep_time)

    def _check_frame_queue(self):
        """Consome frames processados da fila para atualizar o canvas com máxima fluidez e zero lag."""
        latest_item = None
        has_new_events = False

        try:
            while not self.frame_queue.empty():
                item = self.frame_queue.get_nowait()
                latest_item = item
                if item[2]:  # new_events
                    has_new_events = True
        except Exception:
            pass

        if latest_item is not None:
            frame, tracked, _, f_idx = latest_item
            self.last_frame_bgr = frame
            self.current_frame_idx = f_idx
            self._render_frame(frame, tracked)

            if not self.is_scrubbing:
                self.timeline_var.set(f_idx)
                curr_sec = int(f_idx / self.video_fps)
                self.lbl_timeline_current.config(text=f"⏱ {curr_sec // 60:02d}:{curr_sec % 60:02d}")

        if has_new_events:
            self._refresh_route_cards()

        self.root.after(15, self._check_frame_queue)

    # ------------------ Linha do Tempo / Timeline & Rollback ------------------

    def _on_slider_press(self, event):
        self.is_scrubbing = True
        self.was_playing_before_scrub = self.is_playing
        if self.is_playing:
            self._pause()

        width = self.timeline_slider.winfo_width()
        if width > 0:
            fraction = max(0.0, min(1.0, event.x / width))
            target_frame = int(fraction * (self.total_frames - 1))
            self.timeline_var.set(target_frame)
            self._seek_to_frame(target_frame, is_continuous=False)
        return "break"

    def _on_slider_motion(self, event):
        if not self.is_scrubbing:
            return
        width = self.timeline_slider.winfo_width()
        if width > 0:
            fraction = max(0.0, min(1.0, event.x / width))
            target_frame = int(fraction * (self.total_frames - 1))
            self.timeline_var.set(target_frame)

            curr_sec = int(target_frame / self.video_fps)
            self.lbl_timeline_current.config(text=f"⏱ {curr_sec // 60:02d}:{curr_sec % 60:02d}")

            now = time.time()
            if now - self._last_scrub_time > 0.04:
                self._last_scrub_time = now
                self._seek_to_frame(target_frame, is_continuous=True)
        return "break"

    def _on_slider_release(self, event):
        self.is_scrubbing = False
        target_frame = int(self.timeline_var.get())
        self._seek_to_frame(target_frame, is_continuous=False)

        if self.was_playing_before_scrub:
            self.was_playing_before_scrub = False
            self._play()
        return "break"

    def _on_global_release(self, event):
        if self.is_scrubbing:
            self.is_scrubbing = False
            target_frame = int(self.timeline_var.get())
            self._seek_to_frame(target_frame, is_continuous=False)
            if self.was_playing_before_scrub:
                self.was_playing_before_scrub = False
                self._play()

    def _jump_relative(self, delta_seconds):
        """Avança ou retrocede delta_seconds na linha do tempo com base no frame real atual."""
        self.is_scrubbing = False
        target_frame = max(0, min(self.total_frames - 1, self.current_frame_idx + int(delta_seconds * self.video_fps)))
        was_playing = self.is_playing
        if was_playing:
            self._pause()
        self._seek_to_frame(target_frame, is_continuous=False)
        if was_playing:
            self._play()

    def _seek_to_frame(self, target_frame, is_continuous=False):
        target_frame = max(0, min(self.total_frames - 1, int(target_frame)))
        old_frame = self.current_frame_idx

        has_future_events = any(e.get("frame", 0) > target_frame for e in self.route_engine.event_log)
        if target_frame < old_frame or has_future_events:
            deducted = self.route_engine.rollback_to_frame(target_frame)
            self.detector.reset()
            self._refresh_route_cards()

            curr_sec = int(target_frame / self.video_fps)
            time_str = f"{curr_sec // 60:02d}:{curr_sec % 60:02d}"
            if deducted > 0:
                self.lbl_rollback_notice.config(
                    text=f"⏪ Retrocedeu para {time_str} (-{deducted} veículo(s) descontado(s) com sucesso)",
                    fg="#f4a261"
                )
            else:
                self.lbl_rollback_notice.config(
                    text=f"⏪ Retrocedeu para {time_str} (Nenhum veículo contabilizado no intervalo)",
                    fg="#72efdd"
                )
        elif target_frame > old_frame and not is_continuous:
            self.detector.reset()
            curr_sec = int(target_frame / self.video_fps)
            time_str = f"{curr_sec // 60:02d}:{curr_sec % 60:02d}"
            self.lbl_rollback_notice.config(
                text=f"⏩ Avançou para {time_str} (Frame {target_frame})",
                fg="#72efdd"
            )

        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Exception:
                break

        with self.cap_lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = self.cap.read()
            if ret:
                actual_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                self.current_frame_idx = actual_frame
                self.last_frame_bgr = frame
                self._render_frame(frame, [])
            else:
                self.current_frame_idx = target_frame

        self.timeline_var.set(self.current_frame_idx)
        curr_sec = int(self.current_frame_idx / self.video_fps)
        self.lbl_timeline_current.config(text=f"⏱ {curr_sec // 60:02d}:{curr_sec % 60:02d}")

    # ------------------ Modo Desenho de Percurso Único Direcional (Início ➔ Fim) ------------------

    def _start_drawing_route(self):
        """Inicia a criação de percurso em 1 ÚNICA ETAPA: basta traçar o caminho do Início até o Fim."""
        self._pause()
        next_idx = len(self.route_engine.routes) + 1
        name = simpledialog.askstring(
            "Novo Percurso Direcional",
            "Nome do percurso (ex: 'Rota 1: Fundo -> CAM1'):",
            initialvalue=f"Rota {next_idx}: Início -> Fim",
            parent=self.root
        )
        if not name:
            return

        self.temp_route_name = name.strip()
        self.temp_corridor_width = 1.0
        self.drawing_mode = True
        self.current_draw_points = []
        self.current_draw_widths = []
        self.active_draw_width = 1.0
        self.hovered_draw_pt_idx = None
        self.mouse_video_pos = None

        self.draw_banner_frame.pack(fill=tk.X, before=self.canvas_frame, pady=(0, 6))
        self.draw_banner_lbl.config(
            text=f"[{self.temp_route_name}] ➔ PASSO ÚNICO: Clique nos pontos do [INÍCIO] ao [FIM].\n"
                 f"💡 Dica: Role o Scroll do mouse para calibrar a largura de CADA PONTO individualmente! Enter para salvar."
        )
        self.btn_finish_step.config(text="✓ Concluir e Salvar Percurso (Enter)", bg="#2a9d8f", fg="#ffffff")

        if self.last_frame_bgr is not None:
            self._render_frame(self.last_frame_bgr, [])

    def _finish_current_draw_step(self):
        """Salva o Percurso Único Direcional imediatamente em 1 só etapa."""
        if not self.drawing_mode:
            return

        if len(self.current_draw_points) < 2:
            messagebox.showwarning(
                "Percurso Incompleto",
                "Clique em pelo menos 2 pontos do [INÍCIO] até o [FIM] da via.",
                parent=self.root
            )
            return

        cleaned_pts = [self.current_draw_points[0]]
        cleaned_widths = [self.current_draw_widths[0] if self.current_draw_widths else 1.0]
        for i in range(1, len(self.current_draw_points)):
            pt = self.current_draw_points[i]
            w = self.current_draw_widths[i] if i < len(self.current_draw_widths) else 1.0
            if math.hypot(pt[0] - cleaned_pts[-1][0], pt[1] - cleaned_pts[-1][1]) >= 16:
                cleaned_pts.append(pt)
                cleaned_widths.append(w)

        if cleaned_pts[-1] != self.current_draw_points[-1]:
            cleaned_pts.append(self.current_draw_points[-1])
            w_last = self.current_draw_widths[-1] if self.current_draw_widths else 1.0
            cleaned_widths.append(w_last)

        if len(cleaned_pts) < 2 or math.hypot(cleaned_pts[-1][0] - cleaned_pts[0][0], cleaned_pts[-1][1] - cleaned_pts[0][1]) < 12:
            messagebox.showwarning(
                "Percurso Muito Curto",
                "Trace uma distância mínima entre o início e o fim da via.",
                parent=self.root
            )
            return

        r_id = max([r.route_id for r in self.route_engine.routes], default=0) + 1
        color = ROUTE_PALETTE[(r_id - 1) % len(ROUTE_PALETTE)]

        new_route = Route(
            route_id=r_id,
            name=self.temp_route_name,
            path_points=cleaned_pts,
            corridor_width=float(np.mean(cleaned_widths)),
            point_widths=cleaned_widths,
            color=color,
        )
        self.route_engine.add_route(new_route)
        self.selected_route_id = new_route.route_id
        ConfigManager.save_routes(self.route_engine.routes)

        self.hud_message = f"✓ {new_route.name} criada! Gire o Scroll ↕ sobre qualquer ponto para afiná-lo/alargá-lo."
        self.hud_time = time.time()

        self._cancel_drawing()
        self._refresh_route_cards()

        if self.last_frame_bgr is not None:
            self._render_frame(self.last_frame_bgr, [])

    def _cancel_drawing(self):
        """Cancela o modo de desenho."""
        self.drawing_mode = False
        self.current_draw_points = []
        self.current_draw_widths = []
        self.active_draw_width = 1.0
        self.hovered_draw_pt_idx = None
        self.mouse_video_pos = None
        self.draw_banner_frame.pack_forget()

        if self.last_frame_bgr is not None:
            self._render_frame(self.last_frame_bgr, [])

    def _canvas_to_video_coords(self, cx, cy):
        if self.canvas_scale <= 0:
            return (0, 0)
        vx = int((cx - self.canvas_offset_x) / self.canvas_scale)
        vy = int((cy - self.canvas_offset_y) / self.canvas_scale)
        vx = max(0, min(self.video_width - 1, vx))
        vy = max(0, min(self.video_height - 1, vy))
        return (vx, vy)

    def _on_canvas_click(self, event):
        pt = self._canvas_to_video_coords(event.x, event.y)

        if not self.drawing_mode:
            # Identifica se clicou em alguma rota existente para ativá-la
            target = self._find_target_route_at_pos(pt[0], pt[1])
            if target is not None:
                self._select_route(target.route_id)
            return

        # Modo de desenho:
        if self.hovered_draw_pt_idx is not None and self.hovered_draw_pt_idx < len(self.current_draw_widths):
            self.hud_message = f"Ponto #{self.hovered_draw_pt_idx + 1} selecionado ({self.current_draw_widths[self.hovered_draw_pt_idx]:.2f}x). Gire Scroll ↕ para ajustar."
            self.hud_time = time.time()
        else:
            self.current_draw_points.append(pt)
            self.current_draw_widths.append(self.active_draw_width)
            pt_num = len(self.current_draw_points)
            self.hud_message = f"✓ Ponto #{pt_num} marcado ({self.active_draw_width:.2f}x)! Posicione o Ponto #{pt_num + 1} (Scroll ↕ para calibrar largura)"
            self.hud_time = time.time()

        if self.last_frame_bgr is not None:
            self._render_frame(self.last_frame_bgr, [])

    def _on_canvas_drag(self, event):
        if not self.drawing_mode:
            return

        pt = self._canvas_to_video_coords(event.x, event.y)
        if not self.current_draw_points:
            self.current_draw_points.append(pt)
            self.current_draw_widths.append(self.active_draw_width)
        else:
            last = self.current_draw_points[-1]
            if math.hypot(pt[0] - last[0], pt[1] - last[1]) > 25:
                self.current_draw_points.append(pt)
                self.current_draw_widths.append(self.active_draw_width)

        self.mouse_video_pos = pt
        if self.last_frame_bgr is not None:
            self._render_frame(self.last_frame_bgr, [])

    def _on_canvas_motion(self, event):
        pt = self._canvas_to_video_coords(event.x, event.y)
        self.mouse_video_pos = pt

        if self.drawing_mode:
            hovered_idx = None
            for idx, p in enumerate(self.current_draw_points):
                if math.hypot(pt[0] - p[0], pt[1] - p[1]) <= 22:
                    hovered_idx = idx
                    break
            old_h = self.hovered_draw_pt_idx
            self.hovered_draw_pt_idx = hovered_idx
            if (old_h != hovered_idx or len(self.current_draw_points) >= 1) and self.last_frame_bgr is not None:
                self._render_frame(self.last_frame_bgr, [])
        else:
            target_r, v_idx = self._find_vertex_at_pos(pt[0], pt[1])
            new_hover = (target_r.route_id, v_idx) if (target_r and v_idx is not None) else None
            old_hover = self.hovered_vertex_info
            self.hovered_vertex_info = new_hover
            if old_hover != new_hover and self.last_frame_bgr is not None:
                self._render_frame(self.last_frame_bgr, [])

    def _delete_route(self, route_id):
        if messagebox.askyesno("Excluir Percurso", "Tem certeza que deseja remover este percurso?"):
            self.route_engine.remove_route(route_id)
            if self.selected_route_id == route_id:
                self.selected_route_id = self.route_engine.routes[0].route_id if self.route_engine.routes else None
            ConfigManager.save_routes(self.route_engine.routes)
            self._refresh_route_cards()
            if self.last_frame_bgr is not None:
                self._render_frame(self.last_frame_bgr, [])

    def _clear_all_routes(self):
        if messagebox.askyesno("Limpar Percursos", "Deseja remover todos os percursos configurados?"):
            self.route_engine.clear_routes()
            self.selected_route_id = None
            ConfigManager.save_routes(self.route_engine.routes)
            self._refresh_route_cards()
            if self.last_frame_bgr is not None:
                self._render_frame(self.last_frame_bgr, [])

    def _load_preset_routes(self):
        """Carrega os percursos direcionais calibrados com espessura realista para este vídeo."""
        if messagebox.askyesno("Carregar Percursos Padrão", "Carregar os percursos direcionais calibrados (Início ➔ Fim) para este vídeo?"):
            self.route_engine.clear_routes()

            # Rota 1: Fundo -> Viaduto Direita (CAM1)
            r1 = Route(
                route_id=1,
                name="Rota 1: Fundo -> CAM1 (Direita)",
                path_points=[(425, 295), (585, 420), (770, 555), (1060, 675)],
                corridor_width=1.0,
                point_widths=[0.85, 0.95, 1.05, 1.15],
                color=(0, 255, 255),
            )

            # Rota 2: Via Distante Direita -> Fundo
            r2 = Route(
                route_id=2,
                name="Rota 2: Via Distante (Direita -> Fundo)",
                path_points=[(1265, 395), (1120, 355), (955, 325)],
                corridor_width=0.85,
                point_widths=[0.90, 0.85, 0.80],
                color=(255, 160, 0),
            )

            self.route_engine.add_route(r1)
            self.route_engine.add_route(r2)
            self.selected_route_id = 1
            ConfigManager.save_routes(self.route_engine.routes)

            self.hud_message = "✓ Percursos padrão carregados! Role o Scroll do mouse sobre qualquer ponto para calibrar."
            self.hud_time = time.time()

            self._refresh_route_cards()
            if self.last_frame_bgr is not None:
                self._render_frame(self.last_frame_bgr, [])
