# Sistema Inteligente de Contagem de Veículos por Rota

Sistema de visão computacional e inteligência artificial para detecção, rastreamento contínuo e contagem classificada de veículos por percursos direcionais e faixas em vias públicas e viadutos.

## 🚀 Principais Funcionalidades

- **Detecção e Rastreamento em Tempo Real:** Utiliza modelos YOLO11 (YOLO11s Fluido ~40 FPS / YOLO11m / YOLO11l / YOLO11n) em resolução nativa 16:9 HD (`1280x736`) acelerados por GPU DirectML (NVIDIA GeForce RTX, AMD Radeon, Intel Iris).
- **Percurso Único Direcional (Início ➔ Fim em 1 Única Etapa):**
  - Definição simplificada e intuitiva: basta clicar ou arrastar o mouse acompanhando a via de onde os veículos vêm (`[INÍCIO]`) até para onde vão (`[FIM]`).
  - **Setas Direcionais Automáticas (`➔ ➔ ➔`):** Mostram claramente o sentido válido do fluxo. Veículos na contramão ou atravessando transversalmente de outras ruas são automaticamente ignorados.
  - **Faixa de Corredor com Largura Ajustável:** Botões dedicados `[➖ Mais Fina]` e `[➕ Mais Larga]` em cada card de percurso para calibrar a espessura ideal de qualquer via (curta, longa, estreita ou larga).
- **Rastreamento Cinemático Stride-Aware (Alta Velocidade):**
  - Projeção de velocidade vetorial ($\hat{P} = P + \vec{v} \cdot \Delta t$) e interpolação automática, preservando a identidade dos veículos em altas velocidades (`2x`, `3x`, `4x`, `8x`, `10x`, `20x`, `50x`, `100x` e `Turbo`) sem perda de contagem.
- **Tolerância a Semáforos e Oclusões:** Detecta perfeitamente veículos que aguardavam sinal fechado ou que passaram temporariamente atrás de pilares de viadutos.
- **Classificação Automática por Categorias (Ordem de Relevância):**
  - 🏍️ Motos (motocicletas e ciclomotores)
  - 🚗 Carros (veículos de passeio)
  - 🚚 Caminhões (cargas e pesados)
  - 🚌 Ônibus (coletivos e vans)
- **Linha do Tempo Interativa (Timeline Scrubber) com Dedução Automática (Rollback):**
  - Navegue livremente por qualquer instante do vídeo.
  - Ao retroceder, os veículos contabilizados no intervalo são **automaticamente descontados** e seus IDs são liberados para nova contagem.
  - Botões de salto rápido: `[ ⏪ -30s ]`, `[ ◀◀ -10s ]`, `[ ◀ -5s ]`, `[ ◂ -1s ]`, `[ +1s ▸ ]`, `[ +5s ▶ ]`, `[ +10s ▶▶ ]`, `[ +30s ⏩ ]`.
  - Atalhos de teclado: `Seta Esquerda (◀)` retrocede 5s, `Seta Direita (▶)` avança 5s, e `Barra de Espaço` pausa/retoma.
- **Interface Gráfica Interativa (Tkinter/Canvas):**
  - Fila desacoplada não-bloqueante para reprodução ultra-fluida.
  - Totalizador consolidado e painel de estatísticas individuais por percurso.

## 📁 Estrutura do Projeto

```text
ContagemVeiculos/
├── core/
│   ├── config_manager.py     # Gerenciamento e persistência das rotas em JSON
│   ├── detector_tracker.py   # Pipeline YOLO11 HD + Rastreador Cinemático Híbrido GPU DirectML
│   └── route_engine.py       # Lógica matemática de percursos direcionais e validação vetorial
├── ui/
│   └── app_window.py         # Interface gráfica interativa (Tkinter/Canvas)
├── custom_tracker.yaml       # Calibração fina do rastreador
├── routes_config.json        # Configuração das coordenadas e larguras dos percursos ativos
├── Executar.bat              # Script facilitador de inicialização no Windows
├── main.py                   # Ponto de entrada da aplicação
├── test_system.py            # Testes automatizados unitários e de integração
└── README.md
```

## 🛠️ Como Executar

### 1. Pré-requisitos
- Python 3.10 ou superior
- Dependências instaladas:
  ```bash
  pip install ultralytics opencv-python pillow torch onnx onnxruntime-directml
  ```

### 2. Inicialização Rápida
Dê um duplo clique no arquivo:
```cmd
Executar.bat
```
Ou execute via terminal:
```bash
python main.py
```

## ⚙️ Configuração de Percursos

1. Clique no botão **`★ + Novo Percurso (Início ➔ Fim)`**.
2. Digite um nome para o percurso (ex: `Rota 1: Fundo -> CAM1`).
3. Clique ou arraste no vídeo acompanhando a via de onde os veículos surgem até onde eles saem.
4. Pressione **Enter**, clique com o **botão direito** ou no botão **`✓ Concluir Percurso`**.
5. No card lateral da rota, ajuste a espessura da faixa com **`[➖ Mais Fina]`** ou **`[➕ Mais Larga]`** se necessário.
