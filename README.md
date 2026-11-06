# Sistema Inteligente de Contagem de Veículos por Rota

Sistema de visão computacional e inteligência artificial para detecção, rastreamento contínuo e contagem classificada de veículos por rotas e faixas em vias públicas e viadutos.

## 🚀 Principais Funcionalidades

- **Detecção e Rastreamento em Tempo Real:** Utiliza modelos YOLO11 (YOLO11s / YOLO11n) integrados ao algoritmo ByteTrack para manter IDs estáveis mesmo sob oclusões parciais e postes.
- **Lógica de Portais Duplos (Gate A → Gate B):** Permite isolar com precisão os veículos que executam manobras específicas (ex: conversão sob viaduto vs. seguir reto), prevenindo contagens duplicadas e ignorando tráfego no sentido oposto.
- **Suporte a Vias Largas e Multi-Faixas:** Algoritmo calibrado e robusto contra variações laterais, garantindo que veículos em qualquer faixa da via sejam contabilizados com precisão.
- **Classificação Automática por Categorias:**
  - 🚗 Carros
  - 🏍️ Motos (motocicletas e ciclomotores)
  - 🚚 Caminhões
  - 🚌 Ônibus
- **Interface Gráfica Completa (Tkinter):**
  - Reprodução de vídeo em alta fluidez (multi-threaded).
  - Desenho interativo de novas rotas diretamente sobre o vídeo.
  - Seleção de velocidade (0.5x, 1.0x, 2.0x, Modo Turbo).
  - Painel de estatísticas individuais por rota e totalizador consolidado.
  - Persistência automática das configurações em JSON.

## 📁 Estrutura do Projeto

```text
ContagemVeiculos/
├── core/
│   ├── config_manager.py     # Gerenciamento e persistência das rotas em JSON
│   ├── detector_tracker.py   # Pipeline YOLO11 + ByteTrack
│   └── route_engine.py       # Lógica matemática de cruzamento de portais e contagem
├── ui/
│   └── app_window.py         # Interface visual gráfica interativa (Tkinter/Canvas)
├── custom_tracker.yaml       # Calibração fina do rastreador ByteTrack
├── routes_config.json        # Configuração das coordenadas das rotas ativas
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
  pip install ultralytics opencv-python pillow
  ```

### 2. Inicialização Rápida
Basta dar um duplo clique no arquivo:
```cmd
Executar.bat
```
Ou executar via terminal:
```bash
python main.py
```

## ⚙️ Configuração de Rotas

As coordenadas das rotas podem ser desenhadas diretamente pela interface através do botão **"+ Desenhar Nova Rota"** ou configuradas no arquivo `routes_config.json`.
Cada rota define:
- **Linha de Entrada (Gate A):** Linha transversal por onde os veículos entram na via de interesse.
- **Linha de Saída (Gate B):** Linha transversal que confirma a rota realizada (ex: alça do viaduto ou continuação em linha reta).
