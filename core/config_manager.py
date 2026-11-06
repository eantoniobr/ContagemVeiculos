import json
import os
from core.route_engine import Route


class ConfigManager:
    DEFAULT_CONFIG_PATH = "routes_config.json"

    @classmethod
    def save_routes(cls, routes, filepath=DEFAULT_CONFIG_PATH):
        """Salva a lista de rotas em um arquivo JSON."""
        data = {
            "routes": [r.to_dict() for r in routes]
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    @classmethod
    def load_routes(cls, filepath=DEFAULT_CONFIG_PATH):
        """Carrega rotas a partir de um arquivo JSON se existir."""
        if not os.path.exists(filepath):
            return []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            routes = [Route.from_dict(d) for d in data.get("routes", [])]
            return routes
        except Exception as e:
            print(f"Erro ao carregar configurações de rota de {filepath}: {e}")
            return []
