from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def generate_chart(engine_path: str, payload: dict[str, Any]) -> dict[str, Any]:
    target_path = Path(engine_path).resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Astrology engine path does not exist: {target_path}")

    if str(target_path) not in sys.path:
        sys.path.insert(0, str(target_path))

    from astrology_engine import get_vedic_chart_json

    birth_datetime = datetime.fromisoformat(payload["birth_datetime"])
    return get_vedic_chart_json(
        payload["latitude"],
        payload["longitude"],
        birth_datetime,
        ayanamsha=payload.get("ayanamsha", "LAHIRI"),
        house_system=payload.get("house_system", "W"),
    )

