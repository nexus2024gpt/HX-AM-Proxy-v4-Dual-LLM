# api_usage_tracker.py — HX-AM v4 API Usage Tracker
"""
Три слоя:
  ProviderConfig  — описание провайдера (ключ, модель, роль, аккаунт)
  ProviderUsage   — статистика вызовов и токенов (persistent JSON)
  APIUsageTracker — управление конфигом + account-level ротация + stats API

Ключевая идея: лимиты Groq/Gemini/HF выдаются на АККАУНТ, не на ключ.
Несколько ключей одного аккаунта не умножают лимит.
Ротация происходит на уровне аккаунтов — переключаемся между Nexus и Roman
на основе нагрузки (requests_today per account).

Внутри аккаунта с несколькими ключами (Gemini Nexus: 3 ключа):
  - При ошибке ключа (auth error / quota) переходим к следующему ключу аккаунта
  - При rate-limit переходим к другому АККАУНТУ
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("HXAM.tracker")

# ── Директории ───────────────────────────────────────────────
CONFIG_DIR = Path("config")
PROVIDERS_FILE = CONFIG_DIR / "providers.json"
USAGE_FILE = CONFIG_DIR / "api_usage.json"

# ── Каталог актуальных моделей (для дропдауна в UI) ─────────
KNOWN_MODELS = {
    "groq": [
        "llama-3.3-70b-versatile",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
    "gemini": [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-flash-latest",
    ],
    "huggingface": [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "HuggingFaceH4/zephyr-7b-beta",
        "microsoft/Phi-3-mini-4k-instruct",
    ],
}


# ══════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════

@dataclass
class ProviderConfig:
    id: str                 # уникальный ключ: "groq_nexus", "gemini_roman_default"
    provider: str           # "groq" | "gemini" | "huggingface"
    account: str            # "Nexus" | "Roman"
    label: str              # отображаемое имя в UI
    api_key: str
    api_base: str
    model: str
    roles: List[str]        # ["generator"] | ["verifier"] | ["generator", "verifier"]
    enabled: bool = True
    priority: int = 1       # меньше = выше в очереди внутри аккаунта


@dataclass
class ProviderUsage:
    requests_total: int = 0
    requests_today: int = 0
    tokens_in_total: int = 0
    tokens_out_total: int = 0
    tokens_in_today: int = 0
    tokens_out_today: int = 0
    errors_total: int = 0
    errors_today: int = 0
    last_used: Optional[str] = None
    last_error: Optional[str] = None
    last_reset_date: str = field(
        default_factory=lambda: datetime.now(timezone.utc).date().isoformat()
    )


# ══════════════════════════════════════════════════════════════
# ДЕФОЛТНАЯ КОНФИГУРАЦИЯ (генерируется при первом запуске)
# ══════════════════════════════════════════════════════════════

DEFAULT_PROVIDERS: List[ProviderConfig] = [
    # ── GROQ ──────────────────────────────────────────────────
    ProviderConfig(
        id="groq_nexus",
        provider="groq",
        account="Nexus",
        label="Groq · Nexus (HX-AM)",
        api_key="REDACTED_GROQ_NEXUS",
        api_base="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
        roles=["generator"],
        priority=1,
    ),
    ProviderConfig(
        id="groq_roman",
        provider="groq",
        account="Roman",
        label="Groq · Roman",
        api_key="REDACTED_GROQ_ROMAN",
        api_base="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
        roles=["generator"],
        priority=2,
    ),
    # ── GEMINI ────────────────────────────────────────────────
    # Nexus-аккаунт — 3 ключа, одни лимиты. При перегрузке → Roman
    ProviderConfig(
        id="gemini_nexus_hxam",
        provider="gemini",
        account="Nexus",
        label="Gemini · Nexus (HX-AM Proxy)",
        api_key="REDACTED_GEMINI_NEXUS_HXAM",
        api_base="https://generativelanguage.googleapis.com/v1beta",
        model="gemini-2.5-flash",
        roles=["verifier"],
        priority=1,
    ),
    ProviderConfig(
        id="gemini_nexus_openclaw",
        provider="gemini",
        account="Nexus",
        label="Gemini · Nexus (Open Claw)",
        api_key="REDACTED_GEMINI_NEXUS_OPENCLAW",
        api_base="https://generativelanguage.googleapis.com/v1beta",
        model="gemini-2.5-flash",
        roles=["verifier"],
        priority=2,
    ),
    ProviderConfig(
        id="gemini_nexus_default",
        provider="gemini",
        account="Nexus",
        label="Gemini · Nexus (Default)",
        api_key="REDACTED_GEMINI_NEXUS_DEFAULT",
        api_base="https://generativelanguage.googleapis.com/v1beta",
        model="gemini-2.5-flash",
        roles=["verifier"],
        priority=3,
    ),
    # Roman-аккаунт — 2 ключа, независимые лимиты от Nexus
    ProviderConfig(
        id="gemini_roman_hxam",
        provider="gemini",
        account="Roman",
        label="Gemini · Roman (hx_am proxy)",
        api_key="REDACTED_GEMINI_ROMAN_HXAM",
        api_base="https://generativelanguage.googleapis.com/v1beta",
        model="gemini-2.5-flash",
        roles=["verifier"],
        priority=4,
    ),
    ProviderConfig(
        id="gemini_roman_default",
        provider="gemini",
        account="Roman",
        label="Gemini · Roman (Default)",
        api_key="REDACTED_GEMINI_ROMAN_DEFAULT",
        api_base="https://generativelanguage.googleapis.com/v1beta",
        model="gemini-2.5-flash",
        roles=["verifier"],
        priority=5,
    ),
    # ── HUGGINGFACE ───────────────────────────────────────────
    ProviderConfig(
        id="hf_nexus",
        provider="huggingface",
        account="Nexus",
        label="HuggingFace · Nexus",
        api_key="REDACTED_HF_NEXUS",
        api_base="https://api-inference.huggingface.co/v1",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        roles=["generator", "verifier"],
        priority=6,
    ),
    ProviderConfig(
        id="hf_roman",
        provider="huggingface",
        account="Roman",
        label="HuggingFace · Roman",
        api_key="REDACTED_HF_ROMAN",
        api_base="https://api-inference.huggingface.co/v1",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        roles=["generator", "verifier"],
        priority=7,
    ),
]


# ══════════════════════════════════════════════════════════════
# ТРЕКЕР
# ══════════════════════════════════════════════════════════════

class APIUsageTracker:

    def __init__(self, config_dir: str = "config"):
        self._dir = Path(config_dir)
        self._dir.mkdir(exist_ok=True)
        self._pfile = self._dir / "providers.json"
        self._ufile = self._dir / "api_usage.json"
        self._providers: List[ProviderConfig] = []
        self._usage: dict[str, ProviderUsage] = {}
        self._load_providers()
        self._load_usage()

    # ── Конфигурация провайдеров ─────────────────────────────

    def _load_providers(self):
        if not self._pfile.exists():
            self._providers = list(DEFAULT_PROVIDERS)
            self._save_providers()
            logger.info("APITracker: первый запуск — конфиг провайдеров создан")
            return
        try:
            data = json.loads(self._pfile.read_text(encoding="utf-8"))
            self._providers = [ProviderConfig(**p) for p in data]
            logger.info(f"APITracker: загружено {len(self._providers)} провайдеров")
        except Exception as e:
            logger.error(f"APITracker: ошибка загрузки конфига — {e}, используются defaults")
            self._providers = list(DEFAULT_PROVIDERS)

    def _save_providers(self):
        self._pfile.write_text(
            json.dumps([asdict(p) for p in self._providers], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_providers(self) -> List[dict]:
        """Возвращает список всех провайдеров (с замаскированными ключами для UI)."""
        result = []
        for p in self._providers:
            d = asdict(p)
            # Ключ не скрываем — UI должен уметь редактировать
            result.append(d)
        return result

    def update_providers(self, providers_data: List[dict]) -> bool:
        """Обновляет конфиг провайдеров из UI."""
        try:
            self._providers = [ProviderConfig(**p) for p in providers_data]
            self._save_providers()
            logger.info(f"APITracker: сохранено {len(self._providers)} провайдеров")
            return True
        except Exception as e:
            logger.error(f"APITracker: update_providers failed — {e}")
            return False

    def add_provider(self, provider_data: dict) -> bool:
        try:
            p = ProviderConfig(**provider_data)
            # Проверяем уникальность id
            existing_ids = {ep.id for ep in self._providers}
            if p.id in existing_ids:
                p.id = f"{p.id}_{int(time.time())}"
            self._providers.append(p)
            self._save_providers()
            return True
        except Exception as e:
            logger.error(f"APITracker: add_provider failed — {e}")
            return False

    def delete_provider(self, provider_id: str) -> bool:
        before = len(self._providers)
        self._providers = [p for p in self._providers if p.id != provider_id]
        if len(self._providers) < before:
            self._save_providers()
            return True
        return False

    def get_known_models(self) -> dict:
        return KNOWN_MODELS

    # ── Статистика использования ─────────────────────────────

    def _load_usage(self):
        if not self._ufile.exists():
            self._usage = {}
            return
        try:
            raw = json.loads(self._ufile.read_text(encoding="utf-8"))
            self._usage = {k: ProviderUsage(**v) for k, v in raw.items()}
        except Exception as e:
            logger.error(f"APITracker: ошибка загрузки usage — {e}")
            self._usage = {}

    def _save_usage(self):
        self._ufile.write_text(
            json.dumps({k: asdict(v) for k, v in self._usage.items()},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _get_usage(self, provider_id: str) -> ProviderUsage:
        if provider_id not in self._usage:
            self._usage[provider_id] = ProviderUsage()
        u = self._usage[provider_id]
        today = datetime.now(timezone.utc).date().isoformat()
        if u.last_reset_date != today:
            u.requests_today = 0
            u.tokens_in_today = 0
            u.tokens_out_today = 0
            u.errors_today = 0
            u.last_reset_date = today
        return u

    def record_call(
        self,
        provider_id: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        error: bool = False,
        error_msg: str = "",
    ):
        """Записывает результат вызова. Вызывается из LLMClient после каждого запроса."""
        u = self._get_usage(provider_id)
        u.requests_total += 1
        u.requests_today += 1
        u.tokens_in_total += tokens_in
        u.tokens_out_total += tokens_out
        u.tokens_in_today += tokens_in
        u.tokens_out_today += tokens_out
        u.last_used = datetime.now(timezone.utc).isoformat()
        if error:
            u.errors_total += 1
            u.errors_today += 1
            u.last_error = (error_msg[:200] if error_msg else "unknown error")
        self._save_usage()

    # ── Account-level ротация ────────────────────────────────

    def get_providers_for_role(self, role: str) -> List[ProviderConfig]:
        """
        Возвращает отсортированный список провайдеров для роли.

        Алгоритм:
          1. Фильтр: enabled=True, role in roles
          2. Считаем нагрузку на аккаунт сегодня (provider:account → sum requests_today)
          3. Первичная сортировка: по нагрузке аккаунта (меньше нагруженный — первый)
          4. Вторичная: по priority внутри аккаунта (ниже число = выше приоритет)

        Эффект: Nexus и Roman чередуются по мере нарастания нагрузки.
        Groq Nexus отработал 100 req → Groq Roman идёт первым.
        Gemini Nexus ключ 1 всегда пробуется перед ключами 2,3 того же аккаунта.
        """
        today = datetime.now(timezone.utc).date().isoformat()
        eligible = [p for p in self._providers if p.enabled and role in p.roles]
        if not eligible:
            return []

        # Нагрузка по аккаунтам (ключ = "provider:account")
        acc_load: dict[str, int] = {}
        for p in eligible:
            acc_key = f"{p.provider}:{p.account}"
            u = self._usage.get(p.id)
            today_req = 0
            if u and u.last_reset_date == today:
                today_req = u.requests_today
            acc_load[acc_key] = acc_load.get(acc_key, 0) + today_req

        def sort_key(p: ProviderConfig):
            acc_key = f"{p.provider}:{p.account}"
            return (acc_load.get(acc_key, 0), p.priority)

        return sorted(eligible, key=sort_key)

    # ── Stats API ────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Возвращает полную статистику для дашборда."""
        today = datetime.now(timezone.utc).date().isoformat()
        providers_stats = []

        for p in self._providers:
            u = self._get_usage(p.id)
            today_ok = (u.last_reset_date == today)
            providers_stats.append({
                "id": p.id,
                "label": p.label,
                "provider": p.provider,
                "account": p.account,
                "model": p.model,
                "roles": p.roles,
                "enabled": p.enabled,
                "priority": p.priority,
                "requests_total": u.requests_total,
                "requests_today": u.requests_today if today_ok else 0,
                "tokens_in_total": u.tokens_in_total,
                "tokens_out_total": u.tokens_out_total,
                "tokens_in_today": u.tokens_in_today if today_ok else 0,
                "tokens_out_today": u.tokens_out_today if today_ok else 0,
                "errors_total": u.errors_total,
                "errors_today": u.errors_today if today_ok else 0,
                "last_used": u.last_used,
                "last_error": u.last_error,
            })

        # Агрегаты по аккаунтам
        acc_agg: dict[str, dict] = {}
        for s in providers_stats:
            k = f"{s['provider']}:{s['account']}"
            if k not in acc_agg:
                acc_agg[k] = {
                    "provider": s["provider"],
                    "account": s["account"],
                    "requests_today": 0,
                    "tokens_in_today": 0,
                    "tokens_out_today": 0,
                    "errors_today": 0,
                    "requests_total": 0,
                    "key_count": 0,
                }
            a = acc_agg[k]
            a["requests_today"] += s["requests_today"]
            a["tokens_in_today"] += s["tokens_in_today"]
            a["tokens_out_today"] += s["tokens_out_today"]
            a["errors_today"] += s["errors_today"]
            a["requests_total"] += s["requests_total"]
            a["key_count"] += 1

        totals = {
            "date": today,
            "requests_today": sum(s["requests_today"] for s in providers_stats),
            "tokens_in_today": sum(s["tokens_in_today"] for s in providers_stats),
            "tokens_out_today": sum(s["tokens_out_today"] for s in providers_stats),
            "errors_today": sum(s["errors_today"] for s in providers_stats),
            "requests_total": sum(s["requests_total"] for s in providers_stats),
            "tokens_total": sum(
                s["tokens_in_total"] + s["tokens_out_total"] for s in providers_stats
            ),
            "active_providers": sum(1 for p in self._providers if p.enabled),
        }

        return {
            "providers": providers_stats,
            "accounts": list(acc_agg.values()),
            "totals": totals,
        }

    def reset_today(self):
        today = datetime.now(timezone.utc).date().isoformat()
        for u in self._usage.values():
            u.requests_today = 0
            u.tokens_in_today = 0
            u.tokens_out_today = 0
            u.errors_today = 0
            u.last_reset_date = today
        self._save_usage()
        logger.info("APITracker: дневная статистика сброшена")

    def reset_all(self):
        self._usage = {}
        self._save_usage()
        logger.info("APITracker: вся статистика сброшена")


# ── Singleton ────────────────────────────────────────────────
# Инициализируется один раз при импорте — shared между сервером и LLMClient
tracker = APIUsageTracker()
