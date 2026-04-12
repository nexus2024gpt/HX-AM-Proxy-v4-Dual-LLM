# response_normalizer.py — HX-AM v4 Response Normalizer
"""
Программный нормализатор ответов LLM.

Располагается МЕЖДУ сырым LLM-выводом и валидацией PipelineGuard.
Цель: спасти максимум валидных ответов, исключая ложные карантины.

Стратегии (в порядке применения):
  1. Мульти-стратегийная извлечение JSON (7 методов)
  2. Разрешение псевдонимов полей (English / Russian / mixed)
  3. Нормализация значений (вердикт, b_sync, survival, домен)
  4. Подстановка безопасных defaults для опциональных полей
  5. Аудит-лог всех применённых исправлений → сохраняется в артефакте

Принципы:
  - НЕ изменяет смысл данных, только форму
  - Консервативные defaults (WEAK > FALSE, 0.55 > 0.0)
  - Все исправления логируются для отладки
  - Возвращает is_recoverable=False только при полностью нечитаемом ответе
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("HXAM.normalizer")


# ════════════════════════════════════════════════════════════════
# VALUE MAPS
# ════════════════════════════════════════════════════════════════

DOMAIN_MAP: Dict[str, str] = {
    # Russian
    "социология": "sociology", "психология": "psychology",
    "физика": "physics", "биология": "biology",
    "математика": "mathematics", "химия": "chemistry",
    "лингвистика": "linguistics", "экономика": "economics",
    "экология": "ecology", "нейронаука": "neuroscience",
    "геология": "geology", "медицина": "medicine",
    "астрономия": "astronomy", "история": "history",
    "культура": "culture", "физиология": "physiology",
    "архитектура": "architecture", "политология": "political_science",
    "антропология": "anthropology", "философия": "philosophy",
    "когнитивистика": "cognitive_science", "системология": "systems_theory",
    # Abbreviations / mixed
    "bio": "biology", "chem": "chemistry", "math": "mathematics",
    "econ": "economics", "socio": "sociology", "neuro": "neuroscience",
    "psych": "psychology", "social": "sociology", "bio-": "biology",
    "phys": "physics", "geo": "geology", "ling": "linguistics",
}

VERDICT_MAP: Dict[str, str] = {
    # VALID synonyms
    "valid": "VALID", "верно": "VALID", "подтверждено": "VALID",
    "верный": "VALID", "истина": "VALID", "true": "VALID",
    "correct": "VALID", "подтверждается": "VALID", "1": "VALID",
    "подтверждён": "VALID", "принято": "VALID", "yes": "VALID",
    "да": "VALID", "верифицировано": "VALID", "справедливо": "VALID",
    # WEAK synonyms
    "weak": "WEAK", "слабо": "WEAK", "частично": "WEAK",
    "слабый": "WEAK", "неопределённо": "WEAK", "uncertain": "WEAK",
    "partial": "WEAK", "предположительно": "WEAK", "возможно": "WEAK",
    "спорно": "WEAK", "сомнительно": "WEAK", "требует_уточнения": "WEAK",
    "неоднозначно": "WEAK", "conditional": "WEAK",
    # FALSE synonyms
    "false": "FALSE", "неверно": "FALSE", "ложь": "FALSE",
    "неверный": "FALSE", "rejected": "FALSE", "ложный": "FALSE",
    "0": "FALSE", "no": "FALSE", "отклонено": "FALSE",
    "нет": "FALSE", "опровергнуто": "FALSE", "некорректно": "FALSE",
}

SURVIVAL_MAP: Dict[str, str] = {
    "structural": "STRUCTURAL",
    "структурный": "STRUCTURAL", "структурное": "STRUCTURAL",
    "структурная": "STRUCTURAL", "структурно": "STRUCTURAL",
    "terminological": "TERMINOLOGICAL",
    "терминологический": "TERMINOLOGICAL",
    "терминологическое": "TERMINOLOGICAL",
    "терминология": "TERMINOLOGICAL", "терминологически": "TERMINOLOGICAL",
    "unknown": "UNKNOWN", "неизвестно": "UNKNOWN",
}


# ════════════════════════════════════════════════════════════════
# FIELD ALIASES
# ════════════════════════════════════════════════════════════════

GEN_ALIASES: Dict[str, List[str]] = {
    "hypothesis": [
        "hypothesis", "гипотеза", "hypothèse", "hipótesis",
        "teza", "statement", "утверждение", "предположение", "claim",
        "тезис", "инвариант",
    ],
    "mechanism": [
        "mechanism", "механизм", "mécanisme", "mechanismus",
        "how", "description", "описание", "модель", "model_description",
    ],
    "domain": [
        "domain", "домен", "область", "domaine", "dominio",
        "field", "discipline", "дисциплина", "сфера", "primary_domain",
    ],
    "b_sync": [
        "b_sync", "b-sync", "bsync", "sync_score",
        "synchronization_score", "coherence", "b_синхр", "score",
        "cross_domain_score", "structural_coherence",
    ],
    "implication": [
        "implication", "consequence", "следствие", "вывод",
        "prediction", "выводы", "последствие", "вывод_если_верно",
        "if_true", "result",
    ],
}

VER_ALIASES: Dict[str, List[str]] = {
    "verdict": [
        "verdict", "вердикт", "оценка", "результат",
        "decision", "conclusion", "assessment", "evaluation", "вывод",
    ],
    "confidence": [
        "confidence", "уверенность", "conf", "достоверность",
        "probability", "certainty", "degree", "уровень_уверенности",
        "confidence_score",
    ],
    "issues": [
        "issues", "проблемы", "замечания", "critique",
        "недостатки", "problems", "weaknesses", "слабые_стороны",
        "критика", "недостаток", "ошибки",
    ],
    "translation": [
        "translation", "трансляция", "перевод", "step0",
        "step_0", "шаг_0", "cross_domain", "domain_translation",
        "semantic_translation", "step_zero",
    ],
    "refined_hypothesis": [
        "refined_hypothesis", "refined", "уточненная_гипотеза",
        "уточнённая_гипотеза", "улучшенная_гипотеза",
        "improved_hypothesis", "corrected_hypothesis",
        "уточнение", "revised_hypothesis",
    ],
    "operationalization": [
        "operationalization", "операционализация",
        "operationalisierung", "formalization", "формализация",
        "measurement_protocol",
    ],
}

TRANSLATION_ALIASES: Dict[str, List[str]] = {
    "target_domain": [
        "target_domain", "целевой_домен", "target", "domain",
        "другой_домен", "другая_область",
    ],
    "translated_mechanism": [
        "translated_mechanism", "перевод_механизма", "translation",
        "translated", "трансляция", "механизм_в_домене",
    ],
    "survival": [
        "survival", "выживаемость", "тип", "type", "result",
        "structural_survival", "инвариантность",
    ],
}


# ════════════════════════════════════════════════════════════════
# JSON EXTRACTION — 7 STRATEGIES
# ════════════════════════════════════════════════════════════════

def _close_brackets(s: str) -> str:
    """Close unclosed { and [ brackets in truncated JSON."""
    depth_curly = depth_square = 0
    in_string = escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == '{': depth_curly += 1
            elif ch == '}': depth_curly = max(0, depth_curly - 1)
            elif ch == '[': depth_square += 1
            elif ch == ']': depth_square = max(0, depth_square - 1)
    s = s.rstrip().rstrip(',')
    return s + ']' * depth_square + '}' * depth_curly


def _try_parse(text: str) -> Optional[dict]:
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except Exception:
        return None


def extract_json_multi(text: str) -> Tuple[Optional[dict], str]:
    """
    Multi-strategy JSON extraction from LLM text.
    Returns (dict_or_None, strategy_name).
    """
    if not text or not text.strip():
        return None, "empty_input"

    # Strategy 1: Direct parse
    r = _try_parse(text.strip())
    if r:
        return r, "direct"

    # Strategy 2: Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*", "", cleaned).strip()
    r = _try_parse(cleaned)
    if r:
        return r, "markdown_stripped"

    # Strategy 3: Find outermost { ... }
    start = cleaned.find('{')
    if start != -1:
        end = cleaned.rfind('}')
        if end > start:
            r = _try_parse(cleaned[start:end + 1])
            if r:
                return r, "brace_extraction"

        # Strategy 4: Close unclosed brackets
        candidate = cleaned[start:]
        r = _try_parse(_close_brackets(candidate))
        if r:
            return r, "bracket_closing"

        # Strategy 5: Sliding window — trim at last separator, then close
        s = candidate.rstrip()
        for trim_char in [',', ':', '"', ' ', '\n']:
            pos = s.rfind(trim_char)
            if pos > 10:
                try:
                    closed = _close_brackets(s[:pos].rstrip())
                    r = _try_parse(closed)
                    if r and len(r) >= 1:
                        return r, f"sliding_{trim_char!r}"
                except Exception:
                    pass

    # Strategy 6: Any valid JSON object anywhere in text
    for match in re.finditer(r'\{[^{}]{10,}\}', text, re.DOTALL):
        r = _try_parse(match.group(0))
        if r and len(r) >= 1:
            return r, "inner_object"

    # Strategy 7: Regex key-value extraction for critical fields
    partial: Dict[str, Any] = {}
    for m in re.finditer(r'"([a-zA-Zа-яёА-ЯЁ_][a-zA-Zа-яёА-ЯЁ_0-9]*)"\s*:\s*"([^"]{2,})"', text):
        partial[m.group(1)] = m.group(2)
    for m in re.finditer(r'"([a-zA-Zа-яёА-ЯЁ_][a-zA-Zа-яёА-ЯЁ_0-9]*)"\s*:\s*([0-9][0-9.]*)', text):
        try:
            partial[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    for m in re.finditer(r'"([a-zA-Zа-яёА-ЯЁ_][a-zA-Zа-яёА-ЯЁ_0-9]*)"\s*:\s*\[([^\]]{2,})\]', text):
        items = [i.strip().strip('"') for i in m.group(2).split(',') if i.strip().strip('"')]
        if items:
            partial[m.group(1)] = items
    if partial:
        return partial, "regex_kv"

    return None, "failed"


# ════════════════════════════════════════════════════════════════
# FIELD ALIAS RESOLUTION
# ════════════════════════════════════════════════════════════════

def _resolve_aliases(
    data: dict,
    aliases: Dict[str, List[str]],
) -> Tuple[dict, List[str]]:
    """Map aliased/translated field names to canonical names."""
    repairs: List[str] = []
    result: dict = {}
    used_src: set = set()

    for canonical, variants in aliases.items():
        for variant in variants:
            if variant in data and variant not in used_src:
                result[canonical] = data[variant]
                used_src.add(variant)
                if canonical != variant:
                    repairs.append(f"alias '{variant}'→'{canonical}'")
                break

    # Pass through unrecognized fields
    for k, v in data.items():
        if k not in used_src:
            result[k] = v

    return result, repairs


# ════════════════════════════════════════════════════════════════
# VALUE NORMALIZATION HELPERS
# ════════════════════════════════════════════════════════════════

def _norm_domain(val: Any) -> Tuple[str, List[str]]:
    if not val:
        return "general", []
    s = str(val).strip().lower()
    normalized = DOMAIN_MAP.get(s, s)
    repairs = [f"domain '{val}'→'{normalized}'"] if normalized != s else []
    return normalized or "general", repairs


def _norm_b_sync(val: Any) -> Tuple[Optional[float], List[str]]:
    if val is None:
        return None, []
    try:
        f = float(str(val).strip().replace(',', '.'))
        if 0.0 <= f <= 1.0:
            fixed = round(f, 3)
            repairs = [f"b_sync cast '{val}'→{fixed}"] if not isinstance(val, float) else []
            return fixed, repairs
        clamped = round(max(0.0, min(1.0, f)), 3)
        return clamped, [f"b_sync clamped {f}→{clamped}"]
    except (ValueError, TypeError):
        return None, [f"b_sync '{val}' non-numeric"]


def _norm_verdict(val: Any) -> Tuple[Optional[str], List[str]]:
    if not val:
        return None, []
    s = str(val).strip()
    upper = s.upper()
    if upper in ("VALID", "WEAK", "FALSE"):
        return upper, []
    normalized = VERDICT_MAP.get(s.lower())
    if normalized:
        return normalized, [f"verdict '{s}'→'{normalized}'"]
    return None, [f"verdict '{s}' unrecognized"]


def _norm_survival(val: Any) -> Tuple[str, List[str]]:
    if not val:
        return "UNKNOWN", ["survival missing→UNKNOWN"]
    s = str(val).strip()
    upper = s.upper()
    if upper in ("STRUCTURAL", "TERMINOLOGICAL", "UNKNOWN"):
        return upper, []
    normalized = SURVIVAL_MAP.get(s.lower())
    if normalized:
        return normalized, [f"survival '{s}'→'{normalized}'"]
    return "UNKNOWN", [f"survival '{s}' unrecognized→UNKNOWN"]


def _norm_confidence(val: Any) -> float:
    try:
        f = float(str(val).strip().replace(',', '.') if val is not None else "0.5")
        return round(max(0.0, min(1.0, f)), 3)
    except (ValueError, TypeError):
        return 0.5


def _norm_issues(val: Any) -> List[str]:
    if not val:
        return []
    if isinstance(val, list):
        return [str(i).strip() for i in val if str(i).strip()]
    if isinstance(val, str):
        if val.strip().startswith('['):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    return [str(i).strip() for i in parsed if str(i).strip()]
            except Exception:
                pass
        return [s.strip() for s in re.split(r'[;,\n]', val) if s.strip()]
    return []


def _norm_translation(translation: Any) -> Tuple[dict, List[str]]:
    """Normalize Step 0 translation object."""
    repairs: List[str] = []

    if not translation:
        return {"survival": "UNKNOWN"}, ["translation missing→empty"]

    if not isinstance(translation, dict):
        if isinstance(translation, str):
            # Try to parse embedded JSON string
            try:
                parsed = json.loads(translation)
                if isinstance(parsed, dict):
                    translation = parsed
                    repairs.append("translation: parsed from string")
            except Exception:
                return {"survival": "UNKNOWN"}, [f"translation not dict→empty"]
        else:
            return {"survival": "UNKNOWN"}, [f"translation type={type(translation).__name__}→empty"]

    # Resolve aliases within translation object
    result, alias_repairs = _resolve_aliases(translation, TRANSLATION_ALIASES)
    repairs.extend(alias_repairs)

    # Normalize survival
    sv, sv_repairs = _norm_survival(result.get("survival"))
    result["survival"] = sv
    repairs.extend(sv_repairs)

    # Ensure required keys exist
    if "target_domain" not in result:
        result["target_domain"] = "unknown"
        repairs.append("translation.target_domain missing→'unknown'")
    if "translated_mechanism" not in result:
        result["translated_mechanism"] = ""
        repairs.append("translation.translated_mechanism missing→''")

    return result, repairs


# ════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════

def normalize_gen(raw_text: str) -> Tuple[dict, List[str], bool]:
    """
    Normalize generator LLM output.

    Returns:
        (normalized_dict, repair_log, is_recoverable)

    is_recoverable=False только если hypothesis полностью нечитаем.
    Все остальные случаи → True с repair_log.
    """
    repairs: List[str] = []

    # ── JSON extraction ──────────────────────────────────────
    data, strategy = extract_json_multi(raw_text)
    if data is None:
        return {}, ["json_extract failed — all strategies exhausted"], False
    if strategy != "direct":
        repairs.append(f"json_extract strategy='{strategy}'")

    # ── Field alias resolution ───────────────────────────────
    data, alias_repairs = _resolve_aliases(data, GEN_ALIASES)
    repairs.extend(alias_repairs)

    # ── Domain normalization ─────────────────────────────────
    domain, d_repairs = _norm_domain(data.get("domain", ""))
    data["domain"] = domain
    repairs.extend(d_repairs)

    # ── b_sync normalization ─────────────────────────────────
    b_sync, bs_repairs = _norm_b_sync(data.get("b_sync"))
    repairs.extend(bs_repairs)
    if b_sync is None:
        # Fallback: look for any numeric-like field
        for fallback in ("confidence", "certainty", "score", "probability"):
            if fallback in data:
                b_sync, _ = _norm_b_sync(data[fallback])
                if b_sync is not None:
                    repairs.append(f"b_sync derived from '{fallback}'={b_sync}")
                    break
        if b_sync is None:
            b_sync = 0.55
            repairs.append("b_sync missing→default 0.55")
    data["b_sync"] = b_sync

    # ── hypothesis validation & recovery ────────────────────
    hypothesis = str(data.get("hypothesis", "")).strip()

    if not hypothesis or len(hypothesis) < 20:
        # Try to recover from other long string fields
        candidates = [
            ("mechanism", str(data.get("mechanism", "")).strip()),
            ("implication", str(data.get("implication", "")).strip()),
        ]
        # Also scan all string values
        for k, v in data.items():
            if k not in {"domain", "b_sync", "hypothesis"} and isinstance(v, str):
                candidates.append((k, v.strip()))

        recovered = False
        for src_key, candidate in candidates:
            if len(candidate) >= 20:
                # Use first 300 chars as hypothesis
                hypothesis = candidate[:300]
                data["hypothesis"] = hypothesis
                repairs.append(f"hypothesis recovered from '{src_key}'")
                recovered = True
                break

        if not recovered:
            return data, repairs + ["hypothesis unrecoverable: too short/missing"], False

    data["hypothesis"] = hypothesis

    # ── Ensure mechanism exists ──────────────────────────────
    if not data.get("mechanism"):
        data["mechanism"] = hypothesis[:200]
        repairs.append("mechanism missing→copied from hypothesis")

    if repairs:
        logger.info(
            f"normalize_gen: {len(repairs)} repair(s): "
            + " | ".join(repairs[:5])
            + (" | ..." if len(repairs) > 5 else "")
        )

    return data, repairs, True


def normalize_ver(raw_text: str) -> Tuple[dict, List[str], bool]:
    """
    Normalize verifier LLM output.

    Returns:
        (normalized_dict, repair_log, is_recoverable)

    is_recoverable=False только при полном провале парсинга.
    Отсутствующий verdict → default WEAK (консервативно, не блокирует).
    """
    repairs: List[str] = []

    # ── JSON extraction ──────────────────────────────────────
    data, strategy = extract_json_multi(raw_text)
    if data is None:
        return {}, ["json_extract failed — all strategies exhausted"], False
    if strategy != "direct":
        repairs.append(f"json_extract strategy='{strategy}'")

    # ── Field alias resolution ───────────────────────────────
    data, alias_repairs = _resolve_aliases(data, VER_ALIASES)
    repairs.extend(alias_repairs)

    # ── Verdict normalization (critical) ─────────────────────
    verdict, v_repairs = _norm_verdict(data.get("verdict"))
    repairs.extend(v_repairs)
    if verdict is None:
        # Infer from confidence
        conf_raw = _norm_confidence(data.get("confidence", 0))
        if conf_raw >= 0.70:
            verdict = "VALID"
            repairs.append(f"verdict inferred VALID from confidence={conf_raw}")
        elif conf_raw >= 0.35:
            verdict = "WEAK"
            repairs.append(f"verdict inferred WEAK from confidence={conf_raw}")
        else:
            verdict = "WEAK"  # Conservative default: WEAK not FALSE
            repairs.append("verdict unrecoverable→default WEAK (conservative)")
    data["verdict"] = verdict

    # ── Confidence ───────────────────────────────────────────
    conf = _norm_confidence(data.get("confidence"))
    if str(data.get("confidence", "")) != str(conf):
        repairs.append(f"confidence normalized→{conf}")
    data["confidence"] = conf

    # ── Translation / Step 0 ─────────────────────────────────
    translation, t_repairs = _norm_translation(data.get("translation"))
    repairs.extend(t_repairs)
    data["translation"] = translation

    # ── Issues ───────────────────────────────────────────────
    data["issues"] = _norm_issues(data.get("issues"))

    # ── Operationalization ───────────────────────────────────
    op = data.get("operationalization")
    if op is not None and not isinstance(op, dict):
        data["operationalization"] = {}
        repairs.append(f"operationalization type={type(op).__name__}→{{}}")

    # ── Refined hypothesis ───────────────────────────────────
    rh = data.get("refined_hypothesis")
    if rh is not None:
        data["refined_hypothesis"] = str(rh).strip()

    if repairs:
        logger.info(
            f"normalize_ver: {len(repairs)} repair(s): "
            + " | ".join(repairs[:5])
            + (" | ..." if len(repairs) > 5 else "")
        )

    return data, repairs, True


def repairs_summary(gen_repairs: List[str], ver_repairs: List[str]) -> dict:
    """Build a compact audit record for artifact storage."""
    return {
        "gen_repairs": gen_repairs,
        "ver_repairs": ver_repairs,
        "gen_repair_count": len(gen_repairs),
        "ver_repair_count": len(ver_repairs),
        "total_repairs": len(gen_repairs) + len(ver_repairs),
    }