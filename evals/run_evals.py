"""Correr el golden set contra una o varias configuraciones del agente y registrar todo en MLflow.

Uso (desde la raíz del proyecto):
    uv run --group evals python -m evals.run_evals --modelos claude-haiku-5-5 gpt-6-luna
    uv run --group evals python -m evals.run_evals --modelos claude-sonnet-5-5 --estrategias completo herramientas
    uv run --group evals mlflow ui --backend-store-uri sqlite:///private/mlflow.db

Los resultados contienen respuestas sobre mi CV, así que guardarlos en private/ (fuera de git).
"""

import argparse
import asyncio
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import mlflow
import yaml

from app.agent.agent import build_agent
from app.config import get_settings
from app.llm.base import Completed, LLMError, Message, TextDelta, ToolCalled, Usage, cost_usd
from app.profile import load_profile, render_profile
from evals.judges import Judge, deterministic_checks

RESULTS_DIR = Path("private/eval_results")
TRACKING_URI = "sqlite:///private/mlflow.db"
EXPERIMENT = "agente-cv"


async def run_case(agent, case: dict) -> dict:
    """Ejecutar los turnos del caso en orden y medir la latencia y el primer token del último turno."""
    conversation: list[Message] = []
    usage = Usage()
    tools: list[str] = []
    ttft = latency = None
    model = agent.config.model
    for turn in case["turnos"]:
        conversation.append(Message("user", turn))
        started = time.perf_counter()
        first = None
        parts: list[str] = []
        async for event in agent.stream(conversation):
            if isinstance(event, TextDelta):
                first = first if first is not None else time.perf_counter() - started
                parts.append(event.text)
            elif isinstance(event, ToolCalled):
                tools.append(event.name)
            elif isinstance(event, Completed):
                usage += event.usage
                model = event.model
        conversation.append(Message("assistant", "".join(parts).strip()))
        ttft, latency = first, time.perf_counter() - started
    return {"conversation": conversation, "usage": usage, "tools": tools, "ttft": ttft, "latency": latency, "model": model}


def _scale(x: int) -> float:
    """Convertir la rúbrica de 1-5 a 0-1."""
    return (min(max(x, 1), 5) - 1) / 4


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def _percentile(values, q: float) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return values[min(len(values) - 1, round(q * (len(values) - 1)))]


async def evaluate_config(model: str, strategy: str, cases: list[dict], judges: list[Judge], settings, profile,
                          concurrency: int) -> tuple[dict, list[dict]]:
    # Sin respaldo: si el modelo evaluado falla, el error debe contar contra él.
    agent = build_agent(settings, profile, model=model, strategy=strategy, with_fallback=False)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(case: dict) -> dict:
        async with semaphore:
            row = {"id": case["id"], "categoria": case["categoria"], "comportamiento": case["comportamiento"]}
            try:
                result = await run_case(agent, case)
            except LLMError as e:
                return {**row, "error": str(e)}
            answer = result["conversation"][-1].content
            verdicts = await asyncio.gather(*(j.evaluate(case, result["conversation"]) for j in judges))
            det = deterministic_checks(case, answer)
            return {
                **row,
                "pregunta": " | ".join(case["turnos"]),
                "respuesta": answer,
                "modelo_real": result["model"],
                "herramientas": ",".join(result["tools"]),
                "ttft_s": result["ttft"],
                "latencia_s": result["latency"],
                "tokens_entrada": result["usage"].input_tokens,
                "tokens_cache": result["usage"].cached_tokens,
                "tokens_salida": result["usage"].output_tokens,
                "costo_usd": cost_usd(model, result["usage"]),
                "costo_juez_usd": sum(c for _, c in verdicts),
                "correctitud": _mean(_scale(v.correctitud) for v, _ in verdicts),
                "fidelidad": _mean(_scale(v.fidelidad) for v, _ in verdicts),
                "concision": _mean(_scale(v.concision) for v, _ in verdicts),
                "comportamiento_ok": _mean(float(v.comportamiento_correcto) for v, _ in verdicts),
                "idioma_ok": _mean(float(v.idioma_correcto) for v, _ in verdicts),
                "alucinacion": _mean(float(bool(v.afirmaciones_no_respaldadas)) for v, _ in verdicts),
                "no_respaldadas": " || ".join(a for v, _ in verdicts for a in v.afirmaciones_no_respaldadas),
                "justificacion": " || ".join(f"[{j.model}] {v.justificacion}" for j, (v, _) in zip(judges, verdicts)),
                "inclusion": det["inclusion"],
                "prohibidos_ok": float(not det["prohibidos"]),
                "palabras": det["palabras"],
            }

    rows = await asyncio.gather(*(one(c) for c in cases))
    ok = [r for r in rows if "error" not in r]
    metrics = {
        "correctitud": _mean(r["correctitud"] for r in ok),
        "fidelidad": _mean(r["fidelidad"] for r in ok),
        "comportamiento": _mean(r["comportamiento_ok"] for r in ok),
        "concision": _mean(r["concision"] for r in ok),
        "idioma": _mean(r["idioma_ok"] for r in ok),
        "tasa_alucinacion": _mean(r["alucinacion"] for r in ok),
        "inclusion_terminos": _mean(r["inclusion"] for r in ok),
        "sin_prohibidos": _mean(r["prohibidos_ok"] for r in ok),
        "palabras_media": _mean(r["palabras"] for r in ok),
        "latencia_p50_s": _percentile([r["latencia_s"] for r in ok], 0.5),
        "latencia_p95_s": _percentile([r["latencia_s"] for r in ok], 0.95),
        "ttft_p50_s": _percentile([r["ttft_s"] for r in ok], 0.5),
        "costo_total_usd": sum(r["costo_usd"] or 0 for r in ok),
        "costo_por_caso_usd": _mean(r["costo_usd"] for r in ok),
        "costo_juez_usd": sum(r["costo_juez_usd"] for r in ok),
        "errores": len(rows) - len(ok),
    }
    for category in sorted({c["categoria"] for c in cases}):
        metrics[f"comportamiento__{category}"] = _mean(r["comportamiento_ok"] for r in ok if r["categoria"] == category)
    return metrics, rows


def _markdown(summary: list[dict]) -> str:
    cols = [
        ("config", "Configuración"), ("correctitud", "Correctitud"), ("fidelidad", "Fidelidad"),
        ("comportamiento", "Comportamiento"), ("concision", "Concisión"), ("tasa_alucinacion", "Alucinación"),
        ("palabras_media", "Palabras"), ("latencia_p50_s", "Latencia p50 (s)"), ("ttft_p50_s", "1er token p50 (s)"),
        ("costo_por_caso_usd", "Costo/caso (USD)"),
    ]
    lines = ["| " + " | ".join(t for _, t in cols) + " |", "|" + "---|" * len(cols)]
    for s in summary:
        cells = []
        for key, _ in cols:
            v = s.get(key)
            if isinstance(v, float):
                v = f"{v:.5f}" if key.startswith("costo") else f"{v:.2f}" if key.endswith("_s") or key == "palabras_media" else f"{v:.0%}"
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelos", nargs="+", required=True)
    parser.add_argument("--estrategias", nargs="+", default=["completo"], choices=["completo", "herramientas"])
    parser.add_argument("--jueces", nargs="+", default=["claude-sonnet-5-5", "gpt-6.1-sol"])
    parser.add_argument("--golden", type=Path, default=Path("private/golden_set.yaml"))
    parser.add_argument("--perfil", type=Path, default=None, help="por defecto, PROFILE_PATH")
    parser.add_argument("--limite", type=int, default=None, help="solo los primeros N casos")
    parser.add_argument("--solo", nargs="+", default=None, help="solo estos ids de caso")
    parser.add_argument("--concurrencia", type=int, default=4)
    parser.add_argument("--etiqueta", default="", help="versión del prompt/juez, p. ej. v2")
    args = parser.parse_args()

    settings = get_settings()
    profile = load_profile(args.perfil or settings.profile_path)
    cases = yaml.safe_load(args.golden.read_text(encoding="utf-8"))
    if args.solo:
        cases = [c for c in cases if c["id"] in args.solo]
    cases = cases[: args.limite]
    keys = {
        "anthropic_key": settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None,
        "openai_key": settings.openai_api_key.get_secret_value() if settings.openai_api_key else None,
    }
    perfil_texto = render_profile(profile, con_detalle=True)
    judges = [Judge(m, perfil_texto, **keys) for m in args.jueces]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=Path("private/mlartifacts").resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary = []
    for model in args.modelos:
        for strategy in args.estrategias:
            name = f"{model}/{strategy}" + (f"/{args.etiqueta}" if args.etiqueta else "")
            print(f"\n▶ {name}: {len(cases)} casos...", flush=True)
            started = time.perf_counter()
            metrics, rows = await evaluate_config(model, strategy, cases, judges, settings, profile, args.concurrencia)
            with mlflow.start_run(run_name=name):
                mlflow.log_params({
                    "modelo": model, "estrategia": strategy, "effort": settings.llm_effort,
                    "jueces": ",".join(args.jueces), "golden_set": args.golden.name, "n_casos": len(cases),
                    "etiqueta": args.etiqueta or "-",
                })
                mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
                mlflow.log_table(data={k: [r.get(k) for r in rows] for k in rows[0]}, artifact_file="casos.json")
            (RESULTS_DIR / f"{stamp}_{model}_{strategy}{'_' + args.etiqueta if args.etiqueta else ''}.json").write_text(
                json.dumps({"metricas": metrics, "casos": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            summary.append({"config": name, **metrics})
            print(f"  listo en {time.perf_counter() - started:.0f}s | correctitud={metrics['correctitud']:.0%} "
                  f"fidelidad={metrics['fidelidad']:.0%} comportamiento={metrics['comportamiento']:.0%} "
                  f"costo=${metrics['costo_total_usd']:.4f} juez=${metrics['costo_juez_usd']:.4f} "
                  f"errores={metrics['errores']}", flush=True)

    table = _markdown(summary)
    (RESULTS_DIR / f"{stamp}_resumen.md").write_text(table + "\n", encoding="utf-8")
    total = sum(s["costo_total_usd"] + s["costo_juez_usd"] for s in summary)
    print(f"\n{table}\n\nCosto total de esta corrida (agente + jueces): ${total:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
