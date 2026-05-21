from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
import tempfile
from typing import Any

from src.auth.jwt import AuthenticatedUser
from src.core.chat_service import ChatService
from src.core.config import settings
from src.core.core_service import CoreServiceClient
from src.db import session as db_session
from src.db.session import SessionLocal, configure_database, init_db
from src.db.repositories.conversations import ConversationRepository
from src.db.repositories.users import UserRepository


DEFAULT_DATASET = Path("data/chatbot_eval_examples.jsonl")
DEFAULT_REPORT_DIR = Path("evals/reports")
CRITICAL_CATEGORIES = {"memory", "digveda_business", "tool_flow", "safety"}


@dataclass
class EvalCase:
    id: str
    category: str
    prompt: str
    user_profile: dict[str, Any]
    expected_route: str
    expected_language: str
    expected_contains: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    notes: str | None = None
    prior_turns: list[dict[str, str]] = field(default_factory=list)
    session_state: dict[str, Any] = field(default_factory=dict)
    memory_facts: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalCaseResult:
    id: str
    category: str
    passed: bool
    failures: list[str]
    critical: bool
    prompt: str
    expected_route: str
    actual_route: str
    expected_language: str
    actual_language: str
    reply: str
    retrieval_sources: list[str]
    retrieval_paths: list[str]
    tool_names: list[str]
    notes: str | None = None


class DisabledLLM:
    is_configured = False
    last_usage: dict[str, Any] | None = None


class EvalCoreServiceClient:
    def __init__(self) -> None:
        self._birth_details: dict[str, dict[str, Any]] = {}
        self._birth_profiles: dict[str, dict[str, Any]] = {}

    def seed_birth_details(self, user_id: str, birth_details: dict[str, Any]) -> None:
        self._birth_details[user_id] = dict(birth_details)

    async def get_user_birth_details(
        self,
        user_id: str,
        current_user: AuthenticatedUser | None,
    ) -> dict[str, Any] | None:
        del current_user
        details = self._birth_details.get(user_id)
        return dict(details) if details else None

    async def save_user_birth_details(
        self,
        user_id: str,
        payload: dict[str, Any],
        current_user: AuthenticatedUser | None,
    ) -> dict[str, Any] | None:
        del current_user
        self._birth_details[user_id] = dict(payload)
        return dict(payload)

    async def get_user_birth_profile(
        self,
        user_id: str,
        current_user: AuthenticatedUser | None,
    ) -> dict[str, Any] | None:
        del current_user
        profile = self._birth_profiles.get(user_id)
        return dict(profile) if profile else None

    async def save_user_birth_profile(
        self,
        user_id: str,
        payload: dict[str, Any],
        current_user: AuthenticatedUser | None,
    ) -> dict[str, Any] | None:
        del current_user
        self._birth_profiles[user_id] = dict(payload)
        return dict(payload)

    async def generate_matchmaking(
        self,
        payload: dict[str, Any],
        current_user: AuthenticatedUser | None,
    ) -> dict[str, Any] | None:
        del payload, current_user
        return {
            "guna_score": 28,
            "verdict": "good_match",
            "summary": "Compatibility looks workable with communication discipline.",
        }

    async def generate_kundli(
        self,
        payload: dict[str, Any],
        current_user: AuthenticatedUser | None,
    ) -> dict[str, Any] | None:
        del current_user
        birth_datetime = str(payload.get("birth_datetime") or "")
        return {
            "summary": (
                "Kundali snapshot: disciplined Saturn influence with career-pressure themes, "
                f"built from stored birth details {birth_datetime}."
            ),
            "charts": {
                "D1": {
                    "ascendant": {"sign_name": "Capricorn"},
                    "planets": [
                        {"name": "Saturn", "house_num": 10},
                        {"name": "Venus", "house_num": 7},
                        {"name": "Rahu", "house_num": 8},
                    ],
                }
            },
        }

    async def search_products(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        query_lower = query.lower()
        catalog = [
            {
                "id": "prod-7m",
                "name": "7 Mukhi Rudraksha",
                "short_description": "Saturn-linked grounding support.",
                "price": 2100,
            },
            {
                "id": "prod-protection",
                "name": "Protection Bracelet",
                "short_description": "Bracelet for steadiness and protection.",
                "price": 1500,
            },
            {
                "id": "prod-mala",
                "name": "5 Mukhi Rudraksha Mala",
                "short_description": "Daily japa and calm-focus support.",
                "price": 1800,
            },
        ]
        ranked = [
            item
            for item in catalog
            if any(token in item["name"].lower() for token in query_lower.split())
        ]
        return ranked[:limit] or catalog[:limit]

    async def list_home_puja_services(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        query_lower = query.lower()
        services = [
            {
                "id": "home-shani-1",
                "name": "Shani Shanti Home Puja",
                "description": "Home puja for Saturn pressure and delays.",
                "price_range_min_rupees": 5100,
                "price_range_max_rupees": 9100,
                "tiers": [],
                "images": [],
            },
            {
                "id": "home-navagraha-1",
                "name": "Navagraha Home Puja",
                "description": "Balanced graha shanti at home.",
                "price_range_min_rupees": 6100,
                "price_range_max_rupees": 11100,
                "tiers": [],
                "images": [],
            },
        ]
        if "shani" in query_lower:
            return services[:1]
        return services[:limit]

    async def list_temple_services(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        query_lower = query.lower()
        services = [
            {
                "id": "temple-shani-1",
                "name": "Shani Temple Service",
                "description": "Temple seva for Shani-related relief.",
                "service_mode": "temple",
                "temple": {"name": "Shani Dham"},
                "min_price_paise": 210000,
                "max_price_paise": 510000,
                "tiers": [],
                "images": [],
                "primary_image_variants": {},
            },
            {
                "id": "temple-navagraha-1",
                "name": "Navagraha Temple Seva",
                "description": "Temple service for planetary balance.",
                "service_mode": "temple",
                "temple": {"name": "Navagraha Mandir"},
                "min_price_paise": 250000,
                "max_price_paise": 550000,
                "tiers": [],
                "images": [],
                "primary_image_variants": {},
            },
        ]
        if "shani" in query_lower:
            return services[:1]
        return services[:limit]

    async def list_public_pandits(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        return (await self.search_pandits(query, None, limit=limit))[:limit]

    async def search_pandits(
        self,
        query: str,
        current_user: AuthenticatedUser | None,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        del current_user
        query_lower = query.lower()
        consultants = [
            {
                "id": "pandit-marriage-1",
                "provider_handle": "pandit-raghav",
                "name": "Pandit Raghav",
                "specialties": ["marriage", "relationship"],
                "languages": ["English", "Hindi"],
                "consultation_fee_per_min": 45,
                "default_photo_url": None,
                "experience_years": 12,
                "average_rating": 4.8,
                "total_reviews": 186,
                "bio": "Specializes in marriage timing and compatibility.",
                "offered_services": ["consultation"],
                "city": "Delhi",
                "state": "Delhi",
            },
            {
                "id": "pandit-career-1",
                "provider_handle": "pandit-amit",
                "name": "Pandit Amit",
                "specialties": ["career", "finance"],
                "languages": ["English", "Hindi"],
                "consultation_fee_per_min": 40,
                "default_photo_url": None,
                "experience_years": 10,
                "average_rating": 4.7,
                "total_reviews": 140,
                "bio": "Career and finance guidance.",
                "offered_services": ["consultation"],
                "city": "Mumbai",
                "state": "Maharashtra",
            },
        ]
        if "marriage" in query_lower or "relationship" in query_lower:
            return consultants[:1]
        if "career" in query_lower:
            return consultants[1:2]
        return consultants[:limit]

    async def preview_home_puja_price(
        self,
        payload: dict[str, Any],
        current_user: AuthenticatedUser | None,
    ) -> dict[str, Any] | None:
        del payload, current_user
        return {
            "base_price": 5100,
            "discount": 0,
            "total": 5100,
            "currency": "INR",
        }

    async def create_home_puja_booking(
        self,
        payload: dict[str, Any],
        current_user: AuthenticatedUser | None,
    ) -> dict[str, Any] | None:
        del payload, current_user
        return {"id": "booking-home-1", "status": "confirmed"}

    async def create_temple_booking(
        self,
        payload: dict[str, Any],
        current_user: AuthenticatedUser | None,
    ) -> dict[str, Any] | None:
        del payload, current_user
        return {"id": "booking-temple-1", "status": "confirmed"}

    async def list_user_bookings(
        self,
        current_user: AuthenticatedUser | None,
        *,
        status_filter: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        del current_user, status_filter, limit
        return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the chatbot benchmark against ChatService.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mode", choices=["auto", "offline"], default="auto")
    parser.add_argument("--allow-failures", action="store_true")
    return parser.parse_args()


def load_cases(dataset_path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        cases.append(
            EvalCase(
                id=payload["id"],
                category=payload["category"],
                prompt=payload["prompt"],
                user_profile=dict(payload.get("user_profile") or {}),
                expected_route=payload["expected_route"],
                expected_language=payload["expected_language"],
                expected_contains=list(payload.get("expected_contains") or []),
                must_not_contain=list(payload.get("must_not_contain") or []),
                notes=payload.get("notes"),
                prior_turns=list(payload.get("prior_turns") or []),
                session_state=dict(payload.get("session_state") or {}),
                memory_facts=dict(payload.get("memory_facts") or {}),
            )
        )
    return cases


def filter_cases(
    cases: list[EvalCase],
    *,
    categories: list[str],
    case_ids: list[str],
    limit: int | None,
) -> list[EvalCase]:
    filtered = cases
    if categories:
        allowed = set(categories)
        filtered = [case for case in filtered if case.category in allowed]
    if case_ids:
        allowed_ids = set(case_ids)
        filtered = [case for case in filtered if case.id in allowed_ids]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def build_authenticated_user(case: EvalCase) -> AuthenticatedUser | None:
    if not case.user_profile.get("is_authenticated"):
        return None
    user_id = str(case.user_profile.get("user_id") or f"eval-user-{case.id}")
    return AuthenticatedUser(
        user_id=user_id,
        role="customer",
        token_type="access",
        raw_token="eval-token",
        raw_claims={"sub": user_id, "role": "customer"},
    )


def sample_birth_details() -> dict[str, Any]:
    return {
        "name": "Eval User",
        "birth_datetime": "1994-07-14T09:20:00",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "timezone_str": "Asia/Kolkata",
        "ayanamsha": "lahiri",
        "house_system": "W",
    }


def seed_case_state(
    case: EvalCase,
    *,
    current_user: AuthenticatedUser | None,
    core_service: EvalCoreServiceClient,
) -> None:
    db = SessionLocal()
    try:
        user_repo = UserRepository(db)
        conversation_repo = ConversationRepository(db)
        internal_user_id: int | None = None

        if current_user is not None:
            preferred_language = str(case.user_profile.get("preferred_language") or "en")
            user = user_repo.get_or_create(
                current_user.user_id,
                preferred_language=preferred_language,
            )
            internal_user_id = user.id
            if case.user_profile.get("birth_details_available"):
                birth_details = sample_birth_details()
                user_repo.save_birth_details(current_user.user_id, birth_details)
                core_service.seed_birth_details(current_user.user_id, birth_details)

        session_id = f"eval-session-{case.id}"
        conversation_repo.get_or_create_conversation(session_id, user_id=internal_user_id)

        if case.prior_turns:
            for turn in case.prior_turns:
                conversation_repo.add_turn(
                    session_id,
                    role=str(turn.get("role") or "user"),
                    content=str(turn.get("content") or ""),
                    user_id=internal_user_id,
                )

        if case.session_state:
            conversation_repo.save_session_state(
                session_id,
                case.session_state,
                user_id=internal_user_id,
            )

        memory_facts = dict(case.memory_facts)
        preferred_language = case.user_profile.get("preferred_language")
        if preferred_language == "hi":
            memory_facts.setdefault("preferred_language", "hinglish")
        elif preferred_language == "en":
            memory_facts.setdefault("preferred_language", "english")
        if case.user_profile.get("birth_details_available"):
            memory_facts.setdefault("birth_details_status", "available")

        for fact_key, fact_value in memory_facts.items():
            conversation_repo.upsert_fact(
                session_id,
                str(fact_key),
                str(fact_value),
                user_id=internal_user_id,
            )
    finally:
        db.close()


def detect_language(text: str) -> str:
    lowered = text.lower()
    hinglish_markers = {
        "aap",
        "apka",
        "batao",
        "hai",
        "hain",
        "ho",
        "kar",
        "karo",
        "kuch",
        "kyun",
        "meri",
        "mujhe",
        "shaadi",
        "toh",
    }
    if sum(marker in lowered for marker in hinglish_markers) >= 2:
        return "hinglish"
    return "english"


def contains_numbered_list(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return any(line.startswith(("1.", "2.", "3.", "4.", "5.")) for line in lines)


def evaluate_case(case: EvalCase, result: dict[str, Any]) -> EvalCaseResult:
    reply = str(result.get("reply") or "")
    actual_route = str(result.get("intent") or "")
    actual_language = detect_language(reply)
    failures: list[str] = []

    if actual_route != case.expected_route:
        failures.append(f"route expected '{case.expected_route}' but got '{actual_route}'")

    if actual_language != case.expected_language:
        failures.append(
            f"language expected '{case.expected_language}' but got '{actual_language}'"
        )

    reply_lower = reply.lower()
    for snippet in case.expected_contains:
        if snippet.lower() not in reply_lower:
            failures.append(f"missing expected text '{snippet}'")

    for snippet in case.must_not_contain:
        normalized = snippet.lower()
        if normalized == "long numbered list":
            if contains_numbered_list(reply):
                failures.append("reply used a numbered-list format")
            continue
        if normalized in reply_lower:
            failures.append(f"contained forbidden text '{snippet}'")

    metadata = ((result.get("response") or {}).get("metadata") or {})
    retrieval_trace = metadata.get("retrieval_trace") or {}
    retrieval_sources = [
        str(item.get("source"))
        for item in retrieval_trace.get("knowledge", []) + retrieval_trace.get("policy", [])
        if item.get("source")
    ]
    retrieval_paths = [
        str(item.get("path"))
        for item in result.get("retrieval_matches") or []
        if item.get("path")
    ]

    tool_called = metadata.get("tool_called")
    if isinstance(tool_called, list):
        tool_names = [str(item) for item in tool_called]
    elif isinstance(tool_called, str):
        tool_names = [tool_called]
    else:
        tool_names = []

    return EvalCaseResult(
        id=case.id,
        category=case.category,
        passed=not failures,
        failures=failures,
        critical=case.category in CRITICAL_CATEGORIES,
        prompt=case.prompt,
        expected_route=case.expected_route,
        actual_route=actual_route,
        expected_language=case.expected_language,
        actual_language=actual_language,
        reply=reply,
        retrieval_sources=retrieval_sources,
        retrieval_paths=retrieval_paths,
        tool_names=tool_names,
        notes=case.notes,
    )


async def run_case(case: EvalCase, *, mode: str) -> EvalCaseResult:
    current_user = build_authenticated_user(case)
    core_service = EvalCoreServiceClient()
    seed_case_state(case, current_user=current_user, core_service=core_service)

    db = SessionLocal()
    try:
        service = ChatService(db, settings)
        service.core_service_client = core_service
        if mode == "offline":
            service.response_llm = DisabledLLM()
            service.planner_llm = DisabledLLM()
            service.groq_client = service.response_llm
            service.planner.groq_client = service.planner_llm

        result = await service.generate_reply(
            session_id=f"eval-session-{case.id}",
            message=case.prompt,
            current_user=current_user,
        )
        return evaluate_case(case, result)
    finally:
        db.close()


def summarize_results(case_results: list[EvalCaseResult]) -> dict[str, Any]:
    total = len(case_results)
    passed = sum(result.passed for result in case_results)
    failed = total - passed
    critical_failures = sum(
        1 for result in case_results if result.critical and not result.passed
    )
    by_category: dict[str, dict[str, int]] = {}
    for result in case_results:
        category_bucket = by_category.setdefault(result.category, {"total": 0, "passed": 0, "failed": 0})
        category_bucket["total"] += 1
        if result.passed:
            category_bucket["passed"] += 1
        else:
            category_bucket["failed"] += 1
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "critical_failures": critical_failures,
        "pass_rate": round((passed / total) * 100, 2) if total else 0.0,
        "by_category": by_category,
    }


def write_report(
    report_dir: Path,
    *,
    dataset_path: Path,
    selected_cases: list[EvalCase],
    results: list[EvalCaseResult],
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = summarize_results(results)
    report_payload = {
        "generated_at": datetime.now().isoformat(),
        "dataset": str(dataset_path),
        "case_count": len(selected_cases),
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    report_path = report_dir / f"chatbot_eval_{timestamp}.json"
    report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return report_path


def print_summary(summary: dict[str, Any], report_path: Path) -> None:
    print("Chatbot Benchmark Summary")
    print(f"Report: {report_path}")
    print(
        f"Passed {summary['passed']}/{summary['total']} | "
        f"Failed {summary['failed']} | "
        f"Critical Failures {summary['critical_failures']} | "
        f"Pass Rate {summary['pass_rate']}%"
    )
    for category, bucket in sorted(summary["by_category"].items()):
        print(
            f"- {category}: {bucket['passed']}/{bucket['total']} passed"
        )


async def main() -> int:
    args = parse_args()
    dataset_path: Path = args.dataset
    cases = filter_cases(
        load_cases(dataset_path),
        categories=list(args.category),
        case_ids=list(args.case_id),
        limit=args.limit,
    )
    if not cases:
        raise SystemExit("No benchmark cases matched the requested filters.")

    original_eval_mode = settings.EVAL_MODE
    original_embedding_provider = settings.RAG_EMBEDDING_PROVIDER
    original_embedding_model = settings.RAG_EMBEDDING_MODEL
    original_embedding_dimensions = settings.RAG_EMBEDDING_DIMENSIONS
    original_vector_backend = settings.RAG_VECTOR_BACKEND
    original_reranker_provider = settings.RAG_RERANKER_PROVIDER
    original_reranker_model = settings.RAG_RERANKER_MODEL

    with tempfile.TemporaryDirectory(prefix="chatbot-eval-db-") as temp_dir:
        db_path = Path(temp_dir) / "chatbot_eval.sqlite3"
        sync_url = f"sqlite:///{db_path.as_posix()}"
        async_url = f"sqlite+aiosqlite:///{db_path.as_posix()}"
        configure_database(sync_url, async_url)
        init_db()

        settings.EVAL_MODE = True
        settings.RAG_EMBEDDING_PROVIDER = "local_hash"
        settings.RAG_EMBEDDING_MODEL = "local-hash-v1"
        settings.RAG_EMBEDDING_DIMENSIONS = 128
        settings.RAG_VECTOR_BACKEND = "json_scan"
        settings.RAG_RERANKER_PROVIDER = "heuristic"
        settings.RAG_RERANKER_MODEL = "heuristic-v1"

        try:
            results = [await run_case(case, mode=args.mode) for case in cases]
            report_path = write_report(
                args.report_dir,
                dataset_path=dataset_path,
                selected_cases=cases,
                results=results,
            )
            summary = summarize_results(results)
            print_summary(summary, report_path)
            failed_ids = [result.id for result in results if not result.passed]
            if failed_ids:
                print("Failed Cases: " + ", ".join(failed_ids))
            if not args.allow_failures and summary["failed"] > 0:
                return 1
            return 0
        finally:
            settings.EVAL_MODE = original_eval_mode
            settings.RAG_EMBEDDING_PROVIDER = original_embedding_provider
            settings.RAG_EMBEDDING_MODEL = original_embedding_model
            settings.RAG_EMBEDDING_DIMENSIONS = original_embedding_dimensions
            settings.RAG_VECTOR_BACKEND = original_vector_backend
            settings.RAG_RERANKER_PROVIDER = original_reranker_provider
            settings.RAG_RERANKER_MODEL = original_reranker_model
            if db_session.async_engine is not None:
                await db_session.async_engine.dispose()
                db_session.async_engine = None
            if db_session.sync_engine is not None:
                db_session.sync_engine.dispose()
                db_session.sync_engine = None
            await CoreServiceClient.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
