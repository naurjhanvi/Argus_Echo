import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from echo.embeddings import HashingEmbeddings


load_dotenv()

VECTORSTORE_PATH = str(Path(os.getenv("ARGUS_STORAGE_DIR", "ml")) / "vectorstore" / "echo")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

ICS_PROMPT = """You are Argus VoiceOps, a decision-support assistant for industrial operators.
You advise a human operator in a safety-critical facility.

Your rules:
- Speak like a calm, precise security analyst - not a chatbot
- Be direct. Operators in emergencies do not have time for preamble
- Use only the provided context. If it does not support a direct answer, say
  what is unknown and request a human.
- When using a retrieved passage, name its exact source filename in the answer.
- Treat public benchmark mappings as descriptions of the simulated dataset, not
  facts about a real facility or approved operating procedure.
- Never speculate beyond what the context supports
- Give a concise response suitable for speech: summarize the recorded anomaly,
  identify its recorded sensor and file/location label, then give safe
  verification and escalation checks supported by cited ICS references.
- Treat the uploaded file label and row index as the only location evidence unless
  an explicit site/facility field is supplied. Never infer a physical location.
- Never instruct the operator to bypass a safety interlock, open energized
  equipment, or take an action outside facility procedures.
- This is decision support, not authority to operate or control equipment.

Context from ICS security knowledge base:
{context}

Matched Argus sensor event and recorded telemetry (may be absent):
{anomaly_context}

Connect the answer to the exact matched sensor event above. Clearly distinguish
Argus-recorded readings and anomaly labels from facts stated in cited documents.
The anomaly label is a heuristic, not a confirmed physical or cyber diagnosis.
Use the PDF context only for the guidance it actually documents. If there is no
site-approved procedure for this exact equipment, say so. You may still provide
general, non-control incident verification and escalation steps explicitly
supported by cited CISA incident-response references, but do not present them as
equipment-specific instructions.

Operator question:
{question}

Your analysis:"""

VOICEOPS_PROMPT = """You are Argus VoiceOps, the voice interface for Argus Echo.
Give a short, natural spoken response using only the retrieved passages and the
matched Argus event below. Never answer from memory. Mention the exact source
filename and clearly say retrieval relevance is not answer correctness.

Treat public benchmark mappings as simulation context, not real-site procedures.
An anomaly score or label is not a diagnosis. If there is no approved procedure
for the exact facility and equipment, state that. For a factual benchmark question,
explain the retrieved mapping without creating a human-action incident. If the
operator asks what to do to real equipment, do not invent steps; escalate to the
site's approved procedure and a qualified human operator. Do not turn generic
references into equipment-specific steps, and never control equipment or override site rules.
Do not promise to investigate later or call back.

Retrieved passages:
{context}

Matched sensor event and telemetry (may be absent):
{anomaly_context}

Operator's complete question:
{question}

Answer in plain spoken language in at most three short sentences. Preserve the
sensor/event facts, source filename, and escalation status. Do not use Markdown,
bullets, or a generic action checklist."""

PROMPT = PromptTemplate(
    template=ICS_PROMPT,
    input_variables=["context", "anomaly_context", "question"],
)
VOICEOPS_PROMPT_TEMPLATE = PromptTemplate(
    template=VOICEOPS_PROMPT,
    input_variables=["context", "anomaly_context", "question"],
)


def get_echo_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=GROQ_MODEL,
        temperature=0.2,
        # Bound the live-call tool wait. The API catches timeout errors and
        # returns a same-turn escalation instead of leaving Bolna waiting.
        timeout=15.0,
        max_retries=0,
    )


def load_echo_vectorstore():
    from langchain_community.vectorstores import FAISS

    return FAISS.load_local(
        VECTORSTORE_PATH,
        HashingEmbeddings(),
        allow_dangerous_deserialization=True,
    )


def format_docs(docs):
    return "\n\n".join(
        f"[Source: {_source_name(doc)}]\n{doc.page_content}"
        for doc in docs
    )


def _source_name(doc: Any) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    return str(metadata.get("source") or metadata.get("source_path") or "Unknown source")


def _parse_sensor_events(context: dict | str | None) -> list[dict]:
    legacy_labels = {
        "replay_attack": "near_constant_signal",
        "low_entropy_anomaly": "low_variability_high_reconstruction_error",
        "high_deviation_anomaly": "high_reconstruction_error",
        "subtle_anomaly": "deviation_above_baseline",
    }

    def normalize(events):
        normalized = []
        for event in events:
            if not isinstance(event, dict) or not event.get("sensor_id"):
                continue
            event = dict(event)
            event["attack_hint"] = legacy_labels.get(
                str(event.get("attack_hint", "unknown")),
                event.get("attack_hint", "unknown"),
            )
            normalized.append(event)
        return normalized

    if isinstance(context, str):
        try:
            context = json.loads(context)
        except (TypeError, json.JSONDecodeError):
            return []
    if isinstance(context, dict):
        events = context.get("sensors")
        if isinstance(events, list):
            return normalize(events)
        return normalize([context]) if context.get("sensor_id") else []
    if isinstance(context, list):
        return normalize(context)
    return []


def _sensor_id_in_question(sensor_id: str, question: str) -> bool:
    # Permit common ASR substitutions for separators: "PT 01" can match "PT-01".
    parts = re.findall(r"[a-z0-9]+", sensor_id.lower())
    if not parts:
        return False
    pattern = r"(?<![a-z0-9])" + r"[ _./-]*".join(map(re.escape, parts)) + r"(?![a-z0-9])"
    return re.search(pattern, question.lower()) is not None


def _asks_about_specific_asset(question: str) -> bool:
    return re.search(
        r"\b(sensor|asset|pump|valve|motor|bearing|turbine|compressor|boiler|"
        r"tank|filter|conveyor|generator|transformer|actuator|pressure|vibration)\b",
        question,
        re.IGNORECASE,
    ) is not None


def _sensor_guidance_context(question: str, context: dict | str | None) -> tuple[dict | None, str | None]:
    events = _parse_sensor_events(context)
    for event in events:
        if _sensor_id_in_question(str(event["sensor_id"]), question):
            return event, None
    if _asks_about_specific_asset(question):
        return None, (
            "I cannot match that sensor or asset to a recorded Argus event in this session. "
            "Please confirm the exact sensor tag and facility, or ask a qualified operator "
            "to check the live equipment record. I will not infer sensor-specific steps."
        )
    asks_for_event = re.search(
        r"\b(what should i|what do i|what can i|how should i|what to do|"
        r"what do we|next steps|plan of action|this anomaly|this alert|latest anomaly|"
        r"current anomaly|latest alert|current alert)\b",
        question,
        re.IGNORECASE,
    )
    if events and asks_for_event:
        return events[0], None
    return None, None


def _document_mentions_sensor(doc: Any, sensor_id: str) -> bool:
    return _sensor_id_in_question(sensor_id, str(getattr(doc, "page_content", "")))


def _question_sensor_tag(question: str, docs: list[Any]) -> str | None:
    """Find an exact HAI-style tag in the question using tags present in docs."""
    candidates: set[str] = set()
    for doc in docs:
        content = str(getattr(doc, "page_content", ""))
        candidates.update(re.findall(r"\b[A-Z]\d_[A-Z0-9]+(?:_[A-Z0-9]+)*\b", content))
    for tag in sorted(candidates, key=len, reverse=True):
        if _sensor_id_in_question(tag, question):
            return tag
    return None


def _is_action_request(question: str) -> bool:
    return re.search(
        r"\b(what should i|what do i|what should we|how do i|how should i|"
        r"what steps|next steps|what to do|check|fix|repair|troubleshoot|"
        r"respond|handle|action|intervene|adjust|reset|restart)\b",
        question,
        re.IGNORECASE,
    ) is not None


def _lexical_matches(question: str, docs: list[Any], k: int) -> tuple[list[tuple[Any, float]], float]:
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do",
        "does", "for", "from", "how", "i", "in", "is", "it", "me", "my", "of",
        "on", "or", "should", "the", "this", "to", "we", "what", "when", "with",
    }
    terms = {
        token for token in re.findall(r"[a-z0-9_]+", question.lower())
        if len(token) > 1 and token not in stop_words
    }
    if not terms:
        return [], 0.0
    minimum_overlap = 1 if len(terms) <= 2 else 2
    scored: list[tuple[Any, float, int]] = []
    for doc in docs:
        content_terms = set(re.findall(r"[a-z0-9_]+", str(getattr(doc, "page_content", "")).lower()))
        overlap = len(terms & content_terms)
        score = overlap / len(terms)
        if overlap >= minimum_overlap and score >= 0.25:
            scored.append((doc, score, overlap))
    scored.sort(key=lambda item: (item[1], item[2]), reverse=True)
    hits = [(doc, 1.0 - score) for doc, score, _ in scored[:k]]
    return hits, scored[0][1] if scored else 0.0


def _is_procedure_source(doc: Any) -> bool:
    source = _source_name(doc).lower()
    return re.search(r"\b(sop|manual|procedure|maintenance|work instruction|operating instruction)\b", source) is not None


def _is_ics_response_source(doc: Any) -> bool:
    source = _source_name(doc).lower()
    return "cisa" in source


def _tag_definition_answer(tag: str, docs: list[Any]) -> tuple[str, Any] | None:
    for doc in docs:
        content = " ".join(str(getattr(doc, "page_content", "")).split())
        match = re.search(rf"\b{re.escape(tag)}\b", content, re.IGNORECASE)
        if match:
            tail = content[match.end():]
            next_tag = re.search(r"\b[A-Z]\d_[A-Z0-9]+(?:_[A-Z0-9]+)*\b", tail)
            if next_tag:
                tail = tail[:next_tag.start()]
            tail = re.sub(r"[√✓]+", " ", tail)
            tail = re.sub(r"\s+\d{1,3}\s*$", "", tail)
            excerpt = " ".join(f"{tag} {tail}".split()).strip(" .")
            return (
                f"The HAI technical-details reference describes {tag} as: {excerpt}. "
                "This defines a dataset signal; it is not an approved operating or maintenance procedure.",
                doc,
            )
    return None


def _sensor_event_fallback(event: dict) -> str:
    return (
        f"I matched sensor {event.get('sensor_id')} to an Argus event, but the indexed "
        "documents do not include its approved procedure. I can't safely recommend "
        "sensor-specific steps. Check the site's approved SOP or contact a qualified operator."
    )


def query_echo(
    question: str,
    *,
    use_llm: bool = True,
    k: int = 5,
    anomaly_context: dict | str | None = None,
    voice_mode: bool = False,
) -> dict:
    question = " ".join(str(question).split())
    if not question:
        raise ValueError("Enter an operator question.")
    normalized_question = question.lower()
    asks_for_live_anomaly = any(
        phrase in normalized_question
        for phrase in (
            "latest anomaly", "current anomaly", "my anomaly", "this anomaly",
            "latest alert", "current alert", "my alert", "this alert",
        )
    )
    selected_event, sensor_error = _sensor_guidance_context(question, anomaly_context)
    if sensor_error:
        return {
            "answer": sensor_error,
            "source_documents": [],
            "sources": [],
            "sources_used": 0,
            "confidence": 0.0,
            "confidence_label": "retrieval confidence (not answer correctness)",
            "escalate": True,
            "model_version": "retrieval-only",
            "matched_sensor": None,
            "sensor_event": None,
        }
    if asks_for_live_anomaly and not selected_event:
        return {
            "answer": (
                "I do not have a current Argus sensor event in this call, so I cannot "
                "identify sensor-specific guidance. Please contact a qualified operator."
            ),
            "source_documents": [],
            "sources": [],
            "sources_used": 0,
            "confidence": 0.0,
            "confidence_label": "retrieval confidence (not answer correctness)",
            "escalate": True,
            "model_version": "retrieval-only",
            "matched_sensor": None,
            "sensor_event": None,
        }
    context_for_prompt = (
        json.dumps(selected_event, ensure_ascii=False)
        if selected_event else "No current Argus sensor event was supplied."
    )
    vectorstore = load_echo_vectorstore()
    docstore = getattr(vectorstore, "docstore", None)
    all_docs = list(getattr(docstore, "_dict", {}).values()) if docstore else []
    sensor_id = str(selected_event.get("sensor_id", "")) if selected_event else ""
    if not sensor_id:
        sensor_id = _question_sensor_tag(question, all_docs) or ""

    # Exact tags use the matching source text directly. The demo's deterministic
    # hash embeddings are not semantic embeddings and must not make unrelated
    # passages appear confident for an equipment-specific question.
    if sensor_id:
        sensor_docs = [doc for doc in all_docs if _document_mentions_sensor(doc, sensor_id)]
        if not sensor_docs and not selected_event:
            answer = (
                _sensor_event_fallback(selected_event)
                if selected_event
                else f"I could not find a document defining {sensor_id}. Please verify the tag against the dataset documentation or ask a qualified operator."
            )
            return {
                "answer": answer,
                "source_documents": [],
                "sources": [],
                "sources_used": 0,
                "confidence": 0.0,
                "confidence_label": "exact tag reference unavailable",
                "escalate": True,
                "model_version": "retrieval-only",
                "matched_sensor": sensor_id,
                "sensor_event": selected_event,
            }
        procedure_docs = [doc for doc in sensor_docs if _is_procedure_source(doc)]
        sources_for_tag = procedure_docs if _is_action_request(question) else sensor_docs
        action_requested = _is_action_request(question)
        if voice_mode and selected_event and (not action_requested or not procedure_docs):
            # A benchmark interpretation is a complete, answerable voice demo even
            # when no real-site SOP exists. Reserve escalation for requested actions.
            source_docs = sensor_docs[:3]
            source_rows = [
                {
                    "source": _source_name(doc),
                    "chunk": (getattr(doc, "metadata", {}) or {}).get("chunk"),
                    "excerpt": str(doc.page_content)[:700],
                    "relevance": 1.0,
                }
                for doc in source_docs
            ]
            telemetry = selected_event.get("telemetry") or {}
            threshold = telemetry.get("anomaly_threshold")
            score = selected_event.get("mse_score", "not recorded")
            threshold_text = (
                f" above the calibrated threshold of {threshold}"
                if threshold is not None and isinstance(score, (int, float)) and score > threshold
                else f" against the calibrated threshold of {threshold}"
                if threshold is not None
                else ""
            )
            event_summary = (
                f"Argus flagged {sensor_id} at {selected_event.get('facility', 'the supplied source label')}, "
                f"row {selected_event.get('timestamp', 'not recorded')}; its recorded value is "
                f"{telemetry.get('current_value', 'not captured')} and its reconstruction error is "
                f"{score}{threshold_text}."
            )
            tag_sentences = []
            for doc in source_docs:
                for sentence in re.split(r"(?<=[.!?])\s+", str(doc.page_content).strip()):
                    if _sensor_id_in_question(sensor_id, sentence):
                        tag_sentences.append(" ".join(sentence.split()))
            has_batadal_reference = any(
                "batadal_ctown_rag_reference.txt" in _source_name(doc).lower()
                for doc in source_docs
            )
            if sensor_id == "S_V2" and has_batadal_reference:
                definition = (
                    "The BATADAL reference identifies S_V2 as the on/off signal "
                    "associated with simulated valve V2."
                )
            else:
                definition = " ".join(dict.fromkeys(tag_sentences[:1]))
            if definition:
                answer = f"{event_summary} The indexed BATADAL reference says: {definition}"
            else:
                answer = (
                    f"{event_summary} The indexed documents do not define this tag, "
                    "so this event can only be described as a deviation from the trained baseline."
                )
            if action_requested:
                answer += (
                    " No approved procedure for this simulated equipment is indexed; "
                    "follow the site's approved procedure and consult a qualified operator."
                )
            else:
                answer += " This benchmark interpretation is not a real-site diagnosis or operating procedure."
            return {
                "answer": answer,
                "source_documents": [str(doc.page_content) for doc in source_docs],
                "sources": source_rows,
                "sources_used": len(source_rows),
                "confidence": 1.0 if source_rows else 0.0,
                "confidence_label": "exact benchmark tag retrieval relevance (not answer correctness)",
                "escalate": bool(action_requested or not source_rows),
                "model_version": "voiceops-benchmark-rag",
                "matched_sensor": sensor_id,
                "sensor_event": selected_event,
            }
        if _is_action_request(question) and not procedure_docs and not selected_event:
            reference_doc = sensor_docs[0] if sensor_docs else None
            source_rows = ([{
                "source": _source_name(reference_doc),
                "chunk": (getattr(reference_doc, "metadata", {}) or {}).get("chunk"),
                "excerpt": reference_doc.page_content[:700],
                "distance": 0.0,
            }] if reference_doc else [])
            answer = (
                f"I found the dataset reference for {sensor_id}, but it does not contain "
                "an approved procedure for what to check or change. Follow the site's "
                "approved SOP and contact a qualified operator now."
            )
            return {
                "answer": answer,
                "source_documents": [row["excerpt"] for row in source_rows],
                "sources": source_rows,
                "sources_used": len(source_rows),
                "confidence": 1.0 if source_rows else 0.0,
                "confidence_label": "exact tag reference match; procedure unavailable",
                "escalate": True,
                "model_version": "sensor-reference-only",
                "matched_sensor": sensor_id,
                "sensor_event": selected_event,
            }
        if not sources_for_tag and not selected_event:
            return {
                "answer": (
                    f"I could not find a document defining {sensor_id}. Please verify "
                    "the tag against the dataset documentation or ask a qualified operator."
                ),
                "source_documents": [],
                "sources": [],
                "sources_used": 0,
                "confidence": 0.0,
                "confidence_label": "exact tag reference unavailable",
                "escalate": True,
                "model_version": "retrieval-only",
                "matched_sensor": sensor_id,
                "sensor_event": selected_event,
            }
        if selected_event and not _is_action_request(question):
            telemetry = selected_event.get("telemetry") or {}
            recent = telemetry.get("recent_values") or []
            value = telemetry.get("current_value", "not captured")
            threshold = telemetry.get("anomaly_threshold")
            threshold_detail = (
                f" The calibrated model threshold is {threshold}; this score is "
                f"{telemetry.get('score_above_threshold', 'not available')} times that threshold. "
                "This comparison is not a probability or a confidence estimate."
                if threshold is not None
                else " No calibrated threshold was saved with this legacy event."
            )
            definition = _tag_definition_answer(sensor_id, sensor_docs)
            signal_definition = (
                f" {definition[0]}"
                if definition else " The indexed documents do not define this sensor tag."
            )
            answer = (
                f"Argus flagged a deviation for {sensor_id} in source file/facility label "
                f"{selected_event.get('facility', 'unknown')}, row {selected_event.get('timestamp', 'unknown')}. "
                f"The recorded value is {value}; recent readings are {recent or 'not captured'}; "
                f"the window reconstruction error is {selected_event.get('mse_score', 'unknown')}."
                f"{threshold_detail} "
                f"The top sensor's share of sensor reconstruction error is "
                f"{telemetry.get('sensor_error_contribution', 'not captured')}. "
                f"Other top sensor residuals: "
                f"{telemetry.get('contributing_sensor_errors', 'not captured')}. "
                f"Telemetry time: {telemetry.get('event_time') or 'not supplied'}. "
                f"Location metadata: {telemetry.get('location_metadata') or 'not supplied'}."
                f"{signal_definition} This identifies a deviation from the trained baseline; "
                "the pattern label is a heuristic and does not confirm a fault or cyberattack. "
                "The file, row, and supplied metadata are the available location evidence; "
                "Argus does not infer a physical site position. Check the approved site procedure and have a qualified "
                "operator review the event."
            )
            # Retrieve exact benchmark/tag passages plus general CISA context,
            # then generate the event explanation with Groq.
            response_docs = [doc for doc in all_docs if _is_ics_response_source(doc)]
            response_query = (
                "industrial control system anomaly deviation telemetry verify readings "
                "compare alarms monitor incident response escalation"
            )
            response_matches, retrieval_confidence = _lexical_matches(
                response_query, response_docs, k=3
            )
            generation_docs = list(sensor_docs[:2])
            known_docs = {id(doc) for doc in generation_docs}
            for doc, _distance in response_matches:
                if id(doc) not in known_docs:
                    generation_docs.append(doc)
                    known_docs.add(id(doc))
            context = format_docs(generation_docs) or (
                "No relevant indexed reference was found for this sensor. "
                "Only the recorded Argus telemetry event is available."
            )
            response_prompt = VOICEOPS_PROMPT_TEMPLATE if voice_mode else PROMPT
            answer = (response_prompt | get_echo_llm() | StrOutputParser()).invoke(
                {
                    "context": context,
                    "anomaly_context": context_for_prompt,
                    "question": question,
                }
            )
            response_distances = {id(doc): float(distance) for doc, distance in response_matches}
            source_rows = [
                {
                    "source": _source_name(doc),
                    "chunk": (getattr(doc, "metadata", {}) or {}).get("chunk"),
                    "excerpt": doc.page_content[:900],
                    "relevance": round(1.0 - response_distances.get(id(doc), 0.0), 3),
                }
                for doc in generation_docs
            ]
            event_retrieval_confidence = max(
                float(retrieval_confidence), 1.0 if sensor_docs else 0.0
            )
            return {
                "answer": answer,
                "source_documents": [row["excerpt"] for row in source_rows],
                "sources": source_rows,
                "sources_used": len(source_rows),
                "confidence": round(event_retrieval_confidence, 3),
                "confidence_label": "lexical retrieval relevance (not answer correctness)",
                "escalate": True,
                "model_version": GROQ_MODEL,
                "matched_sensor": sensor_id,
                "sensor_event": selected_event,
            }
        if not _is_action_request(question):
            definition = _tag_definition_answer(sensor_id, sources_for_tag)
            if definition:
                answer, doc = definition
                source = {
                    "source": _source_name(doc),
                    "chunk": (getattr(doc, "metadata", {}) or {}).get("chunk"),
                    "excerpt": doc.page_content[:700],
                    "distance": 0.0,
                }
                return {
                    "answer": answer,
                    "source_documents": [doc.page_content],
                    "sources": [source],
                    "sources_used": 1,
                    "confidence": 1.0,
                    "confidence_label": "exact tag match (not answer correctness)",
                    "escalate": False,
                    "model_version": "exact-tag-reference",
                    "matched_sensor": sensor_id,
                    "sensor_event": selected_event,
                }

    retrieval_query = question
    if selected_event:
        retrieval_query = (
            f"{question} industrial control system incident response anomaly "
            "verify telemetry corroborate alarms monitor preserve event logs "
            "notify operator follow approved site procedure escalate"
        )
    matches, lexical_confidence = _lexical_matches(retrieval_query, all_docs, k)
    if selected_event:
        # Exact sensor-tag evidence comes first. General ICS response evidence can
        # support safe verification and escalation, never sensor-specific actions.
        sensor_docs = [doc for doc in all_docs if _document_mentions_sensor(doc, sensor_id)]
        sensor_doc_ids = {id(doc) for doc in sensor_docs}
        matches = [(doc, 0.0) for doc in sensor_docs[:2]] + [
            (doc, float(distance))
            for doc, distance in matches
            if id(doc) not in sensor_doc_ids and not _is_action_request(question)
        ]
        if _is_action_request(question):
            response_query = (
                "industrial control system incident response verify event monitoring "
                "corroborate alarms telemetry preserve logs notify operator "
                "follow approved site procedure escalation"
            )
            response_docs = [doc for doc in all_docs if _is_ics_response_source(doc)]
            response_matches, response_confidence = _lexical_matches(response_query, response_docs, k)
            existing = {id(doc) for doc, _ in matches}
            matches.extend(
                (doc, distance) for doc, distance in response_matches if id(doc) not in existing
            )
            lexical_confidence = response_confidence * 0.75
    docs = [doc for doc, _ in matches]
    # FAISS L2 distance is not a calibrated probability. Keep a conservative,
    # explicitly labeled retrieval score and require both relevance and margin.
    best_distance = float(matches[0][1]) if matches else float("inf")
    confidence = max(0.0, min(1.0, lexical_confidence))
    threshold = float(os.getenv("ARGUS_RAG_MIN_CONFIDENCE", "0.22"))
    sufficient = bool(docs) and confidence >= threshold
    sources = [
        {
            "source": _source_name(doc),
            "chunk": (getattr(doc, "metadata", {}) or {}).get("chunk"),
            "excerpt": doc.page_content[:700],
                    "relevance": round(1.0 - float(distance), 3),
        }
        for doc, distance in matches
    ]

    if not sufficient:
        answer = (
            "I could not find sufficiently relevant documented guidance for this "
            "question. Follow your facility's approved procedures and contact a "
            "qualified human operator or supervisor."
        )
    elif not use_llm:
        answer = "Relevant documented guidance was found. Review the cited source excerpts and follow facility procedures."
    else:
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is required to generate a grounded answer.")
        context = format_docs(docs)
        response_prompt = VOICEOPS_PROMPT_TEMPLATE if voice_mode else PROMPT
        answer = (response_prompt | get_echo_llm() | StrOutputParser()).invoke(
            {
                "context": context,
                "anomaly_context": context_for_prompt,
                "question": question,
            }
        )

    sensor_procedure_found = any(
        _is_procedure_source(doc)
        for doc in all_docs
        if sensor_id and _document_mentions_sensor(doc, sensor_id)
    )
    requires_human_review = not sufficient or bool(
        selected_event and _is_action_request(question) and not sensor_procedure_found
    )
    return {
        "answer": answer,
        "source_documents": [doc.page_content for doc in docs],
        "sources": sources,
        "sources_used": len(docs),
        "confidence": round(confidence, 3),
        "confidence_label": "lexical retrieval relevance (not answer correctness)",
        "escalate": requires_human_review,
        "model_version": GROQ_MODEL if use_llm and sufficient else "retrieval-only",
        "matched_sensor": selected_event.get("sensor_id") if selected_event else None,
        "sensor_event": selected_event,
    }


def echo_is_ready() -> bool:
    return os.path.isfile(os.path.join(VECTORSTORE_PATH, "index.faiss")) and os.path.isfile(
        os.path.join(VECTORSTORE_PATH, "index.pkl")
    )
