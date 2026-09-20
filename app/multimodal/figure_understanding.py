"""Figure 离线检索描述与按问题原图阅读。"""

import re

from app.multimodal.qwen_vl_client import QwenVLClient


OFFLINE_FIELDS = {
    "figure_type": "other",
    "summary": "",
    "entities": [],
    "components": [],
    "relations": [],
    "visual_evidence": [],
    "numerical_evidence": [],
    "retrieval_keywords": [],
    "potential_questions": [],
    "uncertainty": [],
}

FIGURE_QUERY_CACHE_VERSION = 2


class FigureUnderstandingService:
    def __init__(self, settings, repository=None, client=None) -> None:
        self.settings = settings
        self.repository = repository
        self.client = client or QwenVLClient(settings)
        self.offline_call_count = 0
        self.query_call_count = 0
        self.judge_call_count = 0
        self.online_check_call_count = 0

    @property
    def interactive_call_count(self) -> int:
        """查询链路调用数，不包含建库和离线评测 Judge。"""
        return self.query_call_count + self.online_check_call_count

    def enrich_paper(self, paper) -> None:
        """建库阶段每张可读取 Figure 至多调用一次 Qwen-VL。"""
        for figure in paper.figures:
            if figure.kind != "picture" or not figure.image_path or figure.description:
                continue
            try:
                figure.description = self.describe(figure)
            except Exception as error:
                figure.description = {
                    **OFFLINE_FIELDS,
                    "status": "error",
                    "uncertainty": [f"Qwen-VL 离线理解失败：{type(error).__name__}: {error}"[:500]],
                }

    def describe(self, figure) -> dict:
        if not self.client.available:
            return {**OFFLINE_FIELDS, "status": "unavailable", "uncertainty": ["QWEN_VL_API_KEY 未配置。"]}
        prompt = f"""You are a scientific figure understanding assistant.

Your task is to analyze a figure extracted from an academic paper and produce a structured representation for retrieval and later evidence-grounded reasoning.

Section:
{figure.section}

Figure caption:
{figure.caption}

Nearby paper text:
{figure.nearby_text}

Rules:
1. Base the analysis primarily on the visible figure.
2. Use the caption and nearby text only as supporting context.
3. Do not invent information not visible in the figure or explicitly provided in the context.
4. Preserve important model names, module names, dataset names, metric names, labels, and numerical values.
5. Ignore decorative details that are not scientifically meaningful.
6. Focus on information useful for retrieval and future scientific reasoning.
7. If some content is unreadable or uncertain, explicitly record the uncertainty.
8. Do not make broad conclusions about the whole paper based only on this figure.
9. Return valid JSON only, with keys: figure_type, summary, entities, components, relations, visual_evidence, numerical_evidence, retrieval_keywords, potential_questions, uncertainty."""
        self.offline_call_count += 1
        result = self._normalize_offline(self.client.analyze(figure.image_path, prompt))
        result["image_preparation"] = self._image_preparation()
        return result

    def analyze_for_query(self, paper_id: str, figure, question: str) -> dict:
        normalized = self.normalize_query(question)
        if self.repository:
            cached = self.repository.get_figure_query_cache(
                paper_id,
                figure.id,
                normalized,
                cache_version=FIGURE_QUERY_CACHE_VERSION,
            )
            if cached is not None:
                return {
                    **cached,
                    "cached": True,
                    "cache_version": FIGURE_QUERY_CACHE_VERSION,
                }
        if not self.client.available:
            return {
                "relevant": True,
                "answerable": False,
                "figure_evidence": [],
                "interpretation": "",
                "missing_information": ["QWEN_VL_API_KEY 未配置，无法查看原始 Figure。"],
                "confidence": "low",
                "cached": False,
            }
        prompt = f"""User question:
{question}

Section:
{figure.section}

Figure caption:
{figure.caption}

Nearby paper text:
{figure.nearby_text}

You are analyzing a scientific figure as evidence for answering the user's question.
Inspect the original figure carefully. Focus only on the question. Use caption and nearby text only as supporting context. Do not use outside knowledge or invent details.

Mandatory visual inspection procedure:
1. Locate each label named in the question separately and copy its visible text.
2. Record the label's position (left/center/right and top/middle/bottom).
3. Identify the immediate rectangle or region belonging to that label. Do not transfer the color, border, or content of an adjacent box, parent container, legend, or arrow.
4. For color questions, inspect the fill inside each immediate rectangle, not its border or the surrounding group. Report each label-to-color mapping independently.
5. Compare all mappings once more before answering. If text, ownership, or color is ambiguous, lower confidence and state the ambiguity instead of guessing.
6. Distinguish figure, caption and nearby_text sources. If the figure cannot answer, set answerable=false. Preserve technical terms and numbers.

Return JSON only:
{{"relevant":true,"answerable":true,"visual_observations":[{{"label":"","position":"","immediate_region":"","fill_color":"","visible_basis":""}}],"figure_evidence":[{{"evidence":"","source":"figure | caption | nearby_text"}}],"interpretation":"","missing_information":[],"confidence":"high | medium | low"}}"""
        self.query_call_count += 1
        result = self._normalize_query_result(self.client.analyze(figure.image_path, prompt))
        result["image_preparation"] = self._image_preparation()
        if self.repository:
            self.repository.save_figure_query_cache(
                paper_id,
                figure.id,
                normalized,
                result,
                cache_version=FIGURE_QUERY_CACHE_VERSION,
            )
        return {
            **result,
            "cached": False,
            "cache_version": FIGURE_QUERY_CACHE_VERSION,
        }

    def judge_answer(
        self,
        figure,
        question: str,
        expected_answer: str,
        actual_answer: str,
    ) -> dict:
        """离线视觉裁判：独立查看原图，并以评测集标准答案评判回答。"""
        if not self.client.available:
            return {
                "score": None,
                "correct": None,
                "reason": "QWEN_VL_API_KEY 未配置，视觉 Judge 不可用。",
                "kind": "visual",
                "dataset_issue": None,
            }
        prompt = f"""You are an independent evaluator for visual scientific-paper QA.

Question:
{question}

Reference answer from the evaluation dataset:
{expected_answer}

Candidate answer:
{actual_answer}

Inspect the attached original figure yourself. Do not trust any prior figure analysis or claim-verification output. Locate every named label, record its position, identify its immediate box, and inspect the fill inside that box rather than a border, neighboring box, parent container, legend, or arrow.

Judge in this strict order:
1. Produce observed_answer and visual_observations from the image alone.
2. Decide whether the reference answer is supported by those observations.
3. Independently decide whether the candidate answer is supported.
4. Decide whether candidate and reference are semantically equivalent, allowing harmless wording and color-shade synonyms.
5. Set dataset_issue=true whenever the reference conflicts with the observed image. A correct candidate must not be scored as wrong merely because the reference is wrong.

Return JSON only:
{{"observed_answer":"image-only answer","reference_supported":true,"candidate_supported":true,"candidate_matches_reference":true,"dataset_issue":false,"score":2,"visual_observations":[{{"label":"","position":"","fill_color":"","visible_basis":""}}],"reason":"brief reason"}}
score must be 0, 1, or 2: 0=incorrect, 1=partially correct, 2=fully correct. When dataset_issue=true the score is diagnostic only and will be excluded from answer-accuracy metrics."""
        self.judge_call_count += 1
        try:
            value = self.client.analyze(figure.image_path, prompt)
            boolean_fields = (
                "reference_supported",
                "candidate_supported",
                "candidate_matches_reference",
                "dataset_issue",
            )
            if any(not isinstance(value.get(field), bool) for field in boolean_fields):
                raise ValueError("visual judge 缺少必要的布尔判定字段")
            score = value.get("score")
            if isinstance(score, bool) or score not in (0, 1, 2):
                raise ValueError("visual judge score 必须是 0、1、2")
            reference_supported = value["reference_supported"]
            candidate_supported = value["candidate_supported"]
            candidate_matches_reference = value["candidate_matches_reference"]
            dataset_issue = (
                value["dataset_issue"]
                or not reference_supported
                or (candidate_supported and not candidate_matches_reference)
            )
            if dataset_issue:
                score = None
            return {
                "score": score,
                "correct": (
                    None
                    if dataset_issue
                    else score == 2
                    and candidate_supported
                    and candidate_matches_reference
                ),
                "reference_supported": reference_supported,
                "image_supports_reference": reference_supported,
                "candidate_supported": candidate_supported,
                "candidate_matches_reference": candidate_matches_reference,
                "dataset_issue": dataset_issue,
                "observed_answer": str(value.get("observed_answer", ""))[:1000],
                "reason": str(value.get("reason", ""))[:500],
                "kind": "visual",
                "visual_observations": self._list(value, "visual_observations"),
                "image_preparation": self._image_preparation(),
            }
        except Exception as error:
            return {
                "score": None,
                "correct": None,
                "reason": f"视觉 Judge 失败：{type(error).__name__}: {error}"[:500],
                "kind": "visual",
                "dataset_issue": None,
            }

    def verify_answer_online(self, figure, question: str, actual_answer: str) -> dict:
        """可选在线二次看图核验；无 Gold，不等同于离线正确性评测。"""
        if not self.client.available:
            return {
                "supported": None,
                "reason": "QWEN_VL_API_KEY 未配置，在线视觉核验不可用。",
            }
        prompt = f"""You are an independent visual evidence verifier.

Question:
{question}

Candidate answer:
{actual_answer}

Inspect the attached original scientific figure. Do not trust any previous analysis. Locate each named label, record its position and immediate visual region, and inspect the region's fill rather than nearby boxes, borders, containers, legends, or arrows. Build an independent label-to-attribute mapping before comparing it with the candidate answer. Decide whether every material visual claim is directly supported by this image. Return JSON only:
{{"supported":true,"visual_observations":[{{"label":"","position":"","fill_color":"","visible_basis":""}}],"reason":"brief reason","unsupported_details":[]}}"""
        self.online_check_call_count += 1
        try:
            value = self.client.analyze(figure.image_path, prompt)
            return {
                "supported": value.get("supported") is True,
                "reason": str(value.get("reason", ""))[:500],
                "unsupported_details": (
                    value.get("unsupported_details", [])
                    if isinstance(value.get("unsupported_details"), list)
                    else []
                ),
                "visual_observations": self._list(value, "visual_observations"),
                "image_preparation": self._image_preparation(),
            }
        except Exception as error:
            return {
                "supported": None,
                "reason": f"在线视觉核验失败：{type(error).__name__}: {error}"[:500],
                "unsupported_details": [],
            }

    def normalize_query(self, question: str) -> str:
        return re.sub(r"\s+", " ", question.strip().lower())[:500]

    def _image_preparation(self) -> dict:
        value = getattr(self.client, "last_image_preparation", {})
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _list(value: dict, key: str) -> list:
        raw = value.get(key, [])
        return raw[:50] if isinstance(raw, list) else []

    def _normalize_offline(self, value: dict) -> dict:
        output = dict(OFFLINE_FIELDS)
        output["figure_type"] = str(value.get("figure_type", "other"))
        output["summary"] = str(value.get("summary", ""))[:4000]
        for key in OFFLINE_FIELDS:
            if isinstance(OFFLINE_FIELDS[key], list):
                raw = value.get(key, [])
                output[key] = raw[:50] if isinstance(raw, list) else []
        output["status"] = "ok"
        return output

    def _normalize_query_result(self, value: dict) -> dict:
        confidence = value.get("confidence", "low")
        return {
            "relevant": value.get("relevant") is True,
            "answerable": value.get("answerable") is True,
            "figure_evidence": value.get("figure_evidence", []) if isinstance(value.get("figure_evidence"), list) else [],
            "visual_observations": self._list(value, "visual_observations"),
            "interpretation": str(value.get("interpretation", ""))[:4000],
            "missing_information": value.get("missing_information", []) if isinstance(value.get("missing_information"), list) else [],
            "confidence": confidence if confidence in ("high", "medium", "low") else "low",
        }
