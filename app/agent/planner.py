"""复杂科研问题的任务分解和检索改写。"""

import re


class Planner:
    """生成有界、可执行的检索子任务。"""

    CLAIM = re.compile(
        r"\b(contributions?|claims?|methodological claims?|method claims?|"
        r"proposals?|novelt(?:y|ies))\b|贡献|主张|创新|方法论",
        re.I,
    )
    SUPPORT = re.compile(
        r"\b(support(?:ed|ing)?|experiments?|experimental|results?|evidence|"
        r"validate[sd]?|validation|ablation)\b|支撑|支持|实验|结果|证据|验证|消融",
        re.I,
    )
    STRUCTURED_REFERENCE = re.compile(
        r"\b(?:fig(?:ure)?s?|tables?)\.?\s*\d+"
        r"(?:\s*(?:,|and|&|to|-)\s*\d+)*|(?:图|表)\s*\d+",
        re.I,
    )

    def plan(self, question: str, modalities: list[str]) -> dict:
        tasks: list[dict] = []
        if self.CLAIM.search(question) and self.SUPPORT.search(question):
            experiment_modalities = ["text", "table"] if "table" in modalities else ["text"]
            ablation_modalities = ["table", "text"] if "table" in modalities else ["text"]
            tasks.extend([
                self._task(1, "Identify the paper's main methodological claims and claimed contributions.", ["text"]),
                self._task(2, "Identify the main experimental results relevant to those methodological claims.", experiment_modalities),
                self._task(3, "Identify comparisons or ablation evidence that tests individual design choices.", ablation_modalities),
            ])
        else:
            for modality in modalities:
                description = {
                    "text": f"Find method or result statements relevant to: {question}",
                    "figure": f"Find the original scientific figure relevant to: {question}",
                    "table": f"Find structured table evidence relevant to: {question}",
                }[modality]
                tasks.append(self._task(len(tasks) + 1, description, [modality]))
        return {"goal": question, "sub_tasks": tasks}

    def rewrite(self, description: str, question: str, attempt: int = 0) -> str:
        """把自然语言子任务压缩成保留术语、编号和数值的检索词。"""
        combined = f"{description} {question}"
        # Structured references must remain contiguous. Tokenizing first used to
        # turn ``Figure 1`` into ``figure In 1`` when a generic ``figure`` token
        # appeared earlier in the task description, disabling exact-ID filtering.
        references = [
            re.sub(r"\s+", " ", match.group(0)).strip()
            for match in self.STRUCTURED_REFERENCE.finditer(question)
        ]
        terms = re.findall(
            r"[A-Za-z][A-Za-z0-9_.-]*|\d+(?:\.\d+)?%?|[\u4e00-\u9fff]{2,6}",
            combined,
        )
        stop = {"the", "a", "an", "to", "for", "or", "and", "is", "are", "relevant", "find"}
        unique = []
        for term in [*references, *terms]:
            if term.lower() not in stop and term.lower() not in {item.lower() for item in unique}:
                unique.append(term)
        if attempt:
            unique.extend(["experimental evidence", "results", "ablation"])
        return " ".join(unique[:24])[:500]

    def _task(self, index: int, description: str, modalities: list[str]) -> dict:
        return {
            "id": f"task_{index}",
            "description": description,
            "preferred_modalities": modalities,
            "status": "pending",
        }
