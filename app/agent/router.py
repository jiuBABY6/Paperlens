"""查询复杂度与模态路由。"""

import re


class QueryRouter:
    """可解释的保守路由器；简单单事实查询继续走 Standard RAG。"""

    FIGURE = re.compile(
        # Use ASCII token boundaries for English Figure references. Python's
        # ``\b`` treats both digits and Chinese characters as ``\w``, so a
        # compact query such as ``Figure2有几种颜色`` otherwise fails to match.
        r"(?<![A-Za-z0-9_])fig(?:ure)?s?\.?\s*"
        r"(?:\d+(?:\s*(?:,|and|&|to|-)\s*\d+)*)?(?![A-Za-z0-9_])|"
        r"\b(?:diagrams?|architectures?|workflows?|plots?|charts?)\b|"
        r"图\s*\d+|图中|图示|图片中|图像中|架构图|流程图",
        re.I,
    )
    TABLE = re.compile(
        # Keep the same ASCII-boundary behavior as Figure references so
        # ``根据Table2，...`` is not mistaken for a text-only question.
        r"(?<![A-Za-z0-9_])tables?\.?\s*"
        r"(?:\d+(?:\s*(?:,|and|&|to|-)\s*\d+)*)?(?![A-Za-z0-9_])|"
        r"\b(?:tabular|rows?|columns?)\b|表\s*\d*|表格",
        re.I,
    )
    EXPLICIT_TEXT = re.compile(
        r"\b(method description|paper text|textual description|methodological claims?|"
        r"method rationale|design rationale|rationale|"
        r"claimed contributions?|authors? state|according to the text|"
        r"corpus description|dataset description|data description|"
        r"corpus split|train\s*/\s*dev\s*/\s*test split)\b|"
        r"方法描述|方法依据|设计依据|设计动机|正文|文本描述|语料描述|数据集描述|数据划分|"
        r"训练集.{0,8}(?:验证集|开发集).{0,8}测试集|方法主张|作者声称|核心贡献",
        re.I,
    )
    EXPERIMENTAL_CLAIM_CHECK = re.compile(
        r"\b(?:methodological\s+claims?|method\s+claims?|major\s+contributions?|"
        r"claims?|contributions?)\b.*\b(?:support(?:ed|ing)?|validate[sd]?|"
        r"experiments?|experimental\s+results?|evidence)\b|"
        r"\b(?:support(?:ed|ing)?|validate[sd]?|experiments?|experimental\s+results?)\b.*"
        r"\b(?:claims?|contributions?)\b|"
        r"(?:\u65b9\u6cd5|\u4e3b\u5f20|\u8d21\u732e).*(?:\u5b9e\u9a8c|\u7ed3\u679c|\u652f\u6301|\u9a8c\u8bc1)",
        re.I,
    )
    COMPLEX = re.compile(
        r"\b(are all|whether|compare|comparison|why|supported by|evidence for|"
        r"contribute most|most important|relationship|consistent|ablation|across|"
        r"methodological claims?|experimental support)\b|"
        r"是否|为什么|比较|对比|支撑|证据|贡献最大|消融|跨模态|分别",
        re.I,
    )

    def route(self, question: str) -> dict:
        needs_figure = bool(self.FIGURE.search(question))
        explicit_text = bool(self.EXPLICIT_TEXT.search(question))
        # Claim-versus-experiment questions need structured result evidence even
        # when the user does not explicitly name a table. Text-only retrieval
        # often finds "results are shown" but misses the reported measurements.
        implicit_result_table = bool(self.EXPERIMENTAL_CLAIM_CHECK.search(question))
        explicit_table = bool(self.TABLE.search(question))
        needs_table = explicit_table or implicit_result_table
        needs_text = bool(
            explicit_text
            or implicit_result_table
            or not (needs_figure or needs_table)
        )
        pure_visual = needs_figure and not needs_table and not needs_text
        pure_table = (
            explicit_table
            and not needs_figure
            and not needs_text
            and not implicit_result_table
        )
        if pure_visual:
            modalities = ["figure"]
        elif pure_table:
            modalities = ["table"]
        else:
            modalities = ["text"] if needs_text else []
        if needs_figure and "figure" not in modalities:
            modalities.append("figure")
        if needs_table and "table" not in modalities:
            modalities.append("table")
        required_modalities: list[str] = []
        if needs_text:
            required_modalities.append("text")
        if needs_figure:
            required_modalities.append("figure")
        if needs_table:
            required_modalities.append("table")
        complex_query = (
            any(item != "text" for item in modalities)
            or len(modalities) > 1
            or bool(self.COMPLEX.search(question))
        )
        return {
            "complexity": "complex" if complex_query else "simple",
            "route": "agentic_rag" if complex_query else "standard_rag",
            "modalities": modalities,
            "required_modalities": required_modalities,
            "pure_visual": pure_visual,
            "pure_table": pure_table,
            "reason": (
                "需要分解任务或读取非文本证据。"
                if complex_query else "单一文本事实检索足以回答。"
            ),
        }
