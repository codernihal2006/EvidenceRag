import re
from ..models.response import CitationValidation, CitationCheck
from ..utils.text import tokenize


def validate_citations(question: str, answer: str, evidence: list) -> CitationValidation:
    valid = {f"[{e.citation_id}]" for e in evidence}
    evidence_by_id = {f"[{e.citation_id}]": e.chunk.text for e in evidence}
    citations = re.findall(r"\[\d+\]", answer)
    checks = []
    unsupported = []
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip()) if answer.strip() else []
    for sentence in sentences:
        found = re.findall(r"\[\d+\]", sentence)
        if not found:
            if sentence and "couldn't find sufficient" not in sentence.lower() and "not configured" not in sentence.lower():
                unsupported.append(sentence)
            continue
        # This lightweight local check catches invalid IDs and claims with no
        # meaningful lexical overlap before an answer is shown. It is not a
        # substitute for a human or entailment-model review, but is reliable
        # and explainable for the local app.
        claim_terms = set(tokenize(sentence))
        cited_text = " ".join(evidence_by_id.get(c, "") for c in found)
        evidence_terms = set(tokenize(cited_text))
        overlap = len(claim_terms & evidence_terms) / max(1, len(claim_terms))
        is_supported = all(c in valid for c in found) and overlap >= 0.12
        checks.append(CitationCheck(citation=", ".join(found), claim=sentence, supported=is_supported))
        if not is_supported:
            unsupported.append(sentence)
    grounded = [c for c in checks if c.supported]
    score = len(grounded) / max(1, len(checks) + len(unsupported))
    return CitationValidation(supported=not unsupported and all(c in valid for c in citations), grounding_score=score,
        citations=checks, unsupported_claims=unsupported)
