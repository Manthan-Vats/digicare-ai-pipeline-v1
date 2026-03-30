"""
DigiCare pipeline — lab report ingestion and clinical analysis.

 lab report image or PDF---> extracts all test findings via Gemini Vision,
runs them through a clinical validation layer (domain rules for Widal, CRP,
liver enzymes, platelets, ESR, and urine parametersP)---->normalises to LOINC
concepts, stores in PostgreSQL with full provenance, and produces a structured
brief with red/green flags.

Dependencies: google-genai, Pillow, pymupdf, psycopg2-binary
Environment:  GEMINI_API_KEY (required), DATABASE_URL (optional)

    )
"""

import re
import io
import json
import time
import logging
from pathlib import Path
from typing import Optional, Union, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("digicare")


# PDF support

def pdf_to_images(pdf_path: Path) -> List[tuple]:
    """
    Opens a PDF and renders each page at 144 DPI (2x scale).
    Returns a list of (PIL Image, page_number) tuples, 1-indexed.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF support.\n"
            "Run: pip install pymupdf\n"
            "Then restart the script."
        )
    from PIL import Image as PILImage

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        raise ValueError(f"Cannot open PDF: {pdf_path} — {e}")

    images = []
    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom = 144 DPI
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat)
        img = PILImage.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append((img, page_num + 1))

    doc.close()
    log.info(f"PDF converted: {len(images)} pages from {pdf_path.name}")
    return images


# Extraction

class GeminiExtractor:
    """
    Sends each lab report image to Gemini 2.5 Flash and gets back a structured
    JSON array of findings. No intermediate OCR text, no regex parsing.

    Pricing (March 2026): $0.30/1M input tokens, $2.50/1M output tokens.
    Thinking is disabled — not useful for this kind of structured extraction.
    """

    EXTRACT_PROMPT = """You are a precise medical data extractor for Indian lab reports.

Extract ALL test results from this image into a JSON array.

Return ONLY a valid JSON array — no markdown, no backticks, no explanation.

Each element must have exactly these fields:
{
  "test_name_raw": "exact test name as printed in the report",
  "result_raw": "ONLY the actual result value (see rules below)",
  "unit": "unit exactly as printed, or null if not shown",
  "reference_range_raw": "reference range as printed, or null if absent",
  "is_abnormal_flag": "flag printed in report (H, L, HIGH, LOW, POSITIVE, REACTIVE, CRITICAL) or null",
  "evidence_text": "the exact line or region of text from the image that contains this result (copy verbatim)"
}

Rules:

RULE 1 — result_raw contains ONLY the result value, never the unit or reference range:
  "Haemoglobin 15.3 g/dL (11.5-14.5) L"           → result_raw: "15.3"
  "PLASMAGLUCOSE (P.P) 161 80-140 mg/dL H"         → result_raw: "161"   ← NOT 140
  "BUN 11 mg/dL 6-20"                               → result_raw: "11"    ← NOT 20
  "SODIUM 134 mmol/L 136-145 L"                     → result_raw: "134"   ← NOT 145
  "Salmonella typhi O  1:160  Significant Titre 1:80 or More" → result_raw: "1:160"
  "P.vivax TEST IS NON REACTIVE  NON REACTIVE"      → result_raw: "TEST IS NON REACTIVE"
  "Protein (Alb) Present(+)"                        → result_raw: "Present(+)"
  "Pus Cells  0 - 1  /H.P.F"                       → result_raw: "0-1"
  "Widal Conclusion WIDAL TEST IS POSITIVE"         → result_raw: "WIDAL TEST IS POSITIVE"

RULE 2 — Never infer, calculate, or generate values. Only extract what is explicitly printed.

RULE 3 — Graphical bar charts: read the value at the CURRENT pointer or marker position.
  The CURRENT result is the value shown at the active data point (the primary marker).
  A PRIOR result is a secondary marker showing a previous reading.
  If two result values are visible, extract BOTH:
    "TestName" for the current (primary) result
    "TestName - prior" for the prior (secondary) result.
  CRITICAL: Do NOT extract reference range boundary values (e.g. 197 or 771 that label
    the Low/High zones) as test results. Only extract actual data point values.
  Example (Vitamin B12 chart: current marker at 414.4, prior marker at 447.8,
           reference range boundaries at 197 [Low] and 771 [High]):
    {"test_name_raw": "Vitamin B12 (Cyanocobalamin)", "result_raw": "414.4", ...}
    {"test_name_raw": "Vitamin B12 (Cyanocobalamin) - prior", "result_raw": "447.8", ...}

RULE 4 — Preserve exact qualitative strings:
  "NON REACTIVE" NOT "NEGATIVE"
  "WIDAL TEST IS POSITIVE" NOT just "POSITIVE"
  "Present(+)" NOT "Present"
  "1:160" NOT "0.00625"
  "NIL" NOT "0"
  "ABSENT" NOT "Negative"

RULE 5 — Skip: patient name/ID, doctor name, lab name, report date, page numbers,
  report numbers, column headers (Test Name / Result / Reference Range / Unit),
  footer disclaimers, NABL accreditation text, and any line with no test result.

RULE 6 — Skip any parameter whose result value is "/" or "-" or blank (means not tested).

RULE 7 — For urine routine reports, include ALL parameters with actual values:
  Quantity, Colour, Appearance, Deposit, Specific Gravity, Reaction, Protein, Sugar,
  Ketone bodies, Blood, Bile Salt, Bile Pigment, Phosphates, Nitrite,
  Pus Cells, RBC, Epithelial Cells, Casts, Crystals, Yeast Cell, Parasite,
  Micro-organisms, Others. Use null for Ketone/PH/Chyle/Others if their value is "/".

Extract all test results now:"""

    PRICE_INPUT_PER_M  = 0.30   # $ per 1M input tokens (verified March 2026)
    PRICE_OUTPUT_PER_M = 2.50   # $ per 1M output tokens

    def __init__(self, api_key: str):
        try:
            from google import genai
            from google.genai import types
            self.client = genai.Client(api_key=api_key)
            self.types  = types
            self._available = True
            log.info("GeminiExtractor ready")
        except ImportError:
            log.error("google-genai not installed. Run: pip install google-genai")
            self._available = False

    def extract(
        self,
        image_input: Union[Path, object],   # Path or PIL Image
        source_id: str = "",
        page: int = 1,
        max_retries: int = 3,
    ) -> dict:
        """
        Run extraction on one image (Path or PIL Image from a PDF page).
        Retries up to max_retries times on 429 rate-limit responses.
        Returns findings list plus elapsed time, token counts, cost, and error.
        """
        if not self._available:
            return self._error_result("google-genai not installed")

        from PIL import Image as PILImage

        if isinstance(image_input, (str, Path)):
            image_path = Path(image_input)
            if image_path.suffix.lower() == ".pdf":
                return self._error_result(
                    "Pass PDF pages through pdf_to_images() — do not call extract() on a PDF directly."
                )
            img = PILImage.open(image_path).convert("RGB")
        else:
            img = image_input

        attempt = 0
        while attempt <= max_retries:
            start = time.time()
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[self.EXTRACT_PROMPT, img],
                    config=self.types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=4096,
                        response_mime_type="application/json",
                        thinking_config=self.types.ThinkingConfig(thinking_budget=0),
                    ),
                )

                elapsed = time.time() - start
                usage   = response.usage_metadata
                in_tok  = getattr(usage, "prompt_token_count",     0) or 0
                out_tok = getattr(usage, "candidates_token_count", 0) or 0
                cost    = (in_tok / 1_000_000 * self.PRICE_INPUT_PER_M
                           + out_tok / 1_000_000 * self.PRICE_OUTPUT_PER_M)

                raw = (response.text or "").strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw).strip()

                try:
                    items = json.loads(raw)
                    if isinstance(items, dict):
                        items = items.get("findings", list(items.values())[0] if items else [])
                    if not isinstance(items, list):
                        items = []
                except (json.JSONDecodeError, ValueError) as je:
                    log.error(f"JSON parse failed (page {page}): {je} | raw[:200]={raw[:200]!r}")
                    items = []

                findings = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    test_name = str(item.get("test_name_raw") or "").strip()
                    result    = str(item.get("result_raw")    or "").strip()
                    if not test_name or not result or result in ("/", "-", ""):
                        continue

                    finding = {
                        "test_name_raw":      test_name,
                        "result_raw":         result,
                        "result_type":        self._classify(result),
                        "numeric_value":      self._to_float(result),
                        "unit":               item.get("unit") or None,
                        "reference_range_raw": item.get("reference_range_raw") or None,
                        "reference_low":      None,
                        "reference_high":     None,
                        "is_abnormal_flag":   str(item.get("is_abnormal_flag") or "").strip() or None,
                        "evidence_text":       str(item.get("evidence_text") or "").strip(),
                        "source_report_id":   source_id,
                        "page_number":        page,
                        "extraction_method":  "gemini_vision_v2",
                        "confidence_score":   1.0,
                    }

                    ref = finding["reference_range_raw"]
                    if ref:
                        m = re.search(r"([\d.]+)\s*[-–]\s*([\d.]+)", ref)
                        if m:
                            finding["reference_low"]  = float(m.group(1))
                            finding["reference_high"] = float(m.group(2))
                        else:
                            m2 = re.search(r"<\s*([\d.]+)", ref)
                            if m2:
                                finding["reference_low"]  = 0.0
                                finding["reference_high"] = float(m2.group(1))
                            else:
                                m3 = re.search(r">\s*([\d.]+)", ref)
                                if m3:
                                    finding["reference_low"] = float(m3.group(1))

                    findings.append(finding)

                log.info(f"Extracted page {page}: {len(findings)} findings | "
                         f"{elapsed:.1f}s | ${cost:.5f} | {in_tok}in/{out_tok}out tokens")

                return {
                    "findings":    findings,
                    "elapsed_sec": round(elapsed, 2),
                    "tokens_in":   in_tok,
                    "tokens_out":  out_tok,
                    "cost_usd":    round(cost, 6),
                    "error":       None,
                }

            except Exception as e:
                elapsed   = time.time() - start
                error_str = str(e)
                is_rate   = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str

                if is_rate and attempt < max_retries:
                    wait = 5 * (2 ** attempt)  # 5s, 10s, 20s
                    log.warning(f"Rate limit (attempt {attempt+1}/{max_retries}). "
                                f"Waiting {wait}s...")
                    time.sleep(wait)
                    attempt += 1
                    continue

                log.error(f"Extraction failed page {page}: {error_str}")
                return self._error_result(error_str, elapsed=round(elapsed, 2))

        return self._error_result("Max retries exceeded")

    @staticmethod
    def _classify(value: str) -> str:
        if re.match(r"^\d+:\d+$", value):
            return "ratio"
        try:
            float(value.replace(",", ""))
            return "numeric"
        except (ValueError, AttributeError):
            return "qualitative"

    @staticmethod
    def _to_float(value: str) -> Optional[float]:
        try:
            return float(str(value).replace(",", ""))
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _error_result(msg: str, elapsed: float = 0.0) -> dict:
        return {
            "findings": [], "elapsed_sec": elapsed,
            "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
            "error": msg,
        }


# Validation

class ValidationLayer:
    """
    Flags abnormal findings and populates abnormal_reason with a plain-text
    clinical explanation.

    Evaluation order:
    - Lab's own H/L/POSITIVE flag (highest trust — from the report itself)
    - Domain rule for the specific test (Widal, CRP, AST/ALT, platelets, ESR, urine)
    - Numeric comparison against reference_low / reference_high from the report
    - Qualitative string matching (POSITIVE, NON REACTIVE, etc.)

    Domain rule thresholds:
    - Widal: >=1:80 significant (reads lab's own reference range; defaults to 1:80)
    - CRP: <5 normal, 5-20 mild, 20-100 moderate, >100 severe (WHO standard)
    - AST/ALT: ULN=40 U/L; >3x significant; >10x critical (NABL-aligned)
    - Platelets: <50 K/uL severe; <20 critical (WHO)
    - ESR: uses lab reference range; >100 mm/hr marked as critical
    - Urine protein/glucose: any presence = abnormal
    - Urine pus cells: >5/HPF pyuria; >30 severe
    - Urine RBC: >2/HPF haematuria; >25 critical
    """

    IMPOSSIBILITY_BOUNDS = {
        "HEMOGLOBIN":      (1.0,   25.0),
        "WBC":             (0.1,  200.0),
        "PLATELETS":       (1.0, 3000.0),
        "SODIUM":          (100.0, 180.0),
        "POTASSIUM":       (1.5,    9.0),
        "CREATININE":      (0.1,   30.0),
        "GLUCOSE_PP":      (20.0, 1000.0),
        "GLUCOSE_RANDOM":  (20.0, 1000.0),
        "BILIRUBIN_TOTAL": (0.0,   80.0),
        "CRP":             (0.0, 1000.0),
        "CALCIUM":         (4.0,   20.0),
        "VITAMIN_B12":     (10.0, 20000.0),
        "VITAMIN_D":       (1.0,  200.0),
    }

    ABNORMAL_POSITIVE = {
        "POSITIVE", "WIDAL TEST IS POSITIVE", "PRESENT (+)", "PRESENT",
        "PRESENT(+)", "REACTIVE", "GROWTH DETECTED",
    }
    ABNORMAL_NEGATIVE = {
        "NON REACTIVE", "NEGATIVE", "NO GROWTH IN CULTURE",
        "NO BACTERIAL GROWTH DETECTED", "ABSENT", "NIL", "TEST IS NON REACTIVE",
    }


    @staticmethod
    def _parse_widal_titre(result_raw: str) -> Optional[int]:
        """Parse '1:160' → 160, 'NIL' → 0, unparseable → None."""
        ru = str(result_raw).strip().upper()
        if ru in ("NIL", "NEGATIVE", "NO AGGLUTINATION", "ABSENT", "NON REACTIVE"):
            return 0
        m = re.match(r"1:(\d+)", ru)
        return int(m.group(1)) if m else None

    @staticmethod
    def _parse_widal_threshold(reference_range_raw: str, default: int = 80) -> int:
        """
        Parse significant threshold from lab's own reference notation.
        'Significant Titre 1:80 or More' → 80
        'NIL' or missing → use default (80).
        Source: most Indian labs use 1:80 as minimum significant titre.
        """
        if not reference_range_raw:
            return default
        m = re.search(r"1:(\d+)", str(reference_range_raw))
        return int(m.group(1)) if m else default

    def _rule_widal(self, f: dict) -> Optional[tuple]:
        """
        Widal titre interpretation.
        Returns (is_abnormal, direction, reason, is_critical) or None.
        Sources: PMC3616551 (Garhwal), PMC3919416 (Ahmednagar), Wikipedia Widal test.
        Threshold: use lab's own significant titre from reference_range_raw;
                   default 80 (1:80) — the most conservative widely-used cutoff.
        """
        titre = self._parse_widal_titre(f.get("result_raw", ""))
        if titre is None:
            return None  # non-parseable ratio — fall through to other checks
        threshold = self._parse_widal_threshold(f.get("reference_range_raw", ""))
        loinc = f.get("loinc_concept", "")
        test  = f.get("test_name_raw", loinc)

        if titre == 0:
            return (False, None,
                    f"Titre NIL — no significant agglutination detected for {test}", False)
        if titre >= threshold:
            is_critical = (titre >= threshold * 4)  # 4x threshold = very high
            return (
                True, "POSITIVE",
                f"Titre 1:{titre} meets or exceeds significant threshold 1:{threshold} for {test} "
                f"(per lab reference range; threshold source: report reference or default 1:80)",
                is_critical,
            )
        return (
            False, None,
            f"Titre 1:{titre} is below significant threshold 1:{threshold} for {test}",
            False,
        )

    def _rule_crp(self, f: dict) -> Optional[tuple]:
        """
        CRP level interpretation.
        Source: WHO/standard clinical biochemistry (universally accepted thresholds).
          <5 mg/L    — normal
          5-20 mg/L  — mild elevation (minor inflammation, viral illness)
          20-100 mg/L— moderate elevation (bacterial infection, significant inflammation)
          >100 mg/L  — severe elevation (severe bacterial infection, possible sepsis) — CRITICAL
        Unit handling: converts mg/dL to mg/L if unit contains 'dl'.
        """
        val = f.get("numeric_value")
        if val is None:
            return None
        unit_raw = (f.get("unit") or "").lower()
        # Convert if reported in mg/dL (some Indian labs) → mg/L
        val_mgL = val * 10 if ("dl" in unit_raw and "mg" in unit_raw) else val
        unit_display = f.get("unit") or "mg/L"

        if val_mgL < 5.0:
            return (False, None,
                    f"CRP {val} {unit_display} is within normal range (<5.0 mg/L)", False)
        if val_mgL < 20.0:
            return (True, "HIGH",
                    f"CRP {val} {unit_display} mildly elevated (5–20 mg/L: minor inflammation or viral illness)",
                    False)
        if val_mgL < 100.0:
            return (True, "HIGH",
                    f"CRP {val} {unit_display} moderately elevated (20–100 mg/L: active bacterial infection or significant inflammation)",
                    False)
        return (True, "HIGH",
                f"CRP {val} {unit_display} severely elevated (>100 mg/L: possible sepsis or major systemic disease — requires urgent clinical correlation)",
                True)  # critical

    def _rule_liver_enzyme(self, f: dict) -> Optional[tuple]:
        """
        AST/ALT (SGOT/SGPT) interpretation.
        Source: NABL-aligned Indian lab standard. ULN = 40 U/L for both AST and ALT.
          ≤40 U/L       — normal
          41–120 U/L  — mildly elevated (1–3x ULN)
          121–400 U/L — significantly elevated (3–10x ULN: active liver disease)
          >400 U/L     — critically elevated (>10x ULN: requires immediate evaluation)
        """
        val = f.get("numeric_value")
        if val is None:
            return None
        ULN = 40.0  # Upper limit of normal, U/L (NABL Indian standard)
        test = f.get("test_name_raw") or f.get("loinc_concept", "liver enzyme")
        unit = f.get("unit") or "U/L"
        fold = val / ULN

        if val <= ULN:
            return (False, None,
                    f"{test} {val} {unit} is within normal range (≤{int(ULN)} U/L)", False)
        if val <= ULN * 3:
            return (True, "HIGH",
                    f"{test} {val} {unit} mildly elevated ({fold:.1f}x ULN; clinically significant above 3x ULN={int(ULN*3)} U/L)",
                    False)
        if val <= ULN * 10:
            return (True, "HIGH",
                    f"{test} {val} {unit} significantly elevated ({fold:.1f}x ULN; indicates active liver disease)",
                    False)
        return (True, "HIGH",
                f"{test} {val} {unit} critically elevated ({fold:.1f}x ULN; >10x ULN requires immediate hepatological evaluation)",
                True)

    def _rule_platelets(self, f: dict) -> Optional[tuple]:
        """
        Platelet count critical thresholds.
        Source: WHO blood transfusion guidelines + standard haematology.
          >400 K/μL   — thrombocytosis
          150–400    — normal
          100–150    — mild thrombocytopenia
          50–100     — moderate thrombocytopenia
          20–50      — severe thrombocytopenia (active bleeding risk) — CRITICAL
          <20         — critical (life-threatening bleeding risk) — CRITICAL
        Unit note: 'Lakhs/cmm' is multiplied ×100 to convert to K/μL scale.
        """
        val = f.get("numeric_value")
        if val is None:
            return None
        unit_raw = (f.get("unit") or "").lower()
        # Convert Lakhs/cmm → K/μL equivalent: 1.5 Lakhs/cmm = 150 K/μL
        val_k = val * 100 if "lakh" in unit_raw else val
        unit_display = f.get("unit") or "K/μL"
        ref_low  = f.get("reference_low")  or 150
        ref_high = f.get("reference_high") or 400

        if val_k < 20:
            return (True, "LOW",
                    f"Platelet count {val} {unit_display} critically low (<20 K/μL — life-threatening bleeding risk)",
                    True)
        if val_k < 50:
            return (True, "LOW",
                    f"Platelet count {val} {unit_display} severely low (<50 K/μL — active bleeding risk; transfusion threshold)",
                    True)
        if val_k < 100:
            return (True, "LOW",
                    f"Platelet count {val} {unit_display} moderately low (50–100 K/μL — moderate thrombocytopenia)",
                    False)
        if val_k < ref_low:
            return (True, "LOW",
                    f"Platelet count {val} {unit_display} below reference low {ref_low} K/μL (mild thrombocytopenia)",
                    False)
        if val_k > ref_high:
            return (True, "HIGH",
                    f"Platelet count {val} {unit_display} above reference high {ref_high} K/μL (thrombocytosis)",
                    False)
        return None  # in normal range — let generic numeric check confirm

    def _rule_esr(self, f: dict) -> Optional[tuple]:
        """
        ESR (Erythrocyte Sedimentation Rate) interpretation.
        Source: WHO Westergren method reference ranges (standard globally including India).
          Male:   normal <15 mm/1st hr
          Female: normal <20 mm/1st hr
        Cannot determine sex from current data — use 20 mm/hr as conservative threshold.
        >100 mm/hr: markedly elevated (PMR, myeloma, severe infection).
        """
        val = f.get("numeric_value")
        if val is None:
            return None
        ref_high = f.get("reference_high") or 20  # default conservative
        unit = f.get("unit") or "mm/1st hr"

        if val > 100:
            return (True, "HIGH",
                    f"ESR {val} {unit} markedly elevated (>100 mm/1st hr: consider polymyalgia rheumatica, myeloma, or severe infection)",
                    True)
        if val > ref_high:
            return (True, "HIGH",
                    f"ESR {val} {unit} above reference high {ref_high} {unit} (elevated: infection, inflammation, or autoimmune disease)",
                    False)
        return (False, None, f"ESR {val} {unit} within reference range (≤{ref_high} {unit})", False)

    def _rule_urine_albumin(self, f: dict) -> Optional[tuple]:
        """
        Urine protein (albumin) interpretation.
        Source: Standard urinalysis interpretation.
        Any presence of protein in urine is abnormal (except physiological trace).
        """
        result = str(f.get("result_raw") or "").strip()
        ru = result.upper()
        NORMAL_SET = {"ABSENT", "NIL", "NEGATIVE", "-", "NORMAL", "NO", "TRACE (PHYSIOLOGICAL)"}
        ABNORMAL_KEYWORDS = ("PRESENT", "TRACE", "+", "1+", "2+", "3+", "4+")

        if ru in NORMAL_SET or ru in ("", "/"):
            return (False, None, f"Urine protein '{result}' — no proteinuria detected", False)
        if ru in ("TRACE",):
            return (True, "POSITIVE",
                    f"Urine protein '{result}' detected — trace proteinuria (may indicate early glomerular or tubular disease; requires clinical correlation)",
                    False)
        if any(kw in ru for kw in ABNORMAL_KEYWORDS):
            return (True, "POSITIVE",
                    f"Urine protein '{result}' detected — proteinuria present (possible renal disease, pre-eclampsia, or urinary tract infection)",
                    False)
        return None

    def _rule_urine_glucose(self, f: dict) -> Optional[tuple]:
        """
        Urine glucose interpretation.
        Source: Standard urinalysis. Any glucose in urine is abnormal.
        Causes: diabetes mellitus (most common), renal glycosuria, pregnancy.
        """
        result = str(f.get("result_raw") or "").strip()
        ru = result.upper()
        NORMAL_SET = {"ABSENT", "NIL", "NEGATIVE", "-", "NORMAL", "NO"}
        ABNORMAL_KEYWORDS = ("PRESENT", "POSITIVE", "+", "1+", "2+", "3+")

        if ru in NORMAL_SET or ru in ("", "/"):
            return (False, None, f"Urine glucose '{result}' — no glycosuria detected", False)
        if any(kw in ru for kw in ABNORMAL_KEYWORDS):
            return (True, "POSITIVE",
                    f"Urine glucose '{result}' detected — glycosuria present (most common cause: diabetes mellitus; also renal glycosuria or pregnancy)",
                    False)
        return None

    def _rule_urine_pus_cells(self, f: dict) -> Optional[tuple]:
        """
        Urine pus cells interpretation.
        Source: Standard urinalysis. Normal: 0–5/HPF. >5/HPF = pyuria.
        """
        result = str(f.get("result_raw") or "").strip()
        ru = result.upper()

        QUALITATIVE_ABNORMAL = {"NUMEROUS", "MANY", "PACKED", "TOO NUMEROUS TO COUNT", "TNTC"}
        if ru in QUALITATIVE_ABNORMAL:
            return (True, "HIGH",
                    f"Urine pus cells '{result}' — marked pyuria (indicates active urinary tract infection)",
                    True)

        # Parse numeric ranges like "0-1", "3-4", "6-8", "10-15", "0-2"
        m = re.search(r"(\d+)\s*[-–]\s*(\d+)", result)
        if m:
            upper = int(m.group(2))
            if upper > 30:
                return (True, "HIGH",
                        f"Urine pus cells {result}/HPF — severe pyuria (>30/HPF: active UTI or pyelonephritis)",
                        True)
            if upper > 5:
                return (True, "HIGH",
                        f"Urine pus cells {result}/HPF — pyuria (>5/HPF: urinary tract infection indicated)",
                        False)
            return (False, None,
                    f"Urine pus cells {result}/HPF — within normal range (0–5/HPF)", False)

        # Single number
        m2 = re.match(r"^(\d+)\s*$", result)
        if m2:
            n = int(m2.group(1))
            if n > 30:
                return (True, "HIGH",
                        f"Urine pus cells {n}/HPF — severe pyuria (active UTI)", True)
            if n > 5:
                return (True, "HIGH",
                        f"Urine pus cells {n}/HPF — pyuria (>5/HPF: urinary tract infection indicated)", False)
            return (False, None, f"Urine pus cells {n}/HPF — within normal range", False)
        return None

    def _rule_urine_rbc(self, f: dict) -> Optional[tuple]:
        """
        Urine RBC interpretation.
        Source: Standard urinalysis. Normal: 0–2/HPF. >2/HPF = haematuria.
        """
        result = str(f.get("result_raw") or "").strip()
        m = re.search(r"(\d+)\s*[-–]\s*(\d+)", result)
        if m:
            upper = int(m.group(2))
            if upper > 2:
                is_crit = upper > 25
                return (True, "HIGH",
                        f"Urine RBC {result}/HPF — haematuria (>2/HPF: consider UTI, renal calculi, or glomerulonephritis)",
                        is_crit)
            return (False, None, f"Urine RBC {result}/HPF — within normal range (0–2/HPF)", False)
        m2 = re.match(r"^(\d+)\s*$", result)
        if m2:
            n = int(m2.group(1))
            if n > 2:
                return (True, "HIGH",
                        f"Urine RBC {n}/HPF — haematuria (>2/HPF: requires investigation)", n > 25)
            return (False, None, f"Urine RBC {n}/HPF — within normal range", False)
        return None


    _WIDAL_LOINCS = {
        "WIDAL_TYPHI_O", "WIDAL_TYPHI_H",
        "WIDAL_PARA_A_O", "WIDAL_PARA_A_H",
        "WIDAL_PARA_B_O", "WIDAL_PARA_B_H",
    }

    def _apply_domain_rule(self, f: dict) -> Optional[tuple]:
        """
        Dispatch to the appropriate domain-specific rule based on loinc_concept.
        Returns (is_abnormal, direction, reason, is_critical) or None.
        """
        loinc = f.get("loinc_concept", "")
        if loinc in self._WIDAL_LOINCS:
            return self._rule_widal(f)
        if loinc == "CRP":
            return self._rule_crp(f)
        if loinc in ("AST", "ALT"):
            return self._rule_liver_enzyme(f)
        if loinc == "PLATELETS":
            return self._rule_platelets(f)
        if loinc == "ESR":
            return self._rule_esr(f)
        if loinc == "URINE_ALBUMIN":
            return self._rule_urine_albumin(f)
        if loinc == "URINE_GLUCOSE":
            return self._rule_urine_glucose(f)
        if loinc == "URINE_PUS_CELLS":
            return self._rule_urine_pus_cells(f)
        if loinc == "URINE_RBC":
            return self._rule_urine_rbc(f)
        return None


    def validate(self, findings: list) -> list:
        validated = []
        for f in findings:
            f = f.copy()
            flags = []

            # Ensure numeric_value and result_type are populated
            if f.get("numeric_value") is None and f.get("result_raw"):
                f["numeric_value"] = GeminiExtractor._to_float(f["result_raw"])
            if not f.get("result_type"):
                f["result_type"] = GeminiExtractor._classify(str(f.get("result_raw", "")))

            f["is_abnormal"]        = False
            f["is_critical"]        = False
            f["abnormal_direction"] = None
            f["abnormal_reason"]    = None

            flag = str(f.get("is_abnormal_flag") or "").upper().strip()
            if flag in ("H", "HIGH", "CRITICAL", "ABOVE"):
                f["is_abnormal"]        = True
                f["abnormal_direction"] = "HIGH"
                f["abnormal_reason"]    = f"Flagged as HIGH by reporting laboratory (printed flag: {flag})"
            elif flag in ("L", "LOW", "BELOW"):
                f["is_abnormal"]        = True
                f["abnormal_direction"] = "LOW"
                f["abnormal_reason"]    = f"Flagged as LOW by reporting laboratory (printed flag: {flag})"
            elif flag in ("POSITIVE", "REACTIVE"):
                f["is_abnormal"]        = True
                f["abnormal_direction"] = "POSITIVE"
                f["abnormal_reason"]    = f"Flagged as POSITIVE/REACTIVE by reporting laboratory (printed flag: {flag})"
            elif flag in ("NEGATIVE", "NON REACTIVE"):
                f["is_abnormal"] = False

                    # Always run domain rule to populate reason text, even if already flagged
            domain = self._apply_domain_rule(f)
            if domain is not None:
                dom_abn, dom_dir, dom_reason, dom_crit = domain
                if not f["is_abnormal"]:
                    f["is_abnormal"]        = dom_abn
                    f["abnormal_direction"] = dom_dir
                    f["is_critical"]        = dom_crit
                # Always use domain rule reason (more informative than doc flag alone)
                f["abnormal_reason"] = dom_reason

            val      = f.get("numeric_value")
            ref_low  = f.get("reference_low")
            ref_high = f.get("reference_high")

            if val is not None and not f["is_abnormal"]:
                if ref_low is not None and val < ref_low:
                    f["is_abnormal"]        = True
                    f["abnormal_direction"] = "LOW"
                    f["abnormal_reason"]    = (
                        f"Value {val} {f.get('unit') or ''} is below reference low {ref_low} "
                        f"(from lab report reference range)"
                    )
                elif ref_high is not None and val > ref_high:
                    f["is_abnormal"]        = True
                    f["abnormal_direction"] = "HIGH"
                    f["abnormal_reason"]    = (
                        f"Value {val} {f.get('unit') or ''} is above reference high {ref_high} "
                        f"(from lab report reference range)"
                    )

            # Generic critical threshold: 3x above upper or 30% below lower ref
            if val is not None and f["is_abnormal"] and not f["is_critical"]:
                if ref_high and val > ref_high * 3:
                    f["is_critical"] = True
                if ref_low and ref_low > 0 and val < ref_low * 0.3:
                    f["is_critical"] = True

                    loinc = f.get("loinc_concept", "")
            if val is not None and loinc in self.IMPOSSIBILITY_BOUNDS:
                lo, hi = self.IMPOSSIBILITY_BOUNDS[loinc]
                if not (lo <= val <= hi):
                    flags.append(
                        f"IMPOSSIBLE_VALUE: {val} outside physiological range ({lo}–{hi})"
                    )
                    f["confidence_score"] = 0.0
                    log.warning(f"IMPOSSIBLE_VALUE: {f['test_name_raw']} = {val} [{loinc}]")

                    result_upper = str(f.get("result_raw", "")).upper()
            if result_upper in self.ABNORMAL_POSITIVE and not f["is_abnormal"]:
                f["is_abnormal"]        = True
                f["abnormal_direction"] = "POSITIVE"
                f["abnormal_reason"]    = (
                    f"Result '{f.get('result_raw')}' indicates a positive/reactive finding "
                    f"(clinically significant)"
                )
            elif result_upper in self.ABNORMAL_NEGATIVE:
                f["is_abnormal"] = False
                if not f["abnormal_reason"]:
                    f["abnormal_reason"] = None

            f["validation_flags"] = flags
            validated.append(f)

        abnormal = sum(1 for f in validated if f.get("is_abnormal"))
        critical = sum(1 for f in validated if f.get("is_critical"))
        log.info(f"Validation: {len(validated)} findings | {abnormal} abnormal | {critical} critical")
        return validated

class LoincNormalizer:
    """
    Maps raw Indian lab test names to internal LOINC concept codes.

    Matching is case-insensitive. Method suffixes like "- IFCC" or "- Urease"
    are stripped before lookup. Partial matching is sorted longest-key-first so
    specific keys (e.g. "mean corpuscular hemoglobin concentration") beat short
    ones (e.g. "hemoglobin") — this prevents MCH/MCHC from mapping to HEMOGLOBIN.

    Single-char directional suffixes like "- H" and "- O" (Widal) are preserved
    by the suffix-strip regex, which requires the stripped word to be 3+ chars.
    """

    LOINC_MAP = {
            "haemoglobin":                              "HEMOGLOBIN",
        "hemoglobin":                               "HEMOGLOBIN",
        "hb":                                       "HEMOGLOBIN",
        "hgb":                                      "HEMOGLOBIN",
        "wbc":                                      "WBC",
        "white blood cell":                         "WBC",
        "total leukocyte":                          "WBC",
        "total w.b.c":                              "WBC",
        "total wbc":                                "WBC",
        "tlc":                                      "WBC",
        "rbc count":                                "RBC",
        "red blood cell":                           "RBC",
        "red cell count":                           "RBC",
        "total r.b.c":                              "RBC",
        # Explicit long keys for MCH/MCHC (prevents hemoglobin partial match)
        "mean corpuscular hemoglobin concentration": "MCHC",
        "mean corpuscular hemoglobin conc":         "MCHC",
        "mean corpuscular hb conc":                 "MCHC",
        "m.c.h.c":                                  "MCHC",
        "mchc":                                     "MCHC",
        "mean corpuscular hemoglobin":              "MCH",
        "mean corpuscular h":                       "MCH",
        "m.c.h":                                    "MCH",
        "mch":                                      "MCH",
        "haematocrit":                              "HEMATOCRIT",
        "hematocrit":                               "HEMATOCRIT",
        "pcv":                                      "HEMATOCRIT",
        "h.c.t":                                    "HEMATOCRIT",
        "mean corpuscular volume":                  "MCV",
        "mcv":                                      "MCV",
        "m.c.v":                                    "MCV",
        "rdw":                                      "RDW",
        "red cell distribution width":              "RDW",
        "r.d.w":                                    "RDW",
        "platelet count":                           "PLATELETS",
        "platelet":                                 "PLATELETS",
        "plt":                                      "PLATELETS",
        "mean platelet volume":                     "MPV",
        "mpv":                                      "MPV",
        "m.p.v":                                    "MPV",
        "platelet distribution width":              "PDW",
        "pdw":                                      "PDW",
        "neutrophil":                               "NEUTROPHILS_PCT",
        "polymorphs":                               "NEUTROPHILS_PCT",
        "lymphocyte":                               "LYMPHOCYTES_PCT",
        "monocyte":                                 "MONOCYTES_PCT",
        "eosinophil":                               "EOSINOPHILS_PCT",
        "basophil":                                 "BASOPHILS_PCT",
        "absolute neutrophil":                      "ANC",
        "neutrophils absolute":                     "ANC",
        "absolute lymphocyte":                      "ALC",
        "lymphocytes absolute":                     "ALC",
        "absolute monocyte":                        "AMC",
        "monocytes absolute":                       "AMC",
        "eosinophils absolute":                     "AEC",
        "basophils absolute":                       "ABC",
        "esr":                                      "ESR",
        "erythrocyte sedimentation":                "ESR",
        "aptt control":                             "APTT_CONTROL",
        "aptt":                                     "APTT",
        "activated partial thromboplastin":         "APTT",
        "prothrombin time":                         "PT",
        "inr value":                                "INR",
        "inr":                                      "INR",
        "pt ratio":                                 "PT_RATIO",
        "rbc morphology":                           "RBC_MORPHOLOGY",
        "wbc morphology":                           "WBC_MORPHOLOGY",
        "plateletcrit":                             "PCT",
        "plateletocrit":                            "PCT",
        "p-lcr":                                    "P_LCR",
        "immature platelet":                        "IPF",
        "platelets on smear":                       "PLATELETS_SMEAR",
        "mentzer index":                            "MENTZER_INDEX",

            "crp":                                      "CRP",
        "c-reactive protein":                       "CRP",
        "plasma glucose (post-prandial)":           "GLUCOSE_PP",
        "plasma glucose (pp)":                      "GLUCOSE_PP",
        "plasma glucose (p.p)":                     "GLUCOSE_PP",
        "post prandial":                            "GLUCOSE_PP",
        "plasma glucose (r)":                       "GLUCOSE_RANDOM",
        "plasma glucose (random)":                  "GLUCOSE_RANDOM",
        "random blood sugar":                       "GLUCOSE_RANDOM",
        "glucose":                                  "GLUCOSE_RANDOM",
        "plasma glucose":                           "GLUCOSE_PP",
        "bun":                                      "BUN",
        "blood urea nitrogen":                      "BUN",
        "blood urea":                               "UREA",
        "urea":                                     "UREA",
        "creatinine":                               "CREATININE",
        "egfr":                                     "eGFR",
        "glomerular filtration":                    "eGFR",
        "sodium":                                   "SODIUM",
        "potassium":                                "POTASSIUM",
        "chloride":                                 "CHLORIDE",
        "bicarbonate":                              "BICARBONATE",
        "calcium":                                  "CALCIUM",
        "phosphorus":                               "PHOSPHORUS",
        "magnesium":                                "MAGNESIUM",
        "uric acid":                                "URIC_ACID",
        "total protein":                            "TOTAL_PROTEIN",
        "albumin":                                  "ALBUMIN",
        "globulin":                                 "GLOBULIN",
        "ag ratio":                                 "AG_RATIO",
        "bilirubin total":                          "BILIRUBIN_TOTAL",
        "bilirubin direct":                         "BILIRUBIN_DIRECT",
        "bilirubin indirect":                       "BILIRUBIN_INDIRECT",
        "s.g.o.t":                                  "AST",    # dot-notation fix
        "sgot":                                     "AST",
        "ast":                                      "AST",
        "s.g.p.t":                                  "ALT",    # dot-notation fix
        "sgpt":                                     "ALT",
        "alt":                                      "ALT",
        "alp":                                      "ALP",
        "alkaline phosphatase":                     "ALP",
        "ggt":                                      "GGT",
        "gamma glutamyl":                           "GGT",
        "ldh":                                      "LDH",
        "lactate dehydrogenase":                    "LDH",
        "amylase":                                  "AMYLASE",
        "lipase":                                   "LIPASE",
        "cholesterol":                              "CHOLESTEROL_TOTAL",
        "triglyceride":                             "TRIGLYCERIDES",
        "hdl":                                      "HDL",
        "ldl":                                      "LDL",
        "vldl":                                     "VLDL",
        "tsh":                                      "THYROID",
        "thyroid stimulating":                      "THYROID",
        "free t3":                                  "FREE_T3",
        "ft3":                                      "FREE_T3",
        "free t4":                                  "FREE_T4",
        "ft4":                                      "FREE_T4",
        "t3":                                       "T3",
        "t4":                                       "T4",
        "vitamin b12":                              "VITAMIN_B12",
        "cyanocobalamin":                           "VITAMIN_B12",
        "25-oh vitamin d":                          "VITAMIN_D",
        "vitamin d 25-hydroxy":                     "VITAMIN_D",
        "vitamin d":                                "VITAMIN_D",
        "ferritin":                                 "FERRITIN",
        "iron":                                     "IRON",
        "tibc":                                     "TIBC",
        "transferrin saturation":                   "TRANSFERRIN_SAT",
        "hba1c":                                    "HBA1C",
        "glycated haemoglobin":                     "HBA1C",
        "glycated hemoglobin":                      "HBA1C",

            # Widal: H = Flagellar, O = Somatic — must preserve distinction
        "salmonella typhi - o":                     "WIDAL_TYPHI_O",
        "typhi o":                                  "WIDAL_TYPHI_O",
        "salmonella typhi - h":                     "WIDAL_TYPHI_H",
        "typhi h":                                  "WIDAL_TYPHI_H",
        "salmonella paratyphi a - o":               "WIDAL_PARA_A_O",
        "paratyphi a - o":                          "WIDAL_PARA_A_O",
        "salmonella paratyphi a - h":               "WIDAL_PARA_A_H",   # ADDED
        "paratyphi a - h":                          "WIDAL_PARA_A_H",   # ADDED
        "salmonella paratyphi b - o":               "WIDAL_PARA_B_O",
        "paratyphi b - o":                          "WIDAL_PARA_B_O",
        "salmonella paratyphi b - h":               "WIDAL_PARA_B_H",   # ADDED
        "paratyphi b - h":                          "WIDAL_PARA_B_H",   # ADDED
        "widal conclusion":                         "WIDAL_CONCLUSION",
        "widal test":                               "WIDAL_CONCLUSION",
        "p. vivax":                                 "MALARIA_VIVAX_AG",
        "malaria vivax":                            "MALARIA_VIVAX_AG",
        "p. falciparum":                            "MALARIA_FALCIPARUM_AG",
        "p. falci":                                 "MALARIA_FALCIPARUM_AG",
        "malaria falciparum":                       "MALARIA_FALCIPARUM_AG",
        "dengue ns1":                               "DENGUE_NS1",
        "dengue igg":                               "DENGUE_IGG",
        "dengue igm":                               "DENGUE_IGM",
        "covid":                                    "COVID19_AG",
        "sars-cov-2":                               "COVID19_AG",
        "blood culture":                            "BLOOD_CULTURE",
        "urine culture":                            "URINE_CULTURE",

            "colour":                                   "URINE_COLOR",
        "color":                                    "URINE_COLOR",
        "appearance":                               "URINE_APPEARANCE",
        "specific gravity":                         "URINE_SG",
        "reaction":                                 "URINE_PH",
        "protein":                                  "URINE_ALBUMIN",
        "sugar":                                    "URINE_GLUCOSE",
        "ketone":                                   "URINE_KETONES",
        "blood":                                    "URINE_BLOOD",
        "bile salt":                                "URINE_BILE_SALTS",
        "bile pigment":                             "URINE_BILE_PIGMENTS",
        "pus cells":                                "URINE_PUS_CELLS",
        "r.b.c":                                    "URINE_RBC",
        "epithelial cells":                         "URINE_EPITHELIAL",
        "casts":                                    "URINE_CASTS",
        "crystals":                                 "URINE_CRYSTALS",
        "yeast":                                    "URINE_YEAST",
        "parasite":                                 "URINE_PARASITE",
        "micro-organisms":                          "URINE_MICROORGANISMS",
        "bacteria":                                 "URINE_BACTERIA",
        "others":                                   "URINE_OTHER",
        "urobilinogen":                             "URINE_UROBILINOGEN",
        "chyle":                                    "URINE_CHYLE",
        "deposit":                                  "URINE_DEPOSIT",
        "quantity":                                 "URINE_VOLUME",
        "nitrite":                                  "URINE_NITRITE",
        "phosphates":                               "URINE_PHOSPHATES",
    }

    MIN_PARTIAL_MATCH_LEN = 5

    def normalize(self, findings: list) -> list:
        normalized = []
        for f in findings:
            f = f.copy()
            f["loinc_concept"] = self._lookup(f.get("test_name_raw", ""))
            normalized.append(f)
        matched = sum(1 for f in normalized if f.get("loinc_concept"))
        log.info(f"LOINC: {matched}/{len(normalized)} findings normalized")
        return normalized

    def _lookup(self, test_name_raw: str) -> Optional[str]:
        """Case-insensitive lookup with method-suffix stripping."""
        # Step 1: strip em/en-dash method suffixes
        s = re.sub(r"\s*[—–]\s*.+$", "", test_name_raw).strip()
        # Step 2: strip hyphen-space + 3+-char-word method suffix
        s = re.sub(r"\s+-\s+[A-Za-z]{3}[A-Za-z\s,/]*$", "", s).strip()
        # Step 3: strip trailing parenthetical (like "(MCH)", "(SGOT)")
        s = re.sub(r"\s*\([^)]+\)\s*$", "", s).strip()
        normalized = s.lower().strip()

        if normalized in self.LOINC_MAP:
            return self.LOINC_MAP[normalized]

        for key in sorted(self.LOINC_MAP, key=len, reverse=True):
            if key in normalized:
                return self.LOINC_MAP[key]

        if len(normalized) >= self.MIN_PARTIAL_MATCH_LEN:
            for key in sorted(self.LOINC_MAP, key=len, reverse=True):
                if normalized in key:
                    return self.LOINC_MAP[key]

        return None


# Storage

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS digicare_findings (
    id                  BIGSERIAL PRIMARY KEY,
    patient_id          VARCHAR(64)   NOT NULL,
    report_date         DATE          NOT NULL,
    inserted_at         TIMESTAMPTZ   DEFAULT NOW(),
    loinc_concept       VARCHAR(64),
    test_name_raw       TEXT          NOT NULL,
    result_raw          TEXT,
    result_type         VARCHAR(20),
    numeric_value       NUMERIC(12,4),
    unit                VARCHAR(32),
    reference_low       NUMERIC(12,4),
    reference_high      NUMERIC(12,4),
    reference_range_raw TEXT,
    is_abnormal         BOOLEAN       DEFAULT FALSE,
    is_critical         BOOLEAN       DEFAULT FALSE,
    abnormal_direction  VARCHAR(16),
    source_report_id    VARCHAR(128)  NOT NULL,
    page_number         INTEGER       DEFAULT 1,
    extraction_method   VARCHAR(32),
    confidence_score    NUMERIC(4,3)  DEFAULT 1.0,
    validation_flags    TEXT,
    is_abnormal_flag    VARCHAR(32),
    evidence_text       TEXT,
    abnormal_reason     TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_digicare_findings_uq
    ON digicare_findings(
        patient_id,
        report_date,
        COALESCE(loinc_concept, test_name_raw),
        source_report_id,
        page_number
    );

CREATE INDEX IF NOT EXISTS idx_digicare_patient    ON digicare_findings(patient_id);
CREATE INDEX IF NOT EXISTS idx_digicare_loinc      ON digicare_findings(loinc_concept);
CREATE INDEX IF NOT EXISTS idx_digicare_date       ON digicare_findings(report_date);
CREATE INDEX IF NOT EXISTS idx_digicare_abnormal   ON digicare_findings(patient_id, is_abnormal);

CREATE TABLE IF NOT EXISTS digicare_reports (
    report_id           VARCHAR(128)  PRIMARY KEY,
    patient_id          VARCHAR(64)   NOT NULL,
    report_date         DATE,
    source_file         TEXT,
    findings_count      INTEGER       DEFAULT 0,
    total_cost_usd      NUMERIC(10,6),
    elapsed_sec         NUMERIC(8,2),
    inserted_at         TIMESTAMPTZ   DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_digicare_reports_patient ON digicare_reports(patient_id);
"""


class StorageLayer:
    """
    PostgreSQL storage. Uses a ThreadedConnectionPool (1-10 connections) so
    concurrent FastAPI requests each get their own connection. Findings are
    batch-inserted with execute_values in a single round-trip.
    """

    def __init__(self, db_url: str = ""):
        self._pool = None
        if db_url:
            try:
                import psycopg2
                from psycopg2 import pool
                self._pool = pool.ThreadedConnectionPool(1, 10, db_url)
                log.info("StorageLayer: connected")
            except Exception as e:
                log.error(f"DB pool failed: {e}")

    def create_schema(self):
        if not self._pool:
            return
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
                # Add new columns to existing tables (safe on repeated runs)
                for alter_sql in [
                    "ALTER TABLE digicare_findings ADD COLUMN IF NOT EXISTS abnormal_reason TEXT",
                ]:
                    try:
                        cur.execute(alter_sql)
                    except Exception as alter_err:
                        log.debug(f"ALTER TABLE skipped (column may already exist): {alter_err}")
            conn.commit()
            log.info("StorageLayer: schema created/verified (v3.0 — abnormal_reason column ensured)")
        except Exception as e:
            log.error(f"Schema creation failed: {e}")
            conn.rollback()
        finally:
            self._pool.putconn(conn)

    def save_report(self, report_id: str, patient_id: str, report_date: str,
                    findings_count: int = 0, source_file: str = "",
                    total_cost: float = 0.0, elapsed: float = 0.0):
        if not self._pool:
            return
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO digicare_reports
                       (report_id, patient_id, report_date, source_file,
                        findings_count, total_cost_usd, elapsed_sec)
                       VALUES (%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (report_id) DO UPDATE
                       SET findings_count=EXCLUDED.findings_count,
                           total_cost_usd=EXCLUDED.total_cost_usd""",
                    (report_id, patient_id, report_date, source_file,
                     findings_count, total_cost, elapsed)
                )
            conn.commit()
        except Exception as e:
            log.error(f"save_report failed: {e}")
            conn.rollback()
        finally:
            self._pool.putconn(conn)

    def save_findings(self, findings: list, patient_id: str, report_date: str) -> int:
        """Batch insert all findings in a single DB round-trip."""
        if not findings or not self._pool:
            return 0

        try:
            from psycopg2 import extras
        except ImportError:
            log.error("psycopg2 not installed")
            return 0

        rows = []
        for f in findings:
            rows.append((
                patient_id,
                report_date,
                f.get("loinc_concept"),
                f.get("test_name_raw", ""),
                f.get("result_raw"),
                f.get("result_type"),
                f.get("numeric_value"),
                f.get("unit"),
                f.get("reference_low"),
                f.get("reference_high"),
                f.get("reference_range_raw"),
                bool(f.get("is_abnormal", False)),
                bool(f.get("is_critical", False)),
                f.get("abnormal_direction"),
                f.get("source_report_id", ""),
                f.get("page_number", 1),
                f.get("extraction_method", "gemini_vision_v2"),
                f.get("confidence_score", 1.0),
                json.dumps(f.get("validation_flags", [])),
                f.get("is_abnormal_flag"),
                f.get("evidence_text", ""),
                f.get("abnormal_reason"),
            ))

        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                extras.execute_values(cur, """
                    INSERT INTO digicare_findings (
                        patient_id, report_date, loinc_concept, test_name_raw,
                        result_raw, result_type, numeric_value, unit,
                        reference_low, reference_high, reference_range_raw,
                        is_abnormal, is_critical, abnormal_direction,
                        source_report_id, page_number,
                        extraction_method, confidence_score, validation_flags,
                        is_abnormal_flag, evidence_text,
                        abnormal_reason
                    ) VALUES %s
                    ON CONFLICT (
                        patient_id,
                        report_date,
                        COALESCE(loinc_concept, test_name_raw),
                        source_report_id,
                        page_number
                    )
                    DO UPDATE SET
                        result_raw        = EXCLUDED.result_raw,
                        numeric_value     = EXCLUDED.numeric_value,
                        unit              = EXCLUDED.unit,
                        is_abnormal       = EXCLUDED.is_abnormal,
                        is_critical       = EXCLUDED.is_critical,
                        abnormal_direction= EXCLUDED.abnormal_direction,
                        confidence_score  = EXCLUDED.confidence_score,
                        is_abnormal_flag  = EXCLUDED.is_abnormal_flag,
                        evidence_text     = EXCLUDED.evidence_text,
                        abnormal_reason   = EXCLUDED.abnormal_reason,
                        extraction_method = EXCLUDED.extraction_method
                """, rows)
            conn.commit()
            log.info(f"Saved {len(rows)} findings for patient={patient_id}")
            return len(rows)
        except Exception as e:
            log.error(f"save_findings failed: {e}")
            conn.rollback()
            return 0
        finally:
            self._pool.putconn(conn)

    def get_patient_findings(self, patient_id: str,
                             loinc_concept: Optional[str] = None) -> list:
        if not self._pool:
            return []
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                if loinc_concept:
                    cur.execute(
                        "SELECT * FROM digicare_findings WHERE patient_id=%s "
                        "AND loinc_concept=%s ORDER BY report_date",
                        (patient_id, loinc_concept)
                    )
                else:
                    cur.execute(
                        "SELECT * FROM digicare_findings WHERE patient_id=%s "
                        "ORDER BY report_date, loinc_concept",
                        (patient_id,)
                    )
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
        except Exception as e:
            log.error(f"get_patient_findings failed: {e}")
            return []
        finally:
            self._pool.putconn(conn)


# Clinical synthesis

class ClinicalSynthesizer:
    """
    Sends structured findings to Gemini and gets back a formatted clinical brief.
    Gemini only receives values already extracted from the document — it cannot
    generate or infer clinical numbers.
    """

    SYNTHESIS_PROMPT = """You are a clinical AI assistant generating a structured patient brief for a doctor in India.

You will be given structured lab findings extracted from the patient's medical reports.
Each finding includes: test name, result, unit, abnormal status, clinical reasoning, and source report.

Rules:
1. ONLY use the values provided in the findings JSON. Do NOT generate or calculate any values.
2. Every clinical claim MUST cite the source_report_id in [Source: ...] format.
3. Include the abnormal_reason for each red flag — this is the clinical explanation.
4. Never add diagnoses. Describe findings only. Never say "the patient has X disease."
5. Keep the entire brief under 250 words.
6. End every brief with the disclaimer line exactly as shown.

Output format:

🔴 RED FLAGS:
- TestName: value unit | Reason: [copy abnormal_reason from findings] | Source: [source_report_id, date]

🟢 GREEN FLAGS (key normal findings for reassurance):
- TestName: value unit [Source: source_report_id]

📋 CLINICAL BRIEF (2-3 sentences, plain medical English for treating physician):
[Narrative summary. Cite source for any value mentioned.]

⚠️ AI-generated summary. Clinical judgment of the treating physician prevails.

FINDINGS:
{findings_json}"""

    def __init__(self, api_key: str):
        try:
            from google import genai
            from google.genai import types
            self.client = genai.Client(api_key=api_key)
            self.types  = types
            self._available = True
        except ImportError:
            self._available = False

    def synthesize(self, findings: list, patient_id: str) -> dict:
        if not findings or not self._available:
            return {"brief": None, "red_flags": [], "green_flags": [], "patient_id": patient_id}

        abnormal = [f for f in findings if f.get("is_abnormal")]
        normal   = [f for f in findings if not f.get("is_abnormal")]
        if len(abnormal) > 60:
            log.warning(f"Synthesis: {len(abnormal)} abnormal findings — truncating to 60")
        sample   = (abnormal[:60] + normal[:max(0, 60-len(abnormal))])[:60]

        finding_summary = [
            {
                "test": f.get("test_name_raw"),
                "result": f.get("result_raw"),
                "unit": f.get("unit"),
                "is_abnormal": f.get("is_abnormal"),
                "is_critical": f.get("is_critical"),
                "abnormal_direction": f.get("abnormal_direction"),
                "abnormal_reason": f.get("abnormal_reason"),
                "source_report_id": f.get("source_report_id"),
                "report_date": str(f.get("report_date", "")),
                "evidence_text": f.get("evidence_text", ""),
            }
            for f in sample
        ]

        prompt = self.SYNTHESIS_PROMPT.format(
            findings_json=json.dumps(finding_summary, indent=2)
        )

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=self.types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1024,
                    thinking_config=self.types.ThinkingConfig(thinking_budget=0),
                ),
            )
            brief_text = response.text or ""
            red_flags  = [f for f in findings if f.get("is_critical") or
                          (f.get("is_abnormal") and f.get("abnormal_direction") in
                           ("HIGH", "LOW", "POSITIVE"))]
            green_flags = [f for f in findings if not f.get("is_abnormal")]

            return {
                "patient_id":    patient_id,
                "brief":         brief_text,
                "red_flags":     [
                    {
                        "test":           f["test_name_raw"],
                        "result":         f.get("result_raw"),
                        "unit":           f.get("unit"),
                        "reason":         f.get("abnormal_reason"),
                        "is_critical":    f.get("is_critical", False),
                        "source_report":  f.get("source_report_id"),
                        "page":           f.get("page_number", 1),
                        "evidence":       f.get("evidence_text", ""),
                    }
                    for f in red_flags[:15]
                ],
                "green_flags":   [
                    {
                        "test":    f["test_name_raw"],
                        "result":  f.get("result_raw"),
                        "unit":    f.get("unit"),
                        "source":  f.get("source_report_id"),
                    }
                    for f in green_flags[:8]
                ],
                "findings_used":      len(sample),
                "findings_total":     len(findings),
                "disclaimer":         "AI-generated summary. Clinical judgment of the treating physician prevails.",
            }
        except Exception as e:
            log.error(f"Synthesis failed: {e}")
            return {"brief": f"Synthesis error: {e}", "red_flags": [], "green_flags": [],
                    "patient_id": patient_id}


# Chat

class DigiCareChat:
    """
    Answers clinical questions about a patient using their stored findings.
    Filters to relevant findings by keyword, falls back to all abnormal findings
    for general questions. Every answer cites source_report_id and page_number.
    """

    CHAT_PROMPT = """You are a clinical AI assistant for an Indian doctor. Answer based ONLY on the structured lab findings provided.

PATIENT FINDINGS:
{patient_data_json}

DOCTOR'S QUESTION: {question}

══ RULES ══
1. Use ONLY values present in the patient findings above. Do NOT add any values not listed.
2. Cite every value you mention: [Source: report_id, page_number, date]
3. If the findings include an 'abnormal_reason' for a flagged result, include it as clinical context.
4. If the data does not contain enough information to answer, say so explicitly.
5. End every response with the disclaimer.

══ FORMAT ══
Direct answer (1-2 sentences).
Details with citations: "TestName was VALUE UNIT [Source: REPORT_ID, Page N, Date: DATE]"
If abnormal: "Clinical note: [copy abnormal_reason from data]"
⚠️ AI-generated answer. Clinical judgment of the treating physician prevails.

Answer:"""

    KEYWORD_MAP = {
        "haemoglobin": ["HEMOGLOBIN"], "hemoglobin": ["HEMOGLOBIN"],
        "anaemia": ["HEMOGLOBIN", "RBC", "MCV", "MCH"],
        "anemia": ["HEMOGLOBIN", "RBC", "MCV", "MCH"],
        "kidney": ["CREATININE", "BUN", "eGFR", "SODIUM", "POTASSIUM"],
        "renal": ["CREATININE", "BUN", "eGFR"],
        "liver": ["AST", "ALT", "ALP", "BILIRUBIN_TOTAL", "ALBUMIN"],
        "sugar": ["GLUCOSE_PP", "GLUCOSE_RANDOM", "HBA1C"],
        "glucose": ["GLUCOSE_PP", "GLUCOSE_RANDOM"],
        "diabetes": ["GLUCOSE_PP", "GLUCOSE_RANDOM", "HBA1C"],
        "infection": ["WBC", "CRP", "ESR", "NEUTROPHILS_PCT"],
        "crp": ["CRP"], "thyroid": ["THYROID", "FREE_T3", "FREE_T4"],
        "vitamin": ["VITAMIN_B12", "VITAMIN_D"],
        "b12": ["VITAMIN_B12"], "vitamin b12": ["VITAMIN_B12"],
        "widal": ["WIDAL_TYPHI_O", "WIDAL_TYPHI_H", "WIDAL_CONCLUSION"],
        "typhoid": ["WIDAL_TYPHI_O", "WIDAL_TYPHI_H", "WIDAL_CONCLUSION"],
        "malaria": ["MALARIA_VIVAX_AG", "MALARIA_FALCIPARUM_AG"],
        "platelet": ["PLATELETS"],
        "urine": ["URINE_COLOR", "URINE_SG", "URINE_ALBUMIN", "URINE_PUS_CELLS"],
        "abnormal": None,  # None = return all abnormal findings
        "red flag": None,
        "critical": None,
        "all": None,
        "summary": None,
        "overview": None,
    }

    def __init__(self, api_key: str, storage: StorageLayer):
        try:
            from google import genai
            from google.genai import types
            self.client  = genai.Client(api_key=api_key)
            self.types   = types
            self.storage = storage
            self._available = True
        except ImportError:
            self._available = False

    def ask(self, patient_id: str, question: str) -> dict:
        if not self._available:
            return {"answer": "google-genai not installed", "sources": []}

        all_findings = self.storage.get_patient_findings(patient_id)
        if not all_findings:
            return {
                "answer": (
                    "No findings found for this patient. "
                    "Process at least one report first via POST /reports/process."
                ),
                "sources": [],
            }

        relevant = self._filter_relevant(all_findings, question)
        if not relevant:
            # General question — return all abnormal findings
            relevant = [f for f in all_findings if f.get("is_abnormal")] or all_findings[:20]

        patient_data = [
            {
                "test":             f.get("test_name_raw"),
                "result":           f.get("result_raw"),
                "unit":             f.get("unit"),
                "is_abnormal":      f.get("is_abnormal"),
                "is_critical":      f.get("is_critical", False),
                "abnormal_direction": f.get("abnormal_direction"),
                "abnormal_reason":  f.get("abnormal_reason"),     # clinical explanation
                "evidence_text":    f.get("evidence_text", ""),  # source quote
                "source_report_id": f.get("source_report_id"),
                "page_number":      f.get("page_number"),
                "report_date":      str(f.get("report_date", "")),
            }
            for f in relevant
        ]

        prompt = self.CHAT_PROMPT.format(
            patient_data_json=json.dumps(patient_data, indent=2),
            question=question,
            report_id=relevant[0].get("source_report_id", "report") if relevant else "report",
            date=str(relevant[0].get("report_date", "")) if relevant else "",
        )

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=self.types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=512,
                    thinking_config=self.types.ThinkingConfig(thinking_budget=0),
                ),
            )
            sources = list({f.get("source_report_id", "") for f in relevant})
            return {
                "answer":        response.text or "",
                "sources":       sources,
                "findings_used": len(relevant),
                "findings_cited": [
                    {
                        "test":   f.get("loinc_concept") or f.get("test_name_raw"),
                        "result": f.get("result_raw"),
                        "source": f.get("source_report_id"),
                        "page":   f.get("page_number", 1),
                        "reason": f.get("abnormal_reason"),
                    }
                    for f in relevant if f.get("is_abnormal")
                ][:10],
            }
        except Exception as e:
            log.error(f"Chat failed: {e}")
            return {"answer": f"Error: {e}", "sources": []}

    def _filter_relevant(self, findings: list, question: str) -> list:
        q = question.lower()
        relevant_loincs = set()
        return_all_abnormal = False

        for keyword, loincs in self.KEYWORD_MAP.items():
            if keyword in q:
                if loincs is None:
                    return_all_abnormal = True
                else:
                    relevant_loincs.update(loincs)

        if return_all_abnormal:
            return [f for f in findings if f.get("is_abnormal")]
        if relevant_loincs:
            return [f for f in findings if f.get("loinc_concept") in relevant_loincs]
        return []  # caller handles empty by returning all abnormal


# Pipeline

class DigiCarePipeline:
    """Main entry point. Wires together all pipeline stages."""

    def __init__(self, gemini_api_key: str, db_url: str = ""):
        self.extractor   = GeminiExtractor(api_key=gemini_api_key)
        self.validator   = ValidationLayer()
        self.loinc       = LoincNormalizer()
        self.storage     = StorageLayer(db_url=db_url)
        self.synthesizer = ClinicalSynthesizer(api_key=gemini_api_key)
        self.chat        = DigiCareChat(api_key=gemini_api_key, storage=self.storage)

        if db_url:
            self.storage.create_schema()

        log.info("DigiCarePipeline ready")

    def process_report(
        self,
        image_path: Union[str, Path, object],  # str/Path = file; PIL Image = already loaded
        patient_id: str,
        report_id: str,
        report_date: str,
    ) -> dict:
        """Run the full pipeline on a PDF or image. Returns findings, brief, stats, and cost."""
        path = Path(image_path) if not hasattr(image_path, "read") else None
        log.info(f"Pipeline start: {path.name if path else 'PIL Image'} | patient={patient_id}")

        result = {
            "report_id": report_id, "patient_id": patient_id,
            "report_date": report_date, "errors": [],
        }

        # Determine pages to process
        pages = []
        if path and path.suffix.lower() == ".pdf":
            try:
                pages = pdf_to_images(path)   # list of (PIL_Image, page_num)
            except Exception as e:
                result["errors"].append(f"PDF conversion failed: {e}")
                return result
        elif path:
            from PIL import Image as PILImage
            pages = [(PILImage.open(path).convert("RGB"), 1)]
        else:
            pages = [(image_path, 1)]  # PIL Image passed directly

        # Extract findings from each page
        all_findings   = []
        total_cost     = 0.0
        total_elapsed  = 0.0
        total_in_tok   = 0
        total_out_tok  = 0

        for img, page_num in pages:
            ex = self.extractor.extract(img, source_id=report_id, page=page_num)
            if ex.get("error"):
                result["errors"].append(f"Page {page_num}: {ex['error']}")
                continue
            all_findings.extend(ex["findings"])
            total_cost    += ex.get("cost_usd", 0.0)
            total_elapsed += ex.get("elapsed_sec", 0.0)
            total_in_tok  += ex.get("tokens_in", 0)
            total_out_tok += ex.get("tokens_out", 0)

        if not all_findings and result["errors"]:
            return result

        all_findings = self.loinc.normalize(all_findings)

        all_findings = self.validator.validate(all_findings)
        #  DEDUPLICATE findings before DB insert
        unique_map = {}
        
        for f in all_findings:
            key = (
                patient_id,
                report_date,
                f.get("loinc_concept") or f.get("test_name_raw"),
                f.get("source_report_id"),
                f.get("page_number"),
            )
            unique_map[key] = f   # last one wins (latest extraction)
        
        all_findings = list(unique_map.values())
        result["findings"] = all_findings

        # Storage
        saved = self.storage.save_findings(all_findings, patient_id, report_date)
        self.storage.save_report(
            report_id=report_id, patient_id=patient_id, report_date=report_date,
            findings_count=len(all_findings),
            source_file=str(path) if path else "",
            total_cost=total_cost, elapsed=total_elapsed,
        )
        result["findings_saved"] = saved

        brief = self.synthesizer.synthesize(all_findings, patient_id)
        result["brief"] = brief

        result["stats"] = {
            "findings_total":    len(all_findings),
            "findings_abnormal": sum(1 for f in all_findings if f.get("is_abnormal")),
            "findings_critical": sum(1 for f in all_findings if f.get("is_critical")),
            "pages_processed":   len(pages),
        }
        result["cost_breakdown"] = {
            "total_cost_usd": round(total_cost, 6),
            "elapsed_sec":    round(total_elapsed, 2),
            "tokens_in":      total_in_tok,
            "tokens_out":     total_out_tok,
        }

        log.info(f"Pipeline done: {len(all_findings)} findings | "
                 f"{result['stats']['findings_abnormal']} abnormal | "
                 f"${total_cost:.5f} | {total_elapsed:.1f}s")
        return result

    def ask(self, patient_id: str, question: str) -> dict:
        """Chat interface — source-cited Q&A."""
        return self.chat.ask(patient_id, question)

    def get_patient_summary(self, patient_id: str) -> dict:
        findings = self.storage.get_patient_findings(patient_id)
        if not findings:
            return {"patient_id": patient_id, "findings": [], "brief": None}
        brief = self.synthesizer.synthesize(findings, patient_id)
        return {"patient_id": patient_id, "findings_count": len(findings),
                "findings": findings, "brief": brief}
