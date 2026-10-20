"""Batch scoring: safe CSV parsing, validation, scoring."""
import io
from dataclasses import dataclass, field

import pandas as pd

from src.models.schema import MAX_BATCH_ROWS
from src.models.service import ChurnService
from src.models.validation import ValidationResult, validate_batch


def parse_csv_bytes(data: bytes) -> pd.DataFrame:
    """Parse an uploaded CSV, raising ValueError with a friendly message on any problem."""
    if not data or not data.strip():
        raise ValueError("The file is empty.")
    df = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), dtype=str, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise ValueError("The file has no columns. Is it a CSV with a header row?") from None
        except pd.errors.ParserError as exc:
            raise ValueError(f"The file could not be read as CSV: {exc}") from None
    if df is None:
        raise ValueError("The file could not be decoded as text.")
    if df.empty:
        raise ValueError("The file has a header but no data rows.")
    if len(df) > MAX_BATCH_ROWS:
        raise ValueError(f"The file has {len(df):,} rows; the limit is {MAX_BATCH_ROWS:,}.")
    return df


def sanitize_cell(value):
    """Stop spreadsheet formula injection in text we echo back into a CSV."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


@dataclass
class BatchResult:
    scored: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: list[dict] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)
    n_received: int = 0
    n_estimated_total_charges: int = 0

    @property
    def n_scored(self) -> int:
        return len(self.scored)

    @property
    def n_rejected(self) -> int:
        return self.n_received if self.missing_columns else self.n_received - self.n_scored


def run_batch(df: pd.DataFrame, service: ChurnService) -> BatchResult:
    if len(df) > MAX_BATCH_ROWS:
        raise ValueError(f"Too many rows ({len(df):,}); the limit is {MAX_BATCH_ROWS:,}.")
    result: ValidationResult = validate_batch(df)
    if result.missing_columns or result.valid.empty:
        return BatchResult(pd.DataFrame(), result.errors, result.missing_columns, result.n_rows, 0)
    scores = service.score(result.valid)
    out = pd.concat([scores, result.valid], axis=1)
    out.insert(0, "row", out.index + 1)
    id_col = next((c for c in df.columns if str(c).strip() == "customerID"), None)
    if id_col is not None:
        ids = df.reset_index(drop=True).loc[out.index, id_col].map(sanitize_cell)
        out.insert(1, "customerID", ids.to_numpy())
    return BatchResult(out.reset_index(drop=True), result.errors, [], result.n_rows,
                       result.n_total_charges_estimated)
