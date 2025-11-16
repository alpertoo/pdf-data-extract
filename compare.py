import argparse
import re
from pathlib import Path

import pandas as pd
import pdfplumber


# =========================================
# Helper: extract text from a single PDF
# =========================================

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Read all pages of a PDF and return the text as one string.
    """
    text = ""
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# =========================================
# Parser: FedEx
# =========================================

def parse_fedex(text: str) -> pd.DataFrame:
    """
    Extract shipment lines from FedEx PDFs.

    Logic:
      - Read the PDF text line by line.
      - Identify lines that look like a shipment row.
        Example:
            <shipment_number> <date_dd/mm/yyyy> FedEx Priority ... <values>
      - Extract:
          * shipment_number
          * shipment_date (string)
          * charge (last decimal number on the line)

    Regex pattern:
      - ^(\\d{9,})  matches a long shipment number at the start of the line.
      - (\\d{2}/\\d{2}/\\d{4})  captures dates like 13/10/2025.
      - \\d+\\.\\d+  finds decimal values such as 2.99 or 17.10.
        The last decimal on the line is treated as the total charge.
    """

    rows = []

    for line in text.splitlines():
        # Match a shipment line: starts with shipment_number and date
        pattern = r"^(\d{9,})\s+(\d{2}/\d{2}/\d{4})"
        m = re.match(pattern, line)
        if not m:
            continue

        shipment_number = m.group(1)
        shipment_date = m.group(2)

        # Extract all decimal numbers on the line
        nums = re.findall(r"\d+\.\d+", line)
        if not nums:
            continue

        # Last decimal number is the total charge for that shipment
        charge = float(nums[-1])

        rows.append(
            {
                "shipment_number": shipment_number,
                "shipment_date": shipment_date,
                "charge": charge,
                "raw_line": line,
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        df["shipment_date_parsed"] = pd.to_datetime(
            df["shipment_date"], format="%d/%m/%Y", errors="coerce"
        )

    return df


# =========================================
# Parser: Evri
# =========================================

def parse_evri(text: str) -> pd.DataFrame:
    """
    Extract despatch service lines from Evri PDFs.

    Logic:
      - Read the PDF text line by line.
      - Identify lines that follow the Evri numeric pattern:
            <service text> <quantity> <unit_price> <VAT_code> <line_value>
        Example:
            Scottish Highlands & Islands Parcel 36 5.28 S 190.08
      - Extract:
          * service (text before quantity)
          * quantity
          * price (unit price)
          * value (line total)

    Regex pattern:
      - ^\\s*        leading spaces.
      - (.+?)        service name, non greedy, up to first numeric block.
      - ([\\d,]+)    quantity, may contain commas.
      - (\\d+\\.\\d+)  unit price.
      - [A-Z]        VAT code, for example S or O.
      - ([\\d,]+\\.\\d+)  line total value.
    """

    rows = []
    pattern = r"^\s*(.+?)\s+([\d,]+)\s+(\d+\.\d+)\s+[A-Z]\s+([\d,]+\.\d+)"

    for line in text.splitlines():
        match = re.match(pattern, line)
        if not match:
            continue

        service = match.group(1).strip()
        quantity = int(match.group(2).replace(",", ""))
        price = float(match.group(3))
        value = float(match.group(4).replace(",", ""))

        rows.append(
            {
                "service": service,
                "quantity": quantity,
                "price": price,
                "value": value,
                "raw_line": line,
            }
        )

    return pd.DataFrame(rows)


# =========================================
# Cleaning: Evri
# =========================================

def clean_evri(evri_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split Evri rows into:
      - core rows with value > 0 (real charge lines)
      - excluded rows with value == 0 (headers or meta lines)
    """
    evri_core = evri_df[evri_df["value"] > 0].copy()
    evri_excluded = evri_df[evri_df["value"] == 0].copy()
    return evri_core, evri_excluded


# =========================================
# Metrics
# =========================================

def compute_fedex_metrics(fedex_df: pd.DataFrame, fixed_rate: float) -> dict:
    """
    Compute FedEx metrics needed for the summary table.
    """

    despatches = len(fedex_df)

    if despatches == 0:
        return {
            "despatches": 0,
            "spend": 0.0,
            "avg_cost": 0.0,
            "variance": 0.0,
            "total_difference": 0.0,
            "status": "No data",
        }

    spend = round(fedex_df["charge"].sum(), 3)
    avg_cost = round(spend / despatches, 3)
    variance = round(avg_cost - fixed_rate, 3)
    total_difference = round(variance * despatches, 3)

    if variance > 0:
        status = "Over the fixed rate"
    elif variance < 0:
        status = "Under the fixed rate"
    else:
        status = "On the fixed rate"

    return {
        "despatches": despatches,
        "spend": spend,
        "avg_cost": avg_cost,
        "variance": variance,
        "total_difference": total_difference,
        "status": status,
    }


def compute_evri_metrics(evri_core: pd.DataFrame, fixed_rate: float) -> dict:
    """
    Compute Evri metrics needed for the summary table, using cleaned data.
    """

    despatches = int(evri_core["quantity"].sum())

    if despatches == 0:
        return {
            "despatches": 0,
            "spend": 0.0,
            "avg_cost": 0.0,
            "variance": 0.0,
            "total_difference": 0.0,
            "status": "No data",
        }

    spend = round(evri_core["value"].sum(), 3)
    avg_cost = round(spend / despatches, 3)
    variance = round(avg_cost - fixed_rate, 3)
    total_difference = round(variance * despatches, 3)

    if variance > 0:
        status = "Over the fixed rate"
    elif variance < 0:
        status = "Under the fixed rate"
    else:
        status = "On the fixed rate"

    return {
        "despatches": despatches,
        "spend": spend,
        "avg_cost": avg_cost,
        "variance": variance,
        "total_difference": total_difference,
        "status": status,
    }


# =========================================
# Main script
# =========================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare FedEx and Evri costs against fixed rates."
    )

    parser.add_argument(
        "--fedex",
        nargs="+",
        required=True,
        help="Paths to FedEx PDF files",
    )
    parser.add_argument(
        "--evri",
        required=True,
        help="Path to Evri PDF file",
    )
    parser.add_argument(
        "--outdir",
        default="output",
        help="Output folder for CSV files (default: output)",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Fixed cost rates (can be parameterised if needed)
    fixed_rate_fedex = 3.10
    fixed_rate_evri = 2.44

    # ----- FedEx: extract and parse all PDFs -----
    fedex_frames = []
    for fedex_path in args.fedex:
        pdf_path = Path(fedex_path)
        text = extract_text_from_pdf(pdf_path)
        df = parse_fedex(text)
        df["source_file"] = pdf_path.name
        fedex_frames.append(df)

    if fedex_frames:
        fedex_df = pd.concat(fedex_frames, ignore_index=True)
    else:
        fedex_df = pd.DataFrame(columns=["shipment_number", "shipment_date", "charge"])

    # ----- Evri: extract and parse -----
    evri_path = Path(args.evri)
    evri_text = extract_text_from_pdf(evri_path)
    evri_df = parse_evri(evri_text)
    evri_df["source_file"] = evri_path.name

    # Clean Evri rows (value > 0 keeps only real charge lines)
    evri_core, evri_excluded = clean_evri(evri_df)

    # ----- Compute metrics -----
    fedex_metrics = compute_fedex_metrics(fedex_df, fixed_rate_fedex)
    evri_metrics = compute_evri_metrics(evri_core, fixed_rate_evri)

    # ----- Build summary table -----
    summary = pd.DataFrame(
        [
            {
                "carrier": "FedEx",
                "despatches": fedex_metrics["despatches"],
                "spend": fedex_metrics["spend"],
                "avg_cost_per_despatch": fedex_metrics["avg_cost"],
                "fixed_rate": fixed_rate_fedex,
                "variance": fedex_metrics["variance"],
                "total_difference": fedex_metrics["total_difference"],
                "status": fedex_metrics["status"],
            },
            {
                "carrier": "Evri",
                "despatches": evri_metrics["despatches"],
                "spend": evri_metrics["spend"],
                "avg_cost_per_despatch": evri_metrics["avg_cost"],
                "fixed_rate": fixed_rate_evri,
                "variance": evri_metrics["variance"],
                "total_difference": evri_metrics["total_difference"],
                "status": evri_metrics["status"],
            },
        ]
    )

    # ----- Save CSVs -----
    summary.to_csv(outdir / "summary_for_dashboard.csv", index=False)
    fedex_df.to_csv(outdir / "fedex_cleaned.csv", index=False)
    evri_core.to_csv(outdir / "evri_cleaned.csv", index=False)
    evri_excluded.to_csv(outdir / "evri_excluded.csv", index=False)

    print("Summary:")
    print(summary.to_string(index=False))
    print()
    print(f"Files written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
