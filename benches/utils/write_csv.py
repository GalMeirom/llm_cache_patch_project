# write_csv.py
import csv


def write_dicts_to_csv(
    rows: list[dict], csv_path: str, field_order: list[str] | None = None
) -> None:
    """
    Write a list of dicts (same keys) to a CSV.
    Each dict = one row; keys = columns.
    csv_path is the output file name.
    """
    if not rows:
        # If empty, write only header if provided; else create an empty file.
        fieldnames = field_order or []
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            if fieldnames:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        return

    fieldnames = field_order or list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            # Normalize None → "" to keep CSV clean
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})


# Example
if __name__ == "__main__":
    data = [
        {"id": 1, "name": "Alice", "score": 9.5},
        {"id": 2, "name": "Bob", "score": None},
    ]
    write_dicts_to_csv(data, "out.csv")
