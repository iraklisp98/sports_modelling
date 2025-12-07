import argparse
from pathlib import Path
import pandas as pd
def merge_league_csvs(league_dir: Path, output_dir: Path, output_name: str | None = None) -> Path:
    league_dir = league_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(league_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {league_dir}")

    df_list = [pd.read_csv(str(fp)) for fp in csv_files]
    merged = pd.concat(df_list, ignore_index=True)

    league = league_dir.name
    filename = output_name or f"{league}_combined.csv"
    output_path = output_dir / filename
    merged.to_csv(output_path, index=False)
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Merge league CSV files into combined outputs.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).parent / "data"),
        help="Base data directory containing league folders (e.g., ENG, FRA, SPA).",
    )
    parser.add_argument(
        "--leagues",
        type=str,
        nargs="*",
        default=["ENG", "FRA", "SPA"],
        help="League folder names to merge.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent / "data" / "processed"),
        help="Directory to write merged CSV files.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    results = []
    for league in args.leagues:
        league_dir = data_dir / league
        if not league_dir.exists():
            raise FileNotFoundError(f"League directory not found: {league_dir}")
        out_path = merge_league_csvs(league_dir, output_dir)
        results.append(out_path)

    for p in results:
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
import argparse
import glob
import os
from pathlib import Path
import pandas as pd


def merge_league_csvs(league_dir: Path, output_dir: Path, output_name: str | None = None) -> Path:
    league_dir = league_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(league_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {league_dir}")

    df_list = [pd.read_csv(str(fp)) for fp in csv_files]
    merged = pd.concat(df_list, ignore_index=True)

    league = league_dir.name
    filename = output_name or f"{league}_combined.csv"
    output_path = output_dir / filename
    merged.to_csv(output_path, index=False)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Merge league CSV files into combined outputs.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).parent / "data"),
        help="Base data directory containing league folders (e.g., ENG, FRA, SPA).",
    )
    parser.add_argument(
        "--leagues",
        type=str,
        nargs="*",
        default=["ENG", "FRA", "SPA"],
        help="League folder names to merge.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent / "data" / "processed"),
        help="Directory to write merged CSV files.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    results = []
    for league in args.leagues:
        league_dir = data_dir / league
        if not league_dir.exists():
            raise FileNotFoundError(f"League directory not found: {league_dir}")
        out_path = merge_league_csvs(league_dir, output_dir)
        results.append(out_path)

    for p in results:
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
