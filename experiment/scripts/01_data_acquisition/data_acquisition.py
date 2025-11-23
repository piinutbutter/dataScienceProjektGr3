from pathlib import Path
import pandas as pd

# Pfade an dein Projekt anpassen
RAW_DIR = Path("../../data/raw/GRXEUR_M1")        # hier liegen die DAT_ASCII_....csv
OUT_DIR = Path("../../data/Raw/Bars_1m_GRXEUR")   # Zielordner für Parquet-Dateien

OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_grxeur_ascii(path: Path) -> pd.DataFrame:
    """
    Lädt eine DAT_ASCII_GRXEUR_M1_YYYY.csv-Datei und gibt
    einen bereinigten DataFrame mit benannten Spalten zurück.
    """
    df = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=["datetime_str", "open", "high", "low", "close", "volume"],
    )

    # Datum + Zeit in echten Timestamp umwandeln
    df["datetime"] = pd.to_datetime(df["datetime_str"], format="%Y%m%d %H%M%S")

    # Aufräumen
    df = df.drop(columns=["datetime_str"])
    df = df.set_index("datetime").sort_index()

    return df


def main():
    csv_files = sorted(RAW_DIR.glob("DAT_ASCII_GRXEUR_M1_*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"Keine CSV-Dateien unter {RAW_DIR} gefunden.")

    all_dfs = []

    for path in csv_files:
        print(f"Lade {path.name} ...")
        df_year = load_grxeur_ascii(path)

        # Optional: Jahr extrahieren
        year = path.stem.split("_")[-1]
        # Jahresweise abspeichern
        out_year_path = OUT_DIR / f"GRXEUR_M1_{year}.parquet"
        df_year.to_parquet(out_year_path)
        print(f"→ gespeichert nach {out_year_path}")

        all_dfs.append(df_year)

    # Optional: alles zu einem großen DataFrame zusammenfügen
    full_df = pd.concat(all_dfs).sort_index()
    full_out = OUT_DIR / "GRXEUR_M1_2010_2018.parquet"
    full_df.to_parquet(full_out)
    print(f"Vollständiger Datensatz gespeichert unter {full_out}")


if __name__ == "__main__":
    main()