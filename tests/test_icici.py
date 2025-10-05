import pandas as pd
from pathlib import Path
import importlib

def test_parse_matches_csv():
    pkg = importlib.import_module('custom_parsers.icici_parser')
    pdf = Path('data/icici/icic_sample.pdf')
    ref_csv = Path('data/icici/icic_sample.csv')
    ref = pd.read_csv(ref_csv)
    got = pkg.parse(str(pdf))
    got = got.astype(str).fillna('')
    ref = ref.astype(str).fillna('')
    assert got.equals(ref), f"DataFrame didn't match reference.\nREF:\n{ref}\nGOT:\n{got}"
