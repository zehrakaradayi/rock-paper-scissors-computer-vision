"""training.ipynb'nin seçilen bölümlerini çalıştırır, çıktıları notebook'a kaydeder.

Kullanım:  .venv/bin/python tools/run_notebook.py A      # Bölüm 0 + Bölüm 1 (Model A)
           .venv/bin/python tools/run_notebook.py B      # Bölüm 0 + Bölüm 2 (Model B)
           .venv/bin/python tools/run_notebook.py C      # Bölüm 0 + Bölüm 3 (karşılaştırma)

Her hücreden sonra notebook diske yazılır; VS Code'da açıksan "Revert File" ile güncel halini görürsün.
"""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "training.ipynb"
SECTION_TITLES = {"0": "## Bölüm 0", "A": "## Bölüm 1", "B": "## Bölüm 2", "C": "## Bölüm 3"}


def sections(nb):
    """Her hücreye ait olduğu bölümü (0 / A / B / C) verir."""
    cur, out = None, []
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            for key, title in SECTION_TITLES.items():
                if title in cell.source:
                    cur = key
        out.append(cur)
    return out


def main(target: str):
    nb = nbformat.read(NB, as_version=4)
    which = sections(nb)
    client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    with client.setup_kernel():
        for i, (cell, sec) in enumerate(zip(nb.cells, which)):
            if cell.cell_type != "code" or sec not in ("0", target):
                continue
            print(f"▶ hücre {cell.id} (bölüm {sec})", flush=True)
            client.execute_cell(cell, i)
            for o in cell.outputs:
                if o.get("output_type") == "stream":
                    print(o["text"], end="", flush=True)
            nbformat.write(nb, NB)
    print("bitti", flush=True)


if __name__ == "__main__":
    main(sys.argv[1].upper())
