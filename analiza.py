"""
Identifikacija govornika i obrada snimki
=========================================
Identifikacija VAD segmenata, clustering za broj govornika,
agregacija rezultata i spremanje u datoteku.
"""

import os
import numpy as np
from datetime import datetime
from sklearn.cluster import AgglomerativeClustering

from predobrada import ucitaj_signal, vad_segmentacija
from model import izvuci_embedding_sa_segmentacijom


# ================================================================
# IDENTIFIKACIJA - dva praga (prijedlog profesora)
# ================================================================
def identificiraj(embedding_ulaz: np.ndarray, baza: dict,
                  prag_donji: float, prag_gornji: float) -> tuple:
    """
    Usporeduje embedding s bazom i vraca (student, distanca, status).
    status: SIGURAN / NESIGURAN / ULJEZ
    """
    najbolji_student    = None
    najmanja_udaljenost = float("inf")
    emb = embedding_ulaz / np.linalg.norm(embedding_ulaz)

    for ime, referentni in baza.items():
        min_dist = min(float(1 - np.dot(emb, ref)) for ref in referentni)
        if min_dist < najmanja_udaljenost:
            najmanja_udaljenost = min_dist
            najbolji_student    = ime

    if najmanja_udaljenost <= prag_donji:
        return najbolji_student, najmanja_udaljenost, "SIGURAN"
    elif najmanja_udaljenost <= prag_gornji:
        return najbolji_student, najmanja_udaljenost, "NESIGURAN"
    else:
        return None, najmanja_udaljenost, "ULJEZ"


# ================================================================
# CLUSTERING - procjena broja govornika
# ================================================================
def procijeni_broj_govornika(embeddinzi: list, prag: float = 0.25) -> int:
    """
    Agglomerative clustering embeddinga VAD segmenata —
    procjenjuje koliko razlicitih govornika ima na snimci.
    """
    if len(embeddinzi) < 2:
        return len(embeddinzi)
    X = np.array(embeddinzi)
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=prag,
        metric="cosine", linkage="average"
    )
    clustering.fit(X)
    return len(set(clustering.labels_))


# ================================================================
# OBRADA JEDNE SNIMKE
# ================================================================
def obradi_snimku(putanja: str, baza: dict,
                  prag_donji: float, prag_gornji: float,
                  sr: int = 16000,
                  trajanje: float = 1.5,
                  preklapanje: float = 0.3,
                  prop_decrease: float = 0.75,
                  vad_top_db: float = 25,
                  vad_min_duljina: float = 0.3,
                  vad_spajanje: float = 0.15,
                  clustering_prag: float = 0.25) -> tuple:
    """
    Vraca (prepoznati, segmenti, uljezi_seg, n_govornika).
    segmenti = lista (poc, kraj, student, dist, status)
    """
    signal = ucitaj_signal(putanja, sr, prop_decrease)

    vad_seg = vad_segmentacija(signal, sr, vad_top_db, vad_min_duljina, vad_spajanje)
    if not vad_seg:
        return set(), [], 0, 0

    svi_segmenti    = []
    embeddinzi_svih = []
    uljezi_seg      = 0

    for poc, kraj in vad_seg:
        seg = signal[int(poc * sr):int(kraj * sr)]
        emb = izvuci_embedding_sa_segmentacijom(seg, sr, trajanje, preklapanje)
        embeddinzi_svih.append(emb)
        student, dist, status = identificiraj(emb, baza, prag_donji, prag_gornji)
        svi_segmenti.append((poc, kraj, student, dist, status))

    n_govornika = procijeni_broj_govornika(embeddinzi_svih, clustering_prag)

    # Zadrzavamo samo najbolji segment po studentu
    najbolji_po_studentu = {}
    for poc, kraj, student, dist, status in svi_segmenti:
        if student is None:
            uljezi_seg += 1
            continue
        if student not in najbolji_po_studentu or dist < najbolji_po_studentu[student][2]:
            najbolji_po_studentu[student] = (poc, kraj, dist, status)

    filtrirani = []
    prepoznati = set()
    for poc, kraj, student, dist, status in svi_segmenti:
        if student is None:
            filtrirani.append((poc, kraj, None, dist, status))
        else:
            best = najbolji_po_studentu.get(student)
            if best and best[0] == poc and best[1] == kraj:
                filtrirani.append((poc, kraj, student, dist, status))
                if status in ("SIGURAN", "NESIGURAN"):
                    prepoznati.add(student)

    return prepoznati, filtrirani, uljezi_seg, n_govornika


# ================================================================
# POMOCNA FUNKCIJA
# ================================================================
def fmt_s(sekunde: float) -> str:
    m = int(sekunde // 60)
    s = sekunde % 60
    return f"{m:02d}:{s:05.2f}"


# ================================================================
# SPREMANJE REZULTATA
# ================================================================
def spremi_rezultate(prisutnost: dict, svi_rezultati: dict,
                     putanja: str = "prisutnost.txt",
                     timestamp: str = None):
    datum_vrijeme = timestamp or datetime.now().strftime("%d.%m.%Y. u %H:%M:%S")
    prisutni  = [ime for ime, p in prisutnost.items() if p]
    nedostaju = [ime for ime, p in prisutnost.items() if not p]
    sep = "-" * 30 + "\n"

    with open(putanja, "w", encoding="utf-8") as f:
        f.write(f"Analiza provedena: {datum_vrijeme}\n")
        f.write(sep)
        for naziv, (prepoznati, segmenti, _, n_gov) in svi_rezultati.items():
            f.write(f"Snimka: {naziv}\n")
            f.write(f"  Procijenjeni broj govornika: {n_gov}\n")
            for poc, kraj, student, dist, status in segmenti:
                ime    = student if student else "NEPOZNAT"
                oznaka = "?" if status == "NESIGURAN" else ("+" if student else "!")
                f.write(f"  [{oznaka}] {fmt_s(poc)} - {fmt_s(kraj)}"
                        f"  {ime} (dist={dist:.4f}, {status})\n")
            f.write(f"  Prepoznati: {', '.join(sorted(prepoznati)) if prepoznati else 'nitko'}\n")
            f.write(sep)
        f.write(f"Ukupno prisutnih: {len(prisutni)}/{len(prisutnost)}\n")
        for ime in prisutni:
            f.write(f"  + {ime}\n")
        f.write(sep)
        f.write("Studenti koji nedostaju:\n")
        if nedostaju:
            for ime in nedostaju:
                f.write(f"  - {ime}\n")
        else:
            f.write("  SVI SU STUDENTI PRISUTNI\n")


# ================================================================
# EXCEL EXPORT
# ================================================================
def spremi_excel(prisutnost: dict, svi_rezultati: dict,
                 putanja: str = "prisutnost.xlsx",
                 timestamp: str = None):
    """
    Sprema rezultate u Excel tablicu s bojama (zelena/crvena)
    slicno GUI prikazu.
    Instalacija: pip install openpyxl
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    except ImportError:
        print("  GREŠKA: Instaliraj openpyxl: pip install openpyxl")
        return

    from datetime import datetime
    datum_vrijeme = timestamp or datetime.now().strftime("%d.%m.%Y. u %H:%M:%S")

    wb = Workbook()

    # ================================================================
    # List 1: Popis prisutnosti
    # ================================================================
    ws1 = wb.active
    ws1.title = "Popis prisutnosti"

    # Boje
    fill_success  = PatternFill("solid", fgColor="3d7a42")
    fill_danger   = PatternFill("solid", fgColor="7a3d3d")
    fill_zaglavlje = PatternFill("solid", fgColor="1e1e1e")
    fill_header   = PatternFill("solid", fgColor="2a2a2a")

    font_bijeli_bold = Font(color="e8e8e8", bold=True, name="Calibri", size=11)
    font_success     = Font(color="7ecf85", bold=True, name="Calibri", size=11)
    font_danger      = Font(color="cf7e7e", bold=True, name="Calibri", size=11)
    font_mute        = Font(color="888888", name="Calibri", size=10)

    tanki_rub = Side(style="thin", color="3d3d3d")
    rub = Border(left=tanki_rub, right=tanki_rub, top=tanki_rub, bottom=tanki_rub)

    centralno = Alignment(horizontal="center", vertical="center")
    lijevo    = Alignment(horizontal="left",   vertical="center")

    # Naslov
    ws1.merge_cells("A1:D1")
    ws1["A1"] = "Sustav za popisivanje studenata — Popis prisutnosti"
    ws1["A1"].fill      = fill_zaglavlje
    ws1["A1"].font      = Font(color="e8e8e8", bold=True, name="Calibri", size=13)
    ws1["A1"].alignment = centralno
    ws1.row_dimensions[1].height = 30

    ws1.merge_cells("A2:D2")
    ws1["A2"] = f"Analiza provedena: {datum_vrijeme}"
    ws1["A2"].fill      = fill_zaglavlje
    ws1["A2"].font      = font_mute
    ws1["A2"].alignment = centralno
    ws1.row_dimensions[2].height = 20

    # Zaglavlje tablice
    zaglavlja = ["Student", "Prisutan", "Status", "Distanca"]
    for col, zag in enumerate(zaglavlja, 1):
        c = ws1.cell(row=3, column=col, value=zag)
        c.fill      = fill_header
        c.font      = font_bijeli_bold
        c.alignment = centralno
        c.border    = rub
    ws1.row_dimensions[3].height = 22

    # Podaci
    for row, (ime, prisutan) in enumerate(sorted(prisutnost.items()), 4):
        oznaka = "DA" if prisutan else "NE"
        fill   = fill_success if prisutan else fill_danger
        font_s = font_success  if prisutan else font_danger

        # Pronadji min distancu za ovog studenta
        min_dist = None
        for naziv, (prepoznati, segmenti, _, _) in svi_rezultati.items():
            for _, _, student, dist, status in segmenti:
                if student == ime:
                    if min_dist is None or dist < min_dist:
                        min_dist = dist

        podaci = [
            ime,
            oznaka,
            "Prisutan" if prisutan else "Odsutan",
            f"{min_dist:.4f}" if min_dist is not None else "—"
        ]

        for col, vrijednost in enumerate(podaci, 1):
            c = ws1.cell(row=row, column=col, value=vrijednost)
            c.fill      = fill
            c.font      = font_s
            c.alignment = centralno if col > 1 else lijevo
            c.border    = rub
        ws1.row_dimensions[row].height = 20

    ws1.column_dimensions["A"].width = 22
    ws1.column_dimensions["B"].width = 12
    ws1.column_dimensions["C"].width = 14
    ws1.column_dimensions["D"].width = 14

    # ================================================================
    # List 2: Detalji po snimci
    # ================================================================
    ws2 = wb.create_sheet("Detalji po snimci")

    ws2.merge_cells("A1:F1")
    ws2["A1"] = "Detalji analize po snimci"
    ws2["A1"].fill      = fill_zaglavlje
    ws2["A1"].font      = Font(color="e8e8e8", bold=True, name="Calibri", size=13)
    ws2["A1"].alignment = centralno
    ws2.row_dimensions[1].height = 30

    zaglavlja2 = ["Snimka", "Pocetak", "Kraj", "Trajanje (s)", "Govornik", "Status"]
    for col, zag in enumerate(zaglavlja2, 1):
        c = ws2.cell(row=2, column=col, value=zag)
        c.fill      = fill_header
        c.font      = font_bijeli_bold
        c.alignment = centralno
        c.border    = rub
    ws2.row_dimensions[2].height = 22

    fill_siguran   = PatternFill("solid", fgColor="3d7a42")
    fill_nesiguran = PatternFill("solid", fgColor="7a5c2a")
    fill_uljez     = PatternFill("solid", fgColor="7a3d3d")

    trenutni_red = 3
    for naziv, (prepoznati, segmenti, _, n_gov) in svi_rezultati.items():
        for poc, kraj, student, dist, status in segmenti:
            ime_studenta = student if student else "NEPOZNAT"

            if status == "SIGURAN":
                fill_s = fill_siguran
                font_s = font_success
            elif status == "NESIGURAN":
                fill_s = fill_nesiguran
                font_s = Font(color="f0c080", bold=False, name="Calibri", size=10)
            else:
                fill_s = fill_uljez
                font_s = font_danger

            podaci2 = [naziv, fmt_s(poc), fmt_s(kraj),
                       f"{(kraj-poc):.1f}", ime_studenta, status]

            for col, vrijednost in enumerate(podaci2, 1):
                c = ws2.cell(row=trenutni_red, column=col, value=vrijednost)
                c.fill      = fill_s
                c.font      = font_s
                c.alignment = centralno if col > 1 else lijevo
                c.border    = rub
            ws2.row_dimensions[trenutni_red].height = 18
            trenutni_red += 1

    ws2.column_dimensions["A"].width = 26
    ws2.column_dimensions["B"].width = 12
    ws2.column_dimensions["C"].width = 12
    ws2.column_dimensions["D"].width = 14
    ws2.column_dimensions["E"].width = 18
    ws2.column_dimensions["F"].width = 14

    wb.save(putanja)
    print(f"  Excel tablica spremljena: {putanja}")