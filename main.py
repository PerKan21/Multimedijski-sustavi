"""
Sustav za popisivanje studenata - ECAPA-TDNN
=============================================
Glavni program — sadrži sve postavke i terminalni mod pokretanja.
Za GUI pokretanje koristi: python gui.py

Instalacija:
    pip install transformers torch soundfile librosa pydub noisereduce tqdm scikit-learn
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from analiza import obradi_snimku, spremi_rezultate, spremi_excel, fmt_s
from model import ucitaj_model, ucitaj_bazu, izracunaj_pragove

# ================================================================
# POSTAVKE - sve konfiguracijske varijable na jednom mjestu
# ================================================================
SR               = 16000
DIR_BAZA         = "baza"
DIR_SNIMKE       = "snimke"
PODRZANI_FORMATI = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".mp4")
CACHE_PUTANJA    = "baza_cache.pkl"

# --- Segmentacija (i za bazu i za VAD segmente) ---
SEGMENT_TRAJANJE    = 1.5    # Duljina jednog segmenta (sekunde)
SEGMENT_PREKLAPANJE = 0.3    # Preklapanje između segmenata (sekunde)

# --- VAD: detekcija govora u ulaznim snimkama ---
VAD_TOP_DB      = 25         # Prag energije ispod kojeg = tišina (dB)
VAD_MIN_DULJINA = 0.3        # Minimalna duljina govornog segmenta (sekunde)
VAD_SPAJANJE    = 0.15       # Spoji segmente bliže od ovoga (sekunde)

# --- Predobrada signala ---
SUM_PROP_DECREASE = 0.75     # Agresivnost redukcije šuma (0-1)

# --- Pragovi prepoznavanja ---
# dist <= prag_donji               -> [+] SIGURAN
# prag_donji < dist <= prag_gornji -> [?] NESIGURAN
# dist > prag_gornji               -> [!] ULJEZ
FAKTOR_GORNJEG_PRAGA = 1.8   # Gornji prag = donji * ovaj faktor

# Fiksni pragovi — postavi na float između 0.0 i 1.0 za fiksni prag,
# ili None (ili bilo što drugo) za dinamički izračun iz baze
FIKSNI_PRAG_DONJI  = None   # npr. 0.12 za fiksni, None za dinamički
FIKSNI_PRAG_GORNJI = None   # npr. 0.22 za fiksni, None za dinamički

# --- Clustering: procjena broja govornika ---
CLUSTERING_PRAG = 0.25

def primijeni_pragove(prag_donji: float, prag_gornji: float) -> tuple:
    """
    Ako su FIKSNI_PRAG_DONJI / FIKSNI_PRAG_GORNJI postavljeni na
    vrijednost između 0.0 i 1.0, koristi ih umjesto dinamičkih.
    """
    d = prag_donji
    g = prag_gornji

    if isinstance(FIKSNI_PRAG_DONJI, float) and 0.0 < FIKSNI_PRAG_DONJI < 1.0:
        d = FIKSNI_PRAG_DONJI
        print(f"  Fiksni donji prag:  {d:.4f}")
    else:
        print(f"  Dinamički donji prag: {d:.4f}")

    if isinstance(FIKSNI_PRAG_GORNJI, float) and 0.0 < FIKSNI_PRAG_GORNJI < 1.0:
        g = FIKSNI_PRAG_GORNJI
        print(f"  Fiksni gornji prag: {g:.4f}")
    else:
        print(f"  Dinamički gornji prag: {g:.4f}")

    return d, g


# ================================================================
# TERMINALNI MOD
# ================================================================
def main():
    os.makedirs(DIR_BAZA,   exist_ok=True)
    os.makedirs(DIR_SNIMKE, exist_ok=True)

    print("=" * 55)
    print("Sustav za popisivanje studenata (ECAPA-TDNN)")
    print("=" * 55)

    ucitaj_model()

    print("\n[1] Ucitavanje baze govornika...")
    baza = ucitaj_bazu(DIR_BAZA)
    print(f"    Ukupno studenata u bazi: {len(baza)}")
    prag_donji, prag_gornji = izracunaj_pragove(baza)
    prag_donji, prag_gornji = primijeni_pragove(prag_donji, prag_gornji)

    prisutnost = {ime: False for ime in baza}

    snimke = sorted([
        f for f in os.listdir(DIR_SNIMKE)
        if f.lower().endswith(PODRZANI_FORMATI) and not f.endswith("_konv.wav")
    ])

    if not snimke:
        print("\nNema snimki u folderu 'snimke'.")
        return

    print(f"\n[2] Obrada {len(snimke)} ulaznih snimki...")
    svi_rezultati = {}

    for naziv in tqdm(snimke, desc="  Napredak", leave=True):
        putanja = os.path.join(DIR_SNIMKE, naziv)
        prepoznati, segmenti, uljezi_seg, n_gov = obradi_snimku(
            putanja, baza, prag_donji, prag_gornji
        )
        for student in prepoznati:
            prisutnost[student] = True
        svi_rezultati[naziv] = (prepoznati, segmenti, uljezi_seg, n_gov)

    print()
    for naziv, (prepoznati, segmenti, uljezi_seg, n_gov) in svi_rezultati.items():
        print(f"\n  Snimka: {naziv}")
        print(f"  Procijenjeni broj govornika: {n_gov}")
        print(f"  {'Pocetak':10s} {'Kraj':10s} {'Trajanje':10s} Govornik")
        print("  " + "-" * 55)
        for poc, kraj, student, dist, status in segmenti:
            if student:
                ime    = student
                oznaka = "?" if status == "NESIGURAN" else "+"
            else:
                ime    = "NEPOZNAT (uljez)"
                oznaka = "!"
            print(f"  {fmt_s(poc):10s} {fmt_s(kraj):10s} {(kraj-poc):6.1f}s"
                  f"    [{oznaka}] {ime} (dist={dist:.4f})")
        print(f"  Prepoznati: {', '.join(sorted(prepoznati)) if prepoznati else 'nitko'}")

    print("\n" + "=" * 55)
    print("POPIS PRISUTNOSTI")
    print("=" * 55)
    for ime, prisutan in prisutnost.items():
        print(f"  [{'+'  if prisutan else '-'}] {ime}")

    from datetime import datetime
    timestamp = datetime.now().strftime("%d.%m.%Y. u %H:%M:%S")
    ts_ime    = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    os.makedirs("rezultati", exist_ok=True)

    print("\nKako želiš spremiti rezultate?")
    print("  [0] Nemoj spremati, samo izađi")
    print("  [1] Tekstualna datoteka (.txt)")
    print("  [2] Excel tablica (.xlsx)")
    odabir = input("Odabir (0/1/2, Enter = 1): ").strip()

    if odabir == "0":
        print("Rezultati nisu spremljeni.")
    elif odabir == "2":
        putanja = os.path.join("rezultati", f"prisutnost_{ts_ime}.xlsx")
        spremi_excel(prisutnost, svi_rezultati, putanja, timestamp=timestamp)
        print(f"Rezultati spremljeni u: {putanja}")
    else:
        putanja = os.path.join("rezultati", f"prisutnost_{ts_ime}.txt")
        spremi_rezultate(prisutnost, svi_rezultati, putanja, timestamp=timestamp)
        print(f"Rezultati spremljeni u: {putanja}")


if __name__ == "__main__":
    main()