"""
Spajanje audio snimki u jednu — simulacija realnog ulaska studenata
====================================================================
Spaja snimke iz foldera 'stitch' u jedan audio zapis s realističnim
razmacima između govornika, kao da se radi o stvarnoj snimci s ulaza
u predavaonicu.

Razmaci nisu fiksni — variraju po normalnoj distribuciji kako bi
simulirali realan tempo dolaska studenata (netko dođe odmah iza
prethodnog, netko malo čeka, netko kasni).

Pokretanje:
    python stitch.py

Izlaz:
    snimke/stitch_YYYY-MM-DD_HH-MM-SS.wav
"""

import os
import random
import numpy as np
import soundfile as sf
import librosa
from datetime import datetime

# ================================================================
# POSTAVKE
# ================================================================
DIR_ULAZ     = os.path.join("stitch", "snimke")       # Izvorne snimke govornika
DIR_IZLAZ    = os.path.join("stitch", "generirano")   # Generirane spojene snimke
DIR_LOG      = os.path.join("stitch", "log")          # Tekstualni logovi redoslijeda
SR        = 44100             # Izlazni sample rate (originalna kvaliteta)

# Razmak između govornika — normalna distribucija
# Realistično: student priđe, kaže kolegij, ode, sljedeći priđe
RAZMAK_SREDINA = 2.5          # Prosječan razmak između govornika (sekunde)
RAZMAK_STD     = 0.8          # Standardna devijacija razmaka
RAZMAK_MIN     = 1.0          # Minimalni razmak (nitko ne dolazi odmah)
RAZMAK_MAX     = 5.0          # Maksimalni razmak (nitko ne čeka previše dugo)

# Tišina na početku i kraju snimke
UVOD_SEKUNDE   = 1.0          # Tišina na početku (mikrofon se uključio)
OUTRO_SEKUNDE  = 1.5          # Tišina na kraju


# ================================================================
# UČITAVANJE AUDIO
# ================================================================
def ucitaj_audio(putanja: str, sr_ciljni: int) -> np.ndarray:
    if not putanja.lower().endswith(".wav"):
        from pydub import AudioSegment
        import io
        audio = AudioSegment.from_file(putanja)
        buf = io.BytesIO()
        audio.export(buf, format="wav")
        buf.seek(0)
        signal, sr = sf.read(buf)
    else:
        signal, sr = sf.read(putanja)

    signal = np.array(signal, dtype=np.float32)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if sr != sr_ciljni:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=sr_ciljni)
    return signal


def tisina(sekunde: float, sr: int) -> np.ndarray:
    return np.zeros(int(sekunde * sr), dtype=np.float32)


def realistican_razmak() -> float:
    """
    Generira realistični razmak između govornika.
    Miješa tri scenarija:
    - brz dolazak (30%): student čeka odmah iza prethodnog
    - normalan dolazak (50%): standardni tempo
    - kasni (20%): malo duže čekanje
    """
    izbor = random.random()
    if izbor < 0.30:
        # Brz — odmah iza prethodnog
        razmak = random.uniform(RAZMAK_MIN, 1.8)
    elif izbor < 0.80:
        # Normalan
        razmak = random.gauss(RAZMAK_SREDINA, RAZMAK_STD)
    else:
        # Kasni malo
        razmak = random.uniform(3.5, RAZMAK_MAX)
    return max(RAZMAK_MIN, min(RAZMAK_MAX, razmak))


# ================================================================
# GLAVNI PROGRAM
# ================================================================
def main():
    print("=" * 55)
    print("Spajanje snimki — simulacija ulaska studenata")
    print("=" * 55)

    if not os.path.isdir(DIR_ULAZ):
        print(f"GREŠKA: Folder '{DIR_ULAZ}' ne postoji.")
        return

    podrzani = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".mp4")
    snimke = [
        f for f in os.listdir(DIR_ULAZ)
        if f.lower().endswith(podrzani)
    ]

    if not snimke:
        print(f"Nema audio datoteka u folderu '{DIR_ULAZ}'.")
        return

    # Broj snimki za generiranje
    try:
        n_snimki = int(input(f"\nKoliko snimki generirati? (Enter = 1): ").strip() or 1)
        n_snimki = max(1, n_snimki)
    except ValueError:
        n_snimki = 1

    # Izostavljanje studenata
    print(f"\nKoliko studenata izostaviti po snimci?")
    print(f"  Enter / 0 = nitko se ne izostavlja")
    print(f"  1–{len(snimke)-1} = točno taj broj (random studenti svaki put)")
    print(f"  r = random broj i random studenti svaki put")
    unos = input("Odabir: ").strip().lower()
    if unos == "r":
        n_izostavi = -1
        print("  Broj i odabir izostavljenih bit će nasumični za svaku snimku.")
    else:
        try:
            n_izostavi = int(unos or 0)
            n_izostavi = max(0, min(n_izostavi, len(snimke) - 1))
        except ValueError:
            n_izostavi = 0
    if n_izostavi > 0:
        print(f"  Svaka snimka izostavlja točno {n_izostavi} nasumična studenta.")

    # Generiranje tekstualne datoteke
    gen_txt = input("Generirati tekstualnu datoteku s redoslijedom za svaku snimku? (d/N): ").strip().lower() == "d"

    print(f"\nGenerirat će se {n_snimki} snimki, svaka s {len(snimke) - n_izostavi} studenata.")
    print(f"\nParametri razmaka:")
    print(f"  Sredina: {RAZMAK_SREDINA}s, STD: {RAZMAK_STD}s")
    print(f"  Raspon:  {RAZMAK_MIN}s – {RAZMAK_MAX}s")

    os.makedirs(DIR_IZLAZ, exist_ok=True)
    os.makedirs(DIR_LOG,   exist_ok=True)

    for br_snimke in range(1, n_snimki + 1):
        print(f"\n{'=' * 55}")
        print(f"  Snimka {br_snimke}/{n_snimki}")
        print(f"{'=' * 55}")

        # Nasumični redoslijed za ovu snimku
        snimke_ova = snimke.copy()
        random.shuffle(snimke_ova)

        # Nasumično izostavljivanje za ovu snimku
        if n_izostavi == -1:
            # r — normalna distribucija centrirana oko 3, ograničena na [0, max-1]
            max_izostavi = len(snimke_ova) - 1
            n_ova = int(round(random.gauss(3, 1.2)))
            n_ova = max(0, min(n_ova, max_izostavi))
        else:
            # Fiksni broj, random studenti
            n_ova = n_izostavi

        if n_ova > 0:
            izostavljeni = random.sample(snimke_ova, n_ova)
            snimke_ova = [s for s in snimke_ova if s not in izostavljeni]
            print(f"  Izostavljeni ({n_ova}): {', '.join(os.path.splitext(s)[0] for s in sorted(izostavljeni))}")
        else:
            izostavljeni = []
            print(f"  Izostavljeni: nitko")

        print(f"  Redoslijed:")
        for i, s in enumerate(snimke_ova, 1):
            print(f"    {i:2d}. {s}")

        # Spajanje
        segmenti = [tisina(UVOD_SEKUNDE, SR)]
        razmaci  = []
        trenutno = UVOD_SEKUNDE

        for i, naziv in enumerate(snimke_ova):
            putanja = os.path.join(DIR_ULAZ, naziv)
            try:
                signal = ucitaj_audio(putanja, SR)
            except Exception as e:
                print(f"  UPOZORENJE: Ne mogu učitati {naziv} ({e}), preskačem.")
                continue

            trajanje = len(signal) / SR
            segmenti.append(signal)
            trenutno += trajanje

            print(f"\n    [{i+1:2d}] {naziv}  ({trajanje:.2f}s, početak: {trenutno - trajanje:.2f}s)")

            if i < len(snimke_ova) - 1:
                r = realistican_razmak()
                razmaci.append(r)
                segmenti.append(tisina(r, SR))
                trenutno += r
                print(f"         razmak do sljedećeg: {r:.2f}s")

        segmenti.append(tisina(OUTRO_SEKUNDE, SR))
        rezultat = np.concatenate(segmenti)

        ukupno = len(rezultat) / SR
        print(f"\n  Ukupno trajanje: {ukupno:.2f}s ({ukupno/60:.1f} min)")
        if razmaci:
            print(f"  Prosječni razmak: {sum(razmaci)/len(razmaci):.2f}s")

        ts    = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        izlaz = os.path.join(DIR_IZLAZ, f"stitch_{ts}_{br_snimke:02d}.wav")
        sf.write(izlaz, rezultat, SR)
        print(f"  Spremljeno: {izlaz}")

        # Tekstualna datoteka s redoslijedom
        if gen_txt:
            txt_putanja = os.path.join(DIR_LOG, f"stitch_{ts}_{br_snimke:02d}.txt")
            with open(txt_putanja, "w", encoding="utf-8") as f:
                f.write(f"Stitch snimka — {ts}\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Redoslijed govornika ({len(snimke_ova)}):\n")
                for j, naziv in enumerate(snimke_ova, 1):
                    ime = os.path.splitext(naziv)[0]
                    f.write(f"  {j:2d}. {ime}  ({naziv})\n")
                f.write("\n")
                if izostavljeni:
                    f.write(f"Izostavljeni ({len(izostavljeni)}):\n")
                    for naziv in sorted(izostavljeni):
                        ime = os.path.splitext(naziv)[0]
                        f.write(f"  - {ime}  ({naziv})\n")
                else:
                    f.write("Izostavljeni: nitko\n")
                f.write(f"\nUkupno trajanje: {ukupno:.2f}s ({ukupno/60:.1f} min)\n")
                if razmaci:
                    f.write(f"Prosječni razmak: {sum(razmaci)/len(razmaci):.2f}s\n")
            print(f"  Redoslijed:  {txt_putanja}")

    print(f"\n{'=' * 55}")
    print(f"Gotovo! Generirano {n_snimki} snimki u '{DIR_IZLAZ}'.")
    print("=" * 55)


if __name__ == "__main__":
    main()