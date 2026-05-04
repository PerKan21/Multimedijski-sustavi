"""
Spajanje audio snimki u jednu — simulacija realnog ulaska studenata
====================================================================
Spaja snimke iz foldera 'stitch' u jedan audio zapis s realističnim
razmacima između govornika, kao da se radi o stvarnoj snimci s ulaza
u predavaonicu.

Razmaci nisu fiksni — variraju po normalnoj distribuciji kako bi
simulirali realan tempo dolaska studenata.

Pokretanje:
    python stitch.py

Izlaz:
    stitch/generirano/stitch_YYYY-MM-DD_HH-MM-SS_01.wav
"""

import os
import uuid
import random
import numpy as np
import soundfile as sf
import librosa
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ================================================================
# POSTAVKE
# ================================================================
DIR_ULAZ     = os.path.join("stitch", "snimke")
DIR_ULJEZ    = os.path.join("stitch", "snimke", "uljez")
DIR_IZLAZ    = os.path.join("stitch", "generirano")
DIR_LOG      = os.path.join("stitch", "log")
SR        = 44100

RAZMAK_SREDINA = 2.5
RAZMAK_STD     = 0.8
RAZMAK_MIN     = 1.0
RAZMAK_MAX     = 5.0

UVOD_SEKUNDE   = 1.0
OUTRO_SEKUNDE  = 1.5


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
    - brz dolazak (30%)
    - normalan dolazak (50%)
    - kasni (20%)
    """
    izbor = random.random()
    if izbor < 0.30:
        razmak = random.uniform(RAZMAK_MIN, 1.8)
    elif izbor < 0.80:
        razmak = random.gauss(RAZMAK_SREDINA, RAZMAK_STD)
    else:
        razmak = random.uniform(3.5, RAZMAK_MAX)
    return max(RAZMAK_MIN, min(RAZMAK_MAX, razmak))


def _generiraj_naziv_izlaza(br_snimke: int, session_id: str, ts: str) -> str:
    """
    Generira jedinstven naziv izlazne datoteke.
    Koristi session_id (UUID) za razlikovanje različitih pokretanja skripte,
    čime se izbjegava prepisivanje datoteka pri brzom uzastopnom pokretanju.
    Timestamp (ts) se prenosi izvana kako bi bio konzistentan unutar sesije.
    """
    return f"stitch_{ts}_{session_id}_{br_snimke:02d}.wav"


# ================================================================
# ULJEZI
# ================================================================
def skeniraj_uljeze() -> Dict[str, List[str]]:
    """
    Skenira stitch/snimke/uljez/ i vraća dict s listama putanja po kategoriji.

    Podržane strukture:
      stitch/snimke/uljez/muski/  → kategorija "muski"
      stitch/snimke/uljez/zenski/ → kategorija "zenski"
      stitch/snimke/uljez/*.wav   → kategorija "ostali" (ako nema podmapa)

    Vraća npr.:
      {"muski": ["...put1.wav"], "zenski": ["...put2.wav"]}
    """
    podrzani = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".mp4")
    rezultat: Dict[str, List[str]] = {}

    if not os.path.isdir(DIR_ULJEZ):
        return rezultat

    # Provjeri podmape muski/zenski
    for kategorija in ("muski", "zenski"):
        podmap = os.path.join(DIR_ULJEZ, kategorija)
        if os.path.isdir(podmap):
            datoteke = [
                os.path.join(podmap, f)
                for f in sorted(os.listdir(podmap))
                if f.lower().endswith(podrzani)
            ]
            if datoteke:
                rezultat[kategorija] = datoteke

    # Ako nema podmapa, uzmi datoteke direktno iz uljez/
    if not rezultat:
        datoteke = [
            os.path.join(DIR_ULJEZ, f)
            for f in sorted(os.listdir(DIR_ULJEZ))
            if f.lower().endswith(podrzani)
        ]
        if datoteke:
            rezultat["ostali"] = datoteke

    return rezultat


def odaberi_uljeze_za_snimku(uljezi_po_kat: Dict[str, List[str]],
                               odabir_uljeza: str) -> List[Tuple[str, str]]:
    """
    Vraća listu (putanja, kategorija) uljeza za jednu snimku.
    odabir_uljeza: "nijedan" | "muski" | "zenski" | "oba" | "random"
    """
    if odabir_uljeza == "nijedan" or not uljezi_po_kat:
        return []

    rezultat: List[Tuple[str, str]] = []

    def _random_iz_kat(kat: str) -> Optional[Tuple[str, str]]:
        datoteke = uljezi_po_kat.get(kat, [])
        if not datoteke:
            return None
        return (random.choice(datoteke), kat)

    if odabir_uljeza == "muski":
        u = _random_iz_kat("muski") or _random_iz_kat("ostali")
        if u: rezultat.append(u)

    elif odabir_uljeza == "zenski":
        u = _random_iz_kat("zenski") or _random_iz_kat("ostali")
        if u: rezultat.append(u)

    elif odabir_uljeza == "oba":
        u_m = _random_iz_kat("muski")
        u_z = _random_iz_kat("zenski")
        if u_m: rezultat.append(u_m)
        if u_z: rezultat.append(u_z)
        # Ako nema podjele po spolu, uzmi dva random iz "ostali"
        if not rezultat:
            datoteke = uljezi_po_kat.get("ostali", [])
            for put in random.sample(datoteke, min(2, len(datoteke))):
                rezultat.append((put, "ostali"))

    elif odabir_uljeza == "random":
        sve_kategorije = list(uljezi_po_kat.keys())
        kat = random.choice(sve_kategorije)
        u = _random_iz_kat(kat)
        if u: rezultat.append(u)

    return rezultat


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

    try:
        n_snimki = int(input(f"\nKoliko snimki generirati? (Enter = 1): ").strip() or 1)
        n_snimki = max(1, n_snimki)
    except ValueError:
        n_snimki = 1

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

    gen_txt = input("Generirati tekstualnu datoteku s redoslijedom za svaku snimku? (d/N): ").strip().lower() == "d"

    # --- Uljezi ---
    uljezi_po_kat = skeniraj_uljeze()
    odabir_uljeza = "nijedan"

    if uljezi_po_kat:
        ima_muskog  = "muski"  in uljezi_po_kat
        ima_zenskog = "zenski" in uljezi_po_kat
        ima_ostale  = "ostali" in uljezi_po_kat

        print(f"\nUljezi (stitch/snimke/uljez/):")
        if ima_muskog:
            print(f"  Muški:  {len(uljezi_po_kat['muski'])} snimki")
        if ima_zenskog:
            print(f"  Ženski: {len(uljezi_po_kat['zenski'])} snimki")
        if ima_ostale:
            print(f"  Ostali: {len(uljezi_po_kat['ostali'])} snimki")

        print(f"\nDodati uljeza u svaku snimku?")
        print(f"  [0] Nijedan")
        if ima_muskog or ima_ostale:
            print(f"  [1] Muški uljez")
        if ima_zenskog or ima_ostale:
            print(f"  [2] Ženski uljez")
        if (ima_muskog or ima_ostale) and (ima_zenskog or ima_ostale):
            print(f"  [3] Oba uljeza")
        print(f"  [r] Random (svaka snimka dobiva nasumičnog uljeza)")

        unos_u = input("Odabir (Enter = 0): ").strip().lower()
        if unos_u == "1":
            odabir_uljeza = "muski"
            print("  Svaka snimka dobit će muškog uljeza na nasumičnoj poziciji.")
        elif unos_u == "2":
            odabir_uljeza = "zenski"
            print("  Svaka snimka dobit će ženskog uljeza na nasumičnoj poziciji.")
        elif unos_u == "3":
            odabir_uljeza = "oba"
            print("  Svaka snimka dobit će oba uljeza na nasumičnim pozicijama.")
        elif unos_u == "r":
            odabir_uljeza = "random"
            print("  Svaka snimka dobit će nasumičnog uljeza.")
        else:
            odabir_uljeza = "nijedan"
            print("  Bez uljeza.")
    else:
        print(f"\nNapomena: Nije pronađen folder uljeza ({DIR_ULJEZ}), preskačem.")

    print(f"\nGenerirat će se {n_snimki} snimki.")
    print(f"\nParametri razmaka:")
    print(f"  Sredina: {RAZMAK_SREDINA}s, STD: {RAZMAK_STD}s")
    print(f"  Raspon:  {RAZMAK_MIN}s – {RAZMAK_MAX}s")

    os.makedirs(DIR_IZLAZ, exist_ok=True)
    os.makedirs(DIR_LOG,   exist_ok=True)

    # FIX #5: Timestamp i session_id se bilježe na početku sesije,
    # a ne na kraju svake snimke, pa su konzistentni kroz cijelu sesiju.
    session_id = uuid.uuid4().hex[:6]
    session_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for br_snimke in range(1, n_snimki + 1):
        print(f"\n{'=' * 55}")
        print(f"  Snimka {br_snimke}/{n_snimki}")
        print(f"{'=' * 55}")

        snimke_ova = snimke.copy()
        random.shuffle(snimke_ova)

        if n_izostavi == -1:
            max_izostavi = max(0, len(snimke_ova) - 1)
            sredina_izost = max(1, round(len(snimke_ova) * 0.2))
            n_ova = int(round(random.gauss(sredina_izost, 1.0)))
            n_ova = max(0, min(n_ova, max_izostavi))
        else:
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

        uljezi_ova = odaberi_uljeze_za_snimku(uljezi_po_kat, odabir_uljeza)
        if uljezi_ova:
            print(f"  Uljezi ({len(uljezi_ova)}):")
            for put_u, kat_u in uljezi_ova:
                print(f"    [{kat_u}] {os.path.basename(put_u)}")

        # FIX #1: Uklonjena mrtva varijabla `sekvenca`.
        sekvenca_s_uljezima: List[Tuple[str, Optional[str]]] = [
            (s, None) for s in snimke_ova
        ]

        for put_u, kat_u in uljezi_ova:
            pozicija = random.randint(0, len(sekvenca_s_uljezima))
            sekvenca_s_uljezima.insert(pozicija, (put_u, kat_u))

        segmenti = [tisina(UVOD_SEKUNDE, SR)]
        razmaci  = []
        trenutno = UVOD_SEKUNDE

        # FIX #2 + #3: Pozicija se inkrementira tek nakon uspješnog učitavanja,
        # a log_redoslijed bilježi stvarni kombinirani redoslijed (studenti + uljezi).
        log_redoslijed: List[Tuple[int, str, Optional[str], float, float]] = []
        # format: (pozicija, naziv, kategorija_uljeza_ili_None, pocetak_s, trajanje_s)

        pozicija_u_redoslijedu = 0
        for i, (naziv_ili_put, kat_uljeza) in enumerate(sekvenca_s_uljezima):
            je_uljez = kat_uljeza is not None

            if je_uljez:
                putanja = naziv_ili_put
                oznaka_ispis = f"ULJEZ [{kat_uljeza}]"
            else:
                putanja = os.path.join(DIR_ULAZ, naziv_ili_put)
                oznaka_ispis = naziv_ili_put

            try:
                signal = ucitaj_audio(putanja, SR)
            except Exception as e:
                print(f"  UPOZORENJE: Ne mogu učitati {oznaka_ispis} ({e}), preskačem.")
                continue

            # FIX #3: Inkrement tek nakon uspješnog učitavanja
            pozicija_u_redoslijedu += 1
            trajanje = len(signal) / SR
            pocetak = trenutno

            segmenti.append(signal)
            trenutno += trajanje

            print(f"\n    [{pozicija_u_redoslijedu:2d}] {oznaka_ispis}"
                  f"  ({trajanje:.2f}s, početak: {pocetak:.2f}s)")

            # FIX #2: Bilježimo stvarni kombinirani redoslijed za log
            log_redoslijed.append((
                pozicija_u_redoslijedu,
                naziv_ili_put,
                kat_uljeza,
                pocetak,
                trajanje
            ))

            if i < len(sekvenca_s_uljezima) - 1:
                r = realistican_razmak()
                razmaci.append(r)
                segmenti.append(tisina(r, SR))
                trenutno += r
                print(f"         razmak do sljedećeg: {r:.2f}s")

        segmenti.append(tisina(OUTRO_SEKUNDE, SR))
        rezultat_audio = np.concatenate(segmenti)

        ukupno = len(rezultat_audio) / SR
        print(f"\n  Ukupno trajanje: {ukupno:.2f}s ({ukupno/60:.1f} min)")
        if razmaci:
            print(f"  Prosječni razmak: {sum(razmaci)/len(razmaci):.2f}s")

        naziv_izlaza = _generiraj_naziv_izlaza(br_snimke, session_id, session_ts)
        izlaz = os.path.join(DIR_IZLAZ, naziv_izlaza)
        sf.write(izlaz, rezultat_audio, SR)
        print(f"  Spremljeno: {izlaz}")

        if gen_txt:
            txt_naziv = naziv_izlaza.replace(".wav", ".txt")
            txt_putanja = os.path.join(DIR_LOG, txt_naziv)
            with open(txt_putanja, "w", encoding="utf-8") as f:
                f.write(f"Stitch snimka — {session_ts_str}\n")
                f.write("=" * 40 + "\n\n")

                # FIX #2: Log prikazuje stvarni kombinirani redoslijed
                f.write(f"Stvarni redoslijed u snimci ({len(log_redoslijed)}):\n")
                for poz, naziv, kat_ul, poc, traj in log_redoslijed:
                    if kat_ul is not None:
                        oznaka = f"ULJEZ [{kat_ul}] {os.path.basename(naziv)}"
                    else:
                        ime = os.path.splitext(naziv)[0]
                        oznaka = f"{ime}  ({naziv})"
                    f.write(f"  {poz:2d}. {oznaka}"
                            f"  [pocetak: {poc:.2f}s, trajanje: {traj:.2f}s]\n")
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