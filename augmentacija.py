"""
Augmentacija audio podataka za speaker recognition
====================================================
Iz postojećih referentnih snimki u bazi generira dodatne varijante
kako bi se povećao broj uzoraka po studentu i poboljšala robustnost
ECAPA-TDNN embeddinga.

Augmentacije koje se primjenjuju:
  - Pitch shift (visina tona)
  - Time stretch (brzina govora)
  - Dodavanje Gaussovog šuma
  - Promjena glasnoće
  - Room reverb (simulacija prostora)

Struktura:
    baza/
        Tomislav_Perkovic/
            original.wav          <- originalna snimka
            aug_pitch_up.wav      <- generirane varijante
            aug_pitch_down.wav
            aug_brze.wav
            ...

Instalacija:
    pip install librosa soundfile numpy
"""

# ================================================================
# IMPORTS
# ================================================================
import os
import numpy as np
import librosa
import soundfile as sf
import warnings

warnings.filterwarnings("ignore")


# ================================================================
# POSTAVKE - konfiguriraj koje augmentacije zelite i koliko agresivno
# ================================================================
SR          = 16000
DIR_BAZA    = "baza"
PODRZANI_FORMATI = (".wav", ".mp3", ".m4a", ".ogg", ".flac")

# Definicija augmentacija: (naziv, parametri)
# Svaka augmentacija generira jednu novu datoteku po originalnoj snimci
AUGMENTACIJE = [
    ("pitch_up",    {"pitch_steps":  2.0}),   # Visi ton
    ("pitch_down",  {"pitch_steps": -2.0}),   # Nizi ton
    ("pitch_up2",   {"pitch_steps":  4.0}),   # Znacajno visi ton
    ("pitch_down2", {"pitch_steps": -4.0}),   # Znacajno nizi ton
    ("brze",        {"time_rate":    1.15}),   # 15% brzi govor
    ("sporije",     {"time_rate":    0.85}),   # 15% sporiji govor
    ("sum_mali",    {"sum_razina":   0.003}),  # Mali pozadinski sum
    ("sum_veci",    {"sum_razina":   0.007}),  # Veci pozadinski sum
    ("tiho",        {"glasnoca":     0.6}),    # Tiše
    ("glasno",      {"glasnoca":     1.4}),    # Glasnije
    ("reverb",      {"reverb":       True}),   # Simulacija prostora
]


# ================================================================
# UCITAVANJE AUDIO - ucitava datoteku i normalizira na 16kHz mono
# ================================================================
def ucitaj_audio(putanja: str) -> tuple:
    """Ucitava audio datoteku i vraca (signal, sr)."""
    # Konverzija ne-wav formata kroz pydub ako treba
    if not putanja.lower().endswith(".wav"):
        from pydub import AudioSegment
        ime = os.path.splitext(putanja)[0]
        wav_put = ime + "_temp.wav"
        AudioSegment.from_file(putanja).export(wav_put, format="wav")
        signal, sr = sf.read(wav_put)
        os.remove(wav_put)
    else:
        signal, sr = sf.read(putanja)

    signal = np.array(signal, dtype=np.float32)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if sr != SR:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=SR)
    return signal, SR


# ================================================================
# AUGMENTACIJE - svaka funkcija prima signal i vraca modificirani signal
# ================================================================

def aug_pitch(signal: np.ndarray, sr: int, pitch_steps: float) -> np.ndarray:
    """Mijenja visinu tona za pitch_steps polutonova."""
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=pitch_steps)


def aug_time_stretch(signal: np.ndarray, time_rate: float) -> np.ndarray:
    """Ubrzava ili usporava govor bez mijenjanja tona."""
    return librosa.effects.time_stretch(signal, rate=time_rate)


def aug_sum(signal: np.ndarray, sum_razina: float) -> np.ndarray:
    """Dodaje Gaussov bijeli sum."""
    sum_signal = np.random.normal(0, sum_razina, len(signal))
    return signal + sum_signal


def aug_glasnoca(signal: np.ndarray, glasnoca: float) -> np.ndarray:
    """Mijenja glasnocu mnozeci signal s faktorom."""
    return signal * glasnoca


def aug_reverb(signal: np.ndarray, sr: int) -> np.ndarray:
    """
    Simulira akustiku prostorije jednostavnim FIR filterom.
    Dodaje kratke odgode (early reflections) koje imitiraju odjek.
    """
    # Jednostavni room impulse response
    ir_duljina = int(0.1 * sr)  # 100ms
    ir = np.zeros(ir_duljina)
    ir[0] = 1.0                          # Direktni signal
    ir[int(0.02 * sr)] = 0.4             # Prva refleksija (20ms)
    ir[int(0.04 * sr)] = 0.25            # Druga refleksija (40ms)
    ir[int(0.07 * sr)] = 0.15            # Treca refleksija (70ms)
    ir[int(0.09 * sr)] = 0.08            # Cetvrta refleksija (90ms)

    reverb_signal = np.convolve(signal, ir)[:len(signal)]
    # Normalizacija
    return reverb_signal / np.max(np.abs(reverb_signal) + 1e-8) * np.max(np.abs(signal))


def primijeni_augmentaciju(signal: np.ndarray, sr: int, naziv: str, params: dict) -> np.ndarray:
    """Primjenjuje odabranu augmentaciju na signal."""
    if "pitch_steps" in params:
        return aug_pitch(signal, sr, params["pitch_steps"])
    elif "time_rate" in params:
        return aug_time_stretch(signal, params["time_rate"])
    elif "sum_razina" in params:
        return aug_sum(signal, params["sum_razina"])
    elif "glasnoca" in params:
        return aug_glasnoca(signal, params["glasnoca"])
    elif "reverb" in params:
        return aug_reverb(signal, sr)
    else:
        return signal


# ================================================================
# NORMALIZACIJA - osigurava da signal ne prekoraci [-1, 1]
# ================================================================
def normaliziraj(signal: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        return signal / max_val * 0.95
    return signal


# ================================================================
# GLAVNI PROGRAM - prolazi kroz sve studente u bazi i za svaku
#                  originalnu snimku generira sve augmentacije
# ================================================================
def main():
    print("=" * 55)
    print("Augmentacija audio podataka")
    print("=" * 55)

    ukupno_generirano = 0

    for ime_studenta in sorted(os.listdir(DIR_BAZA)):
        putanja_studenta = os.path.join(DIR_BAZA, ime_studenta)
        if not os.path.isdir(putanja_studenta):
            continue

        print(f"\nStudent: {ime_studenta}")

        # Pronadji originalne snimke (preskoci vec augmentirane)
        originalne = [
            f for f in os.listdir(putanja_studenta)
            if f.lower().endswith(PODRZANI_FORMATI)
            and not f.startswith("aug_")
            and not f.endswith("_konv.wav")
        ]

        if not originalne:
            print("  Nema originalnih snimki, preskacam.")
            continue

        print(f"  Originalnih snimki: {len(originalne)}")

        for naziv_snimke in originalne:
            putanja_snimke = os.path.join(putanja_studenta, naziv_snimke)
            ime_bez_ext    = os.path.splitext(naziv_snimke)[0]

            try:
                signal, sr = ucitaj_audio(putanja_snimke)
            except Exception as e:
                print(f"  UPOZORENJE: Ne mogu ucitati {naziv_snimke} ({e})")
                continue

            # Generiraj svaku augmentaciju
            for aug_naziv, aug_params in AUGMENTACIJE:
                izlaz_naziv = f"aug_{ime_bez_ext}_{aug_naziv}.wav"
                izlaz_put   = os.path.join(putanja_studenta, izlaz_naziv)

                # Preskoci ako vec postoji
                if os.path.exists(izlaz_put):
                    continue

                try:
                    aug_signal = primijeni_augmentaciju(signal, sr, aug_naziv, aug_params)
                    aug_signal = normaliziraj(aug_signal)
                    sf.write(izlaz_put, aug_signal, sr)
                    ukupno_generirano += 1
                except Exception as e:
                    print(f"  UPOZORENJE: Augmentacija '{aug_naziv}' nije uspjela za {naziv_snimke} ({e})")

        # Prebrojaj ukupno snimki nakon augmentacije
        sve_snimke = [
            f for f in os.listdir(putanja_studenta)
            if f.lower().endswith(".wav") and not f.endswith("_konv.wav")
        ]
        print(f"  Ukupno snimki nakon augmentacije: {len(sve_snimke)}")

    print(f"\n{'=' * 55}")
    print(f"Gotovo! Generirano {ukupno_generirano} novih snimki.")
    print(f"Pokreni ecapa_tdnn.py za treniranje s prosirenom bazom.")


if __name__ == "__main__":
    main()