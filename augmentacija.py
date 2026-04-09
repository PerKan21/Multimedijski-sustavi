"""
Augmentacija audio podataka za speaker recognition
====================================================
Iz postojećih referentnih snimki u bazi generira dodatne varijante
kako bi se povećao broj uzoraka po studentu i poboljšala robustnost
ECAPA-TDNN embeddinga.

Svako pokretanje skripte dodaje još jedan sloj augmentacije:
  - 1. pokretanje: aug_*.wav        (augmentacije originalnih snimki)
  - 2. pokretanje: 2aug_*.wav       (augmentacije augmentiranih snimki)
  - 3. pokretanje: 3aug_*.wav       (augmentacije 2. sloja)
  - itd.

Instalacija:
    pip install librosa soundfile numpy
"""

# ================================================================
# IMPORTS
# ================================================================
import os
import re
import numpy as np
import librosa
import soundfile as sf
import warnings

warnings.filterwarnings("ignore")


# ================================================================
# POSTAVKE
# ================================================================
SR               = 16000
DIR_BAZA         = "baza"
PODRZANI_FORMATI = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".mp4")

AUGMENTACIJE = [
    ("pitch_up",    {"pitch_steps":  2.0}),
    ("pitch_down",  {"pitch_steps": -2.0}),
    ("pitch_up2",   {"pitch_steps":  4.0}),
    ("pitch_down2", {"pitch_steps": -4.0}),
    ("brze",        {"time_rate":    1.15}),
    ("sporije",     {"time_rate":    0.85}),
    ("sum_mali",    {"sum_razina":   0.003}),
    ("sum_veci",    {"sum_razina":   0.007}),
    ("tiho",        {"glasnoca":     0.6}),
    ("glasno",      {"glasnoca":     1.4}),
    ("reverb",      {"reverb":       True}),
]


# ================================================================
# ODREĐIVANJE SLOJA - provjerava koji sloj augmentacije je sljedeći
#                     na temelju postojećih datoteka u folderu
# ================================================================
def odredi_sloj(putanja_studenta: str) -> int:
    """
    Gleda postojeće aug_ datoteke i vraća sljedeći broj sloja.
    Nema aug_ datoteka -> sloj 1 (prefiks: aug_)
    Ima aug_ ali ne 2aug_ -> sloj 2 (prefiks: 2aug_)
    Ima 2aug_ ali ne 3aug_ -> sloj 3 (prefiks: 3aug_)
    itd.
    """
    datoteke = os.listdir(putanja_studenta)
    max_sloj = 0
    for dat in datoteke:
        # Provjeri je li augmentirana datoteka
        if dat.startswith("aug_"):
            max_sloj = max(max_sloj, 1)
        else:
            # Provjeri format Naug_ (npr. 2aug_, 3aug_)
            match = re.match(r'^(\d+)aug_', dat)
            if match:
                max_sloj = max(max_sloj, int(match.group(1)))
    return max_sloj + 1


def prefiks_sloja(sloj: int) -> str:
    """Vraća prefiks za dani sloj: sloj 1 -> 'aug_', sloj 2 -> '2aug_', itd."""
    return "aug_" if sloj == 1 else f"{sloj}aug_"


def prefiks_prethodnog_sloja(sloj: int) -> str:
    """Vraća prefiks prethodnog sloja (snimki koje augmentiramo)."""
    if sloj == 1:
        return None  # augmentiramo originalne
    return "aug_" if sloj == 2 else f"{sloj - 1}aug_"


# ================================================================
# UCITAVANJE AUDIO
# ================================================================
def ucitaj_audio(putanja: str) -> tuple:
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
# AUGMENTACIJE
# ================================================================
def aug_pitch(signal, sr, pitch_steps):
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=pitch_steps)

def aug_time_stretch(signal, time_rate):
    return librosa.effects.time_stretch(signal, rate=time_rate)

def aug_sum(signal, sum_razina):
    return signal + np.random.normal(0, sum_razina, len(signal))

def aug_glasnoca(signal, glasnoca):
    return signal * glasnoca

def aug_reverb(signal, sr):
    ir_duljina = int(0.1 * sr)
    ir = np.zeros(ir_duljina)
    ir[0] = 1.0
    ir[int(0.02 * sr)] = 0.4
    ir[int(0.04 * sr)] = 0.25
    ir[int(0.07 * sr)] = 0.15
    ir[int(0.09 * sr)] = 0.08
    reverb_signal = np.convolve(signal, ir)[:len(signal)]
    return reverb_signal / np.max(np.abs(reverb_signal) + 1e-8) * np.max(np.abs(signal))

def primijeni_augmentaciju(signal, sr, params):
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
    return signal

def normaliziraj(signal):
    max_val = np.max(np.abs(signal))
    return signal / max_val * 0.95 if max_val > 0 else signal


# ================================================================
# GLAVNI PROGRAM
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

        # Odredi koji sloj augmentacije radimo
        sloj        = odredi_sloj(putanja_studenta)
        novi_prefiks = prefiks_sloja(sloj)
        stari_prefiks = prefiks_prethodnog_sloja(sloj)

        print(f"  Sloj augmentacije: {sloj} (prefiks: {novi_prefiks})")

        # Odaberi snimke koje augmentiramo
        if sloj == 1:
            # Augmentiramo originalne (bez ikakvog aug_ prefiksa)
            izvorne = [
                f for f in os.listdir(putanja_studenta)
                if f.lower().endswith(PODRZANI_FORMATI)
                and not re.match(r'^(\d*aug_)', f)
                and not f.endswith("_konv.wav")
            ]
        else:
            # Augmentiramo snimke prethodnog sloja
            izvorne = [
                f for f in os.listdir(putanja_studenta)
                if f.startswith(stari_prefiks)
                and f.lower().endswith(".wav")
                and not f.endswith("_konv.wav")
            ]

        if not izvorne:
            print(f"  Nema snimki za augmentaciju sloja {sloj}, preskacam.")
            continue

        print(f"  Snimki za augmentaciju: {len(izvorne)}")

        for naziv_snimke in izvorne:
            putanja_snimke = os.path.join(putanja_studenta, naziv_snimke)
            ime_bez_ext    = os.path.splitext(naziv_snimke)[0]

            try:
                signal, sr = ucitaj_audio(putanja_snimke)
            except Exception as e:
                print(f"  UPOZORENJE: Ne mogu ucitati {naziv_snimke} ({e})")
                continue

            for aug_naziv, aug_params in AUGMENTACIJE:
                izlaz_naziv = f"{novi_prefiks}{ime_bez_ext}_{aug_naziv}.wav"
                izlaz_put   = os.path.join(putanja_studenta, izlaz_naziv)

                if os.path.exists(izlaz_put):
                    continue

                try:
                    aug_signal = primijeni_augmentaciju(signal, sr, aug_params)
                    aug_signal = normaliziraj(aug_signal)
                    sf.write(izlaz_put, aug_signal, sr)
                    ukupno_generirano += 1
                except Exception as e:
                    print(f"  UPOZORENJE: '{aug_naziv}' nije uspjela za {naziv_snimke} ({e})")

        # Ukupno snimki
        sve = [f for f in os.listdir(putanja_studenta)
               if f.lower().endswith(".wav") and not f.endswith("_konv.wav")]
        print(f"  Ukupno snimki nakon augmentacije: {len(sve)}")

    print(f"\n{'=' * 55}")
    print(f"Gotovo! Generirano {ukupno_generirano} novih snimki.")
    print(f"Obrisi baza_cache.pkl i pokreni main.py.")


if __name__ == "__main__":
    main()