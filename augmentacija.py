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
    pip install librosa soundfile numpy scipy
"""

# ================================================================
# IMPORTS
# ================================================================
import os
import re
import io
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
    # --- Time stretch ---
    ("brze_malo",    {"time_rate":    1.05}),   # +5% brzine
    ("sporije_malo", {"time_rate":    0.95}),   # -5% brzine
    # --- Gaussov šum ---
    ("sum_mali",     {"sum_razina":   0.003}),
    ("sum_veci",     {"sum_razina":   0.007}),
    # --- Glasnoća ---
    ("tiho",         {"glasnoca":     0.6}),
    ("glasno",       {"glasnoca":     1.4}),
    # --- Reverb ---
    ("reverb",       {"reverb":       True}),
    # --- Telefonski filtar (bandpass 300-3400 Hz) ---
    ("telefon",      {"telefon":      True}),
    # --- MP3 kompresija (simulira artefakte kompresije) ---
    ("mp3",          {"mp3":          True}),
    # --- Slučajni crop (nasumični isječak signala) ---
    ("crop",         {"crop":         True}),
]


# ================================================================
# ODREĐIVANJE SLOJA
# ================================================================
def odredi_sloj(putanja_studenta: str) -> int:
    datoteke = os.listdir(putanja_studenta)
    max_sloj = 0
    for dat in datoteke:
        if dat.startswith("aug_"):
            max_sloj = max(max_sloj, 1)
        else:
            match = re.match(r'^(\d+)aug_', dat)
            if match:
                max_sloj = max(max_sloj, int(match.group(1)))
    return max_sloj + 1


def prefiks_sloja(sloj: int) -> str:
    return "aug_" if sloj == 1 else f"{sloj}aug_"


def prefiks_prethodnog_sloja(sloj: int) -> str:
    if sloj == 1:
        return None
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


def aug_telefon(signal, sr):
    """
    Bandpass filtar 300–3400 Hz — simulira frekvencijski odziv
    telefonske linije / lošijeg mikrofona.
    """
    from scipy.signal import butter, sosfilt
    sos = butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
    return sosfilt(sos, signal).astype(np.float32)


def aug_mp3(signal, sr):
    """
    Prolazi signal kroz MP3 enkoder/dekoder (128 kbps) i natrag.
    Simulira artefakte lossy kompresije koji su česti u stvarnim snimkama.
    """
    try:
        from pydub import AudioSegment
        # Signal -> WAV bytes -> pydub -> MP3 bytes -> pydub -> signal
        buf_wav = io.BytesIO()
        sf.write(buf_wav, signal, sr, format="WAV")
        buf_wav.seek(0)
        audio = AudioSegment.from_wav(buf_wav)
        buf_mp3 = io.BytesIO()
        audio.export(buf_mp3, format="mp3", bitrate="128k")
        buf_mp3.seek(0)
        audio_dec = AudioSegment.from_mp3(buf_mp3)
        samples = np.array(audio_dec.get_array_of_samples(), dtype=np.float32)
        samples /= (2 ** (audio_dec.sample_width * 8 - 1))
        if audio_dec.channels > 1:
            samples = samples.reshape(-1, audio_dec.channels).mean(axis=1)
        if audio_dec.frame_rate != sr:
            samples = librosa.resample(samples, orig_sr=audio_dec.frame_rate, target_sr=sr)
        # Poravnaj duljinu
        min_len = min(len(signal), len(samples))
        return samples[:min_len]
    except Exception:
        return signal


def aug_crop(signal, sr):
    """
    Nasumični crop — uzima random isječak minimalno 1.0s.
    Simulira da model vidi različite dijelove govora iz iste snimke.
    """
    min_uzorci = int(1.0 * sr)
    if len(signal) <= min_uzorci:
        return signal
    max_pocetak = len(signal) - min_uzorci
    pocetak = np.random.randint(0, max_pocetak)
    # Duljina cropa: između 60% i 90% originala
    max_duljina = len(signal) - pocetak
    min_duljina = max(min_uzorci, int(len(signal) * 0.6))
    duljina = np.random.randint(min(min_duljina, max_duljina), max_duljina + 1)
    return signal[pocetak:pocetak + duljina]


def primijeni_augmentaciju(signal, sr, params):
    if "time_rate"  in params: return aug_time_stretch(signal, params["time_rate"])
    if "sum_razina" in params: return aug_sum(signal, params["sum_razina"])
    if "glasnoca"   in params: return aug_glasnoca(signal, params["glasnoca"])
    if "reverb"     in params: return aug_reverb(signal, sr)
    if "telefon"    in params: return aug_telefon(signal, sr)
    if "mp3"        in params: return aug_mp3(signal, sr)
    if "crop"       in params: return aug_crop(signal, sr)
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

        sloj         = odredi_sloj(putanja_studenta)
        novi_prefiks  = prefiks_sloja(sloj)
        stari_prefiks = prefiks_prethodnog_sloja(sloj)

        print(f"  Sloj augmentacije: {sloj} (prefiks: {novi_prefiks})")

        if sloj == 1:
            izvorne = [
                f for f in os.listdir(putanja_studenta)
                if f.lower().endswith(PODRZANI_FORMATI)
                and not re.match(r'^(\d*aug_)', f)
                and not f.endswith("_konv.wav")
            ]
        else:
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

        sve = [f for f in os.listdir(putanja_studenta)
               if f.lower().endswith(".wav") and not f.endswith("_konv.wav")]
        print(f"  Ukupno snimki nakon augmentacije: {len(sve)}")

    print(f"\n{'=' * 55}")
    print(f"Gotovo! Generirano {ukupno_generirano} novih snimki.")
    print(f"Obrisi baza_cache.pkl i pokreni main.py.")


if __name__ == "__main__":
    main()