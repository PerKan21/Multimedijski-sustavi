"""
Augmentacija audio podataka za speaker recognition
====================================================

Konzervativna augmentacija referentnih snimki za prepoznavanje govornika.

Cilj:
    - povećati robusnost na šum, glasnoću, kompresiju i blage promjene snimanja
    - NE mijenjati identitet glasa agresivnim transformacijama

Namjerno maknuto:
    - time stretch 0.95 / 1.05  -> može promijeniti temporalne karakteristike govora
    - jak šum 0.007             -> može degradirati speaker embedding
    - glasnoća 0.6 / 1.4        -> preagresivno
    - telefon bandpass          -> previše mijenja frekvencijski sadržaj ako test nije telefonski
    - višeslojna augmentacija   -> ne radimo augmentacije augmentacija

Instalacija:
    pip install librosa soundfile numpy scipy pydub
"""

# ================================================================
# IMPORTS
# ================================================================
import io
import logging
import os
import re
import tempfile
import warnings

import librosa
import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# ================================================================
# POSTAVKE
# ================================================================
SR = 16000
DIR_BAZA = "baza"
PODRZANI_FORMATI = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".mp4")

# Važno:
# - radi se samo jedan sloj augmentacije iz ORIGINALNIH snimki
# - ne radi se aug_ od već augmentiranih snimki
PREFIKS_AUG = "aug_"

AUGMENTACIJE = [
    # Blagi šum — simulira realne uvjete snimanja bez mijenjanja identiteta glasa
    ("sum_mali", {"sum_razina": 0.002}),
    ("sum_blagi", {"sum_razina": 0.004}),

    # Blaga promjena glasnoće — realno jer studenti ne govore uvijek jednako glasno
    ("glasnoca_tise", {"glasnoca": 0.85}),
    ("glasnoca_glasnije", {"glasnoca": 1.15}),

    # Blagi reverb — koristan ako su snimke iz prostorije
    ("reverb_blagi", {"reverb": True}),

    # MP3 kompresija — korisno ako ulaz dolazi iz WhatsAppa, mobitela, mp3/m4a itd.
    ("mp3", {"mp3": True}),

    # Blagi crop — uzima 80–95% snimke, ne agresivno rezanje
    ("crop_blagi", {"crop": True}),
]


# ================================================================
# POMOĆNE FUNKCIJE
# ================================================================
def je_augmentirana_datoteka(naziv: str) -> bool:
    """Provjerava je li datoteka već augmentirana."""
    return naziv.startswith(PREFIKS_AUG) or re.match(r"^\d+aug_", naziv) is not None


def je_podrzani_audio(naziv: str) -> bool:
    """Provjerava je li datoteka podržani audio format."""
    n = naziv.lower()
    return n.endswith(PODRZANI_FORMATI) and not n.endswith("_konv.wav") and not n.endswith("_temp.wav")


def izlazni_naziv_vec_postoji(putanja_studenta: str, ime_bez_ext: str, aug_naziv: str) -> bool:
    """Provjerava postoji li već output bez obzira na ekstenziju originala."""
    izlaz = os.path.join(putanja_studenta, f"{PREFIKS_AUG}{ime_bez_ext}_{aug_naziv}.wav")
    return os.path.exists(izlaz)


# ================================================================
# UČITAVANJE AUDIO
# ================================================================
def ucitaj_audio(putanja: str) -> tuple[np.ndarray, int]:
    """
    Učitava audio, pretvara ga u mono float32 i resampla na 16 kHz.

    Popravak u odnosu na staru verziju:
    - za ne-WAV datoteke koristi sigurni privremeni file
    - temp file se briše i ako se dogodi greška
    """
    temp_path = None

    try:
        if not putanja.lower().endswith(".wav"):
            from pydub import AudioSegment

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_path = tmp.name

            AudioSegment.from_file(putanja).export(temp_path, format="wav")
            signal, sr = sf.read(temp_path)
        else:
            signal, sr = sf.read(putanja)

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    if sr != SR:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=SR)

    return signal.astype(np.float32), SR


# ================================================================
# AUGMENTACIJE
# ================================================================
def aug_sum(signal: np.ndarray, sum_razina: float) -> np.ndarray:
    """Dodaje blagi Gaussov šum."""
    noise = np.random.normal(0, sum_razina, len(signal)).astype(np.float32)
    return signal + noise


def aug_glasnoca(signal: np.ndarray, glasnoca: float) -> np.ndarray:
    """Blago mijenja glasnoću."""
    return signal * glasnoca


def aug_reverb(signal: np.ndarray, sr: int) -> np.ndarray:
    """
    Blagi umjetni reverb.

    Stara verzija je imala prejake refleksije:
        0.40, 0.25, 0.15, 0.08

    Ova verzija je blaža:
        0.20, 0.10, 0.05, 0.03
    """
    if len(signal) == 0:
        return signal

    ir_duljina = int(0.1 * sr)
    ir = np.zeros(ir_duljina, dtype=np.float32)

    ir[0] = 1.0
    ir[int(0.02 * sr)] = 0.20
    ir[int(0.04 * sr)] = 0.10
    ir[int(0.07 * sr)] = 0.05
    ir[int(0.09 * sr)] = 0.03

    reverb_signal = np.convolve(signal, ir)[:len(signal)].astype(np.float32)

    max_reverb = np.max(np.abs(reverb_signal))
    max_orig = np.max(np.abs(signal))

    if max_reverb <= 0 or max_orig <= 0:
        return signal

    return reverb_signal / max_reverb * max_orig


def aug_mp3(signal: np.ndarray, sr: int) -> np.ndarray:
    """
    Prolazi signal kroz MP3 enkoder/dekoder i vraća ga kao numpy signal.

    Popravak:
    - ako MP3 augmentacija ne uspije, greška se propusti gore,
      a main() ispiše upozorenje i nastavi s ostalim augmentacijama
    """
    from pydub import AudioSegment

    buf_wav = io.BytesIO()
    sf.write(buf_wav, signal, sr, format="WAV")
    buf_wav.seek(0)

    audio = AudioSegment.from_wav(buf_wav)

    buf_mp3 = io.BytesIO()
    audio.export(buf_mp3, format="mp3", bitrate="128k")
    buf_mp3.seek(0)

    audio_dec = AudioSegment.from_mp3(buf_mp3)

    samples = np.array(audio_dec.get_array_of_samples(), dtype=np.float32)
    samples /= float(2 ** (audio_dec.sample_width * 8 - 1))

    if audio_dec.channels > 1:
        samples = samples.reshape(-1, audio_dec.channels).mean(axis=1)

    if audio_dec.frame_rate != sr:
        samples = librosa.resample(samples, orig_sr=audio_dec.frame_rate, target_sr=sr)

    # Poravnaj duljinu da ne dobiješ čudno duže/kraće snimke
    if len(samples) >= len(signal):
        return samples[:len(signal)].astype(np.float32)

    # Ako je dekodirani signal kraći, dopuni nulama do originalne duljine
    out = np.zeros_like(signal, dtype=np.float32)
    out[:len(samples)] = samples.astype(np.float32)
    return out


def aug_crop(signal: np.ndarray, sr: int) -> np.ndarray:
    """
    Blagi crop — uzima 80–95% originalne snimke.

    Stara verzija 60–90% je mogla previše odrezati govor.
    Za speaker recognition želimo malo varijacije, ali ne promijeniti sadržaj previše.
    """
    min_uzorci = int(1.0 * sr)

    if len(signal) <= min_uzorci:
        return signal

    min_duljina = max(min_uzorci, int(len(signal) * 0.80))
    max_duljina = max(min_duljina, int(len(signal) * 0.95))

    if max_duljina >= len(signal):
        max_duljina = len(signal)

    if min_duljina >= len(signal):
        return signal

    duljina = np.random.randint(min_duljina, max_duljina + 1)
    max_pocetak = len(signal) - duljina

    if max_pocetak <= 0:
        return signal[:duljina]

    pocetak = np.random.randint(0, max_pocetak + 1)
    return signal[pocetak:pocetak + duljina]


def primijeni_augmentaciju(signal: np.ndarray, sr: int, params: dict) -> np.ndarray:
    """Primjenjuje jednu augmentaciju prema parametrima."""
    if "sum_razina" in params:
        return aug_sum(signal, params["sum_razina"])

    if "glasnoca" in params:
        return aug_glasnoca(signal, params["glasnoca"])

    if "reverb" in params:
        return aug_reverb(signal, sr)

    if "mp3" in params:
        return aug_mp3(signal, sr)

    if "crop" in params:
        return aug_crop(signal, sr)

    return signal


def normaliziraj(signal: np.ndarray) -> np.ndarray:
    """
    Peak normalizacija na [-0.95, 0.95].

    Napomena:
    - korisno je jer augmentacije glasnoće mogu dovesti do clippinga
    - ne mijenja relativni oblik signala, samo skalu
    """
    if len(signal) == 0:
        return signal

    max_val = np.max(np.abs(signal))
    if max_val <= 0:
        return signal.astype(np.float32)

    return (signal / max_val * 0.95).astype(np.float32)


# ================================================================
# GLAVNI PROGRAM
# ================================================================
def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    print("=" * 55)
    print("Augmentacija audio podataka")
    print("=" * 55)

    if not os.path.isdir(DIR_BAZA):
        print(f"GREŠKA: Mapa baze ne postoji: {DIR_BAZA}")
        return

    ukupno_generirano = 0

    for ime_studenta in sorted(os.listdir(DIR_BAZA)):
        putanja_studenta = os.path.join(DIR_BAZA, ime_studenta)

        if not os.path.isdir(putanja_studenta):
            continue

        print(f"\nStudent: {ime_studenta}")

        # Samo originalne snimke. Ne augmentiramo već augmentirane snimke.
        izvorne = [
            f for f in sorted(os.listdir(putanja_studenta))
            if je_podrzani_audio(f)
            and not je_augmentirana_datoteka(f)
        ]

        if not izvorne:
            print("  Nema originalnih snimki za augmentaciju, preskačem.")
            continue

        print(f"  Originalnih snimki za augmentaciju: {len(izvorne)}")
        print(f"  Broj augmentacija po snimci: {len(AUGMENTACIJE)}")

        generirano_student = 0

        for naziv_snimke in izvorne:
            putanja_snimke = os.path.join(putanja_studenta, naziv_snimke)
            ime_bez_ext = os.path.splitext(naziv_snimke)[0]

            try:
                signal, sr = ucitaj_audio(putanja_snimke)
            except Exception as e:
                print(f"  UPOZORENJE: Ne mogu učitati {naziv_snimke} ({e})")
                continue

            if len(signal) == 0:
                print(f"  UPOZORENJE: Prazna snimka: {naziv_snimke}")
                continue

            for aug_naziv, aug_params in AUGMENTACIJE:
                izlaz_naziv = f"{PREFIKS_AUG}{ime_bez_ext}_{aug_naziv}.wav"
                izlaz_put = os.path.join(putanja_studenta, izlaz_naziv)

                if os.path.exists(izlaz_put):
                    continue

                try:
                    aug_signal = primijeni_augmentaciju(signal, sr, aug_params)
                    aug_signal = normaliziraj(aug_signal)
                    sf.write(izlaz_put, aug_signal, sr)
                    ukupno_generirano += 1
                    generirano_student += 1

                except Exception as e:
                    print(f"  UPOZORENJE: '{aug_naziv}' nije uspjela za {naziv_snimke} ({e})")
                    log.warning("Augmentacija '%s' failed za %s: %s", aug_naziv, naziv_snimke, e)

        sve_wav = [
            f for f in os.listdir(putanja_studenta)
            if f.lower().endswith(".wav") and not f.endswith("_konv.wav") and not f.endswith("_temp.wav")
        ]

        print(f"  Generirano novih snimki za studenta: {generirano_student}")
        print(f"  Ukupno WAV snimki nakon augmentacije: {len(sve_wav)}")

    print(f"\n{'=' * 55}")
    print(f"Gotovo! Generirano {ukupno_generirano} novih snimki.")
    print("Obriši baza_cache.pkl i pokreni main.py.")


if __name__ == "__main__":
    main()
