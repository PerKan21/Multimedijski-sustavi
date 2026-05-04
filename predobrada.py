"""
Predobrada audio signala
=========================
Konverzija formata, VAD segmentacija i normalizacija.

Redoslijed preprocessinga (isti za bazu i za ulazne snimke):
    1. Resample na 16kHz, mono
    2. VAD — uklanja tišinu, izlučuje govorne segmente
    3. Normalizacija glasnoće po segmentu
    4. Slanje u ECAPA-TDNN

Napomena o noise reduction:
    Blagi denoising može pomoći, ali agresivni denoising može
    uništiti fine spektralne detalje koje ECAPA-TDNN koristi
    za razlikovanje govornika. VAD je prioritet, ne denoising.
    Denoising je opcionalan i kontroliran parametrom prop_decrease
    (0.0 = isključen, 0.3 = blag, 0.75 = agresivan).
"""

import os
import logging
import numpy as np
import librosa
import noisereduce as nr
import soundfile as sf
from pydub import AudioSegment

log = logging.getLogger(__name__)


# ================================================================
# FFMPEG SETUP — cross-platform (Windows, Linux, macOS)
# ================================================================
def _pronadji_ffmpeg() -> str:
    """
    Pronalazi ffmpeg na sustavu neovisno o platformi.
    Koristi 'where' na Windowsu, 'which' na Linuxu/macOS.
    """
    import shutil
    putanja = shutil.which("ffmpeg")
    if putanja:
        return putanja
    return ""

_ffmpeg_path = _pronadji_ffmpeg()
if _ffmpeg_path:
    AudioSegment.converter = _ffmpeg_path
    # ffprobe je u istom direktoriju kao ffmpeg
    _ffprobe = os.path.join(
        os.path.dirname(_ffmpeg_path),
        "ffprobe" + (".exe" if os.name == "nt" else "")
    )
    if os.path.exists(_ffprobe):
        AudioSegment.ffprobe = _ffprobe
    log.debug(f"ffmpeg pronađen: {_ffmpeg_path}")
else:
    log.warning("ffmpeg nije pronađen na sustavu. Konverzija ne-WAV formata neće raditi.")


# ================================================================
# KONVERZIJA — pretvara m4a/mp3/ogg u wav koji soundfile može čitati
# Popravak: sprema se uz originalnu datoteku (bez nepotrebnog foldera)
# ================================================================
def u_wav(putanja: str) -> str:
    """
    Konvertira audio datoteku u WAV format ako već nije WAV.
    Konvertirana datoteka sprema se kao <ime>_konv.wav u istom direktoriju.
    """
    if putanja.lower().endswith(".wav"):
        return putanja

    dir_dat = os.path.dirname(putanja)
    ime_dat = os.path.splitext(os.path.basename(putanja))[0]
    wav_put = os.path.join(dir_dat, f"{ime_dat}_konv.wav")

    if not os.path.exists(wav_put):
        try:
            AudioSegment.from_file(putanja).export(wav_put, format="wav")
            log.debug(f"Konvertirano: {putanja} -> {wav_put}")
        except Exception as e:
            raise RuntimeError(f"Greška pri konverziji '{putanja}' u WAV: {e}") from e

    return wav_put


# ================================================================
# UCITAVANJE SIGNALA — resample + mono
# Samo tehnička konverzija, bez ikakvog preprocessinga
# ================================================================
def ucitaj_sirovi_signal(putanja: str, sr_ciljni: int) -> np.ndarray:
    """Učitava audio datoteku i pretvara u mono signal ciljanog SR. Bez preprocessinga."""
    putanja = u_wav(putanja)
    signal, sr = sf.read(putanja)
    signal = np.array(signal, dtype=np.float32)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if sr != sr_ciljni:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=sr_ciljni)
    return signal


# ================================================================
# OPCIONALNI DENOISING
# Blag denoising može pomoći, ali agresivni može pokvariti
# fine spektralne detalje koje ECAPA-TDNN koristi.
# prop_decrease=0.0 znači isključen.
# ================================================================
def ukloni_sum(signal: np.ndarray, sr: int, prop_decrease: float = 0.0) -> np.ndarray:
    """
    Opcionalna redukcija šuma. Default je 0.0 (isključeno).
    Ako je prop_decrease > 0, koristi ne-govorni dio signala
    za procjenu šuma umjesto fiksnih prvih 0.5s.
    """
    if prop_decrease <= 0.0:
        return signal
    duljina_uzorka = int(0.3 * sr)
    uzorak = signal[:duljina_uzorka] if len(signal) > duljina_uzorka else signal
    return nr.reduce_noise(
        y=signal, y_noise=uzorak, sr=sr,
        stationary=False, prop_decrease=prop_decrease
    )


# ================================================================
# NORMALIZACIJA PO SEGMENTU
# Normalizira se svaki segment zasebno — ne cijeli signal odjednom.
# ================================================================
def normaliziraj_segment(signal: np.ndarray) -> np.ndarray:
    """Peak normalizacija jednog segmenta na raspon [-0.95, 0.95]."""
    max_val = np.max(np.abs(signal))
    return signal / max_val * 0.95 if max_val > 0 else signal


# ================================================================
# VAD — detekcija i izlučivanje govornih segmenata
# ================================================================
def vad_segmentacija(signal: np.ndarray, sr: int,
                     top_db: float, min_duljina: float, spajanje: float) -> list:
    """
    Vraća listu (pocetak_s, kraj_s) govornih segmenata.
    """
    intervali = librosa.effects.split(
        signal, top_db=top_db, frame_length=512, hop_length=128
    )
    if len(intervali) == 0:
        return []

    segmenti = [(s / sr, e / sr) for s, e in intervali]

    # Spoji bliske segmente
    spojeni = [segmenti[0]]
    for poc, kraj in segmenti[1:]:
        if poc - spojeni[-1][1] <= spajanje:
            spojeni[-1] = (spojeni[-1][0], kraj)
        else:
            spojeni.append((poc, kraj))

    # Filtriraj prekratke segmente
    return [(p, k) for p, k in spojeni if (k - p) >= min_duljina]


# ================================================================
# GLAVNI PIPELINE — isti za bazu i za ulazne snimke
# ================================================================
def predobradi_signal(putanja: str, sr: int,
                      prop_decrease: float,
                      vad_top_db: float, vad_min_duljina: float,
                      vad_spajanje: float) -> tuple:
    """
    Glavni preprocessing pipeline — isti za bazu i ulazne snimke.
    Vraća (signal, vad_segmenti) gdje je signal očišćen od tišine
    i normaliziran po segmentu.
    """
    # 1. Učitaj i resample
    signal = ucitaj_sirovi_signal(putanja, sr)

    # 2. Opcionalni denoising (default: isključen)
    signal = ukloni_sum(signal, sr, prop_decrease)

    # 3. VAD — detektira govorne segmente
    vad_seg = vad_segmentacija(signal, sr, vad_top_db, vad_min_duljina, vad_spajanje)

    if not vad_seg:
        # Nema detektiranog govora — normaliziraj cijeli signal
        log.warning(f"VAD nije detektirao govor u '{putanja}', koristi se cijeli signal.")
        return normaliziraj_segment(signal), []

    # 4. Izluči segmente, normaliziraj svaki zasebno i spoji
    segmenti_signala = []
    for poc, kraj in vad_seg:
        seg = signal[int(poc * sr):int(kraj * sr)]
        segmenti_signala.append(normaliziraj_segment(seg))

    procisceni = np.concatenate(segmenti_signala)
    return procisceni, vad_seg


# ================================================================
# LEGACY — zadržano zbog kompatibilnosti s postojećim pozivima
# ================================================================
def ucitaj_signal(putanja: str, sr_ciljni: int, prop_decrease: float) -> np.ndarray:
    """
    Backwards-compatible wrapper. Koristi predobradi_signal interno.
    Preporučuje se koristiti predobradi_signal direktno.
    """
    signal, _ = predobradi_signal(
        putanja, sr_ciljni, prop_decrease,
        vad_top_db=25, vad_min_duljina=0.3, vad_spajanje=0.15
    )
    return signal
