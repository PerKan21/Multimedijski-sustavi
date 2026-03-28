"""
Speaker Recognition - x-vector / ECAPA-TDNN
=============================================
Pristup: Prethodno istrenirani UniSpeech-SAT model (Microsoft HuggingFace)
         baziran na ECAPA-TDNN arhitekturi, fine-tunan za speaker verification
         na VoxCeleb datasetu.

Struktura direktorija:
    baza/
        Ime Studenta/           <- jedan folder po studentu
            snimka1.wav
            snimka2.m4a
    snimke/
        snimka_01.wav           <- ulazne snimke, bez podfoldera
        snimka_02.m4a
        ...

Instalacija:
    pip install transformers torch soundfile librosa pydub noisereduce tqdm

"""

# ================================================================
# IMPORTS - ucitavanje svih potrebnih biblioteka
# ================================================================
import os
import warnings
import numpy as np
import subprocess
import torch
import soundfile as sf
import librosa
import noisereduce as nr
import pickle
from datetime import datetime
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel, logging as hf_logging
from pydub import AudioSegment

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()


# ================================================================
# FFMPEG SETUP - pronalazak ffmpeg-a za konverziju audio formata
# ================================================================
ffmpeg_path = subprocess.run(
    "where ffmpeg", capture_output=True, text=True, shell=True
).stdout.strip().split("\n")[0]
if ffmpeg_path:
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffprobe   = ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe")


# ================================================================
# POSTAVKE - sve konfiguracijske varijable na jednom mjestu
# ================================================================
SR               = 16000
DIR_BAZA         = "baza"
DIR_SNIMKE       = "snimke"
PODRZANI_FORMATI = (".wav", ".mp3", ".m4a", ".ogg", ".flac")

# Postavke segmentacije (za dulje snimke)
SEGMENT_TRAJANJE    = 3    # Duljina jednog segmenta u sekundama
SEGMENT_PREKLAPANJE = 0.5  # Preklapanje između segmenata u sekundama

# Postavke uklanjanja šuma
SUM_PROP_DECREASE = 0.75   # Agresivnost uklanjanja suma (0-1)

# Cache embeddinga - obrisi baza_cache.pkl ako promijenis snimke u bazi
CACHE_PUTANJA = "baza_cache.pkl"

# Minimalni prag ispod kojeg optimalni prag ne smije ici
MINIMALNI_PRAG = 0.08


# ================================================================
# UCITAVANJE MODELA - preuzimanje ECAPA-TDNN modela s HuggingFace
#                     (prvi put skida ~360MB, nakon toga koristi cache)
# ================================================================
print("Ucitavanje ECAPA-TDNN modela (UniSpeech-SAT)...")
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/unispeech-sat-base-plus-sv")
model = AutoModel.from_pretrained("microsoft/unispeech-sat-base-plus-sv")
model.eval()
print("Model ucitan.\n")


# ================================================================
# KONVERZIJA - pretvara m4a/mp3/ogg u wav koji soundfile moze citati
# ================================================================
def u_wav(putanja: str) -> str:
    if putanja.lower().endswith(".wav"):
        return putanja
    dir_dat  = os.path.dirname(putanja)
    ime_dat  = os.path.splitext(os.path.basename(putanja))[0]
    konv_dir = os.path.join(dir_dat, f"{ime_dat}_konv")
    os.makedirs(konv_dir, exist_ok=True)
    wav_put  = os.path.join(konv_dir, f"{ime_dat}.wav")
    if not os.path.exists(wav_put):
        AudioSegment.from_file(putanja).export(wav_put, format="wav")
    return wav_put


# ================================================================
# PREDOBRADA - uklanjanje suma i normalizacija glasnoce
#              koristi prvih 0.5s kao uzorak suma (pretpostavka: tišina)
#              normalizacija izjednacava glasnocu razlicitih mikrofona
# ================================================================
def ukloni_sum(signal: np.ndarray, sr: int) -> np.ndarray:
    duljina_uzorka = int(0.5 * sr)
    uzorak = signal[:duljina_uzorka] if len(signal) > duljina_uzorka else signal
    return nr.reduce_noise(
        y=signal,
        y_noise=uzorak,
        sr=sr,
        stationary=False,
        prop_decrease=SUM_PROP_DECREASE
    )


def normaliziraj_glasnocu(signal: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        return signal / max_val * 0.95
    return signal


# ================================================================
# EKSTRAKCIJA EMBEDDINGA - generira 768-dimenzionalni vektor koji
#                          reprezentira identitet govornika.
#                          Za dulje snimke dijeli na segmente i
#                          uzima prosjecni embedding.
# ================================================================
def izvuci_embedding(putanja_datoteke: str) -> np.ndarray:
    putanja_datoteke = u_wav(putanja_datoteke)
    signal, sr = sf.read(putanja_datoteke)
    signal = np.array(signal, dtype=np.float32)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if sr != SR:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=SR)

    # Predobrada
    signal = ukloni_sum(signal, SR)
    signal = normaliziraj_glasnocu(signal)

    # Segmentacija za dulje snimke
    segment_uzorci = int(SEGMENT_TRAJANJE * SR)
    korak          = segment_uzorci - int(SEGMENT_PREKLAPANJE * SR)

    if len(signal) <= segment_uzorci:
        segmenti = [signal]
    else:
        segmenti = []
        pocetak  = 0
        while pocetak < len(signal):
            segment = signal[pocetak:pocetak + segment_uzorci]
            if len(segment) >= int(0.5 * SR):
                segmenti.append(segment)
            pocetak += korak

    embeddinzi = []
    for segment in segmenti:
        inputs = feature_extractor(segment, sampling_rate=SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddinzi.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

    srednji = np.mean(embeddinzi, axis=0)
    return srednji / np.linalg.norm(srednji)


# ================================================================
# CACHE EMBEDDINGA - sprema/ucitava izracunate embeddings kako bi
#                    se izbjeglo ponavljanje dugotrajnog izracuna.
#                    Obrisi baza_cache.pkl ako promijenis snimke u bazi.
# ================================================================
def spremi_cache(baza: dict):
    with open(CACHE_PUTANJA, "wb") as f:
        pickle.dump(baza, f)


def ucitaj_cache() -> dict | None:
    if os.path.exists(CACHE_PUTANJA):
        with open(CACHE_PUTANJA, "rb") as f:
            return pickle.load(f)
    return None


# ================================================================
# IZGRADNJA BAZE - ucitava sve podfoldere iz baze i za svakog
#                  studenta racuna srednji embedding od svih snimki.
#                  Koristi cache ako postoji.
# ================================================================
def ucitaj_bazu(dir_baza: str) -> dict:
    cache = ucitaj_cache()
    if cache is not None:
        print(f"  Ucitan cache s {len(cache)} studenata.")
        print(f"  (Obrisi {CACHE_PUTANJA} ako si promijenio snimke u bazi)\n")
        return cache

    baza     = {}
    studenti = [s for s in os.listdir(dir_baza) if os.path.isdir(os.path.join(dir_baza, s))]

    for ime_studenta in tqdm(studenti, desc="  Izgradnja baze", leave=False):
        putanja_studenta = os.path.join(dir_baza, ime_studenta)
        embeddinzi = []
        for datoteka in os.listdir(putanja_studenta):
            if datoteka.lower().endswith(PODRZANI_FORMATI) and not datoteka.endswith("_konv.wav"):
                putanja = os.path.join(putanja_studenta, datoteka)
                try:
                    embeddinzi.append(izvuci_embedding(putanja))
                except Exception as e:
                    print(f"\n  UPOZORENJE: Ne mogu ucitati {datoteka} ({e})")
        if embeddinzi:
            # Spremi listu svih embeddinga umjesto jednog prosjeka
            # Pri identifikaciji uzimamo minimalnu udaljenost od svih referentnih
            baza[ime_studenta] = [e / np.linalg.norm(e) for e in embeddinzi]
            print(f"  Ucitano: {ime_studenta}  ({len(embeddinzi)} snimki)")
        else:
            print(f"  UPOZORENJE: Nema snimki za '{ime_studenta}', preskacam.")

    spremi_cache(baza)
    print(f"  Cache spremljen: {CACHE_PUTANJA}\n")
    return baza


# ================================================================
# OPTIMALNI PRAG - automatski izracunava prag kao polovicu prosjecne
#                  medusobne udaljenosti govornika u bazi.
#                  Nikad ne ide ispod MINIMALNI_PRAG.
# ================================================================
def izracunaj_prag(baza: dict) -> float:
    # Koristi srednji embedding po studentu samo za izracun praga
    imena    = list(baza.keys())
    srednji  = {ime: np.mean(baza[ime], axis=0) for ime in imena}
    distance = []
    for i in range(len(imena)):
        for j in range(i + 1, len(imena)):
            dist = float(1 - np.dot(srednji[imena[i]], srednji[imena[j]]))
            distance.append(dist)
    if not distance:
        return MINIMALNI_PRAG
    optimalni = max(np.mean(distance) / 2, MINIMALNI_PRAG)
    print(f"  Optimalni prag: {optimalni:.4f}  "
          f"(prosjek medusobnih distanci: {np.mean(distance):.4f})")
    return optimalni


# ================================================================
# IDENTIFIKACIJA - usporeduje ulazni embedding s bazom kosinusnom
#                  udaljenoscu i vraca studenta s najmanjom udaljenoscu
#                  ako je ispod praga, inace vraca None
# ================================================================
def identificiraj(embedding_ulaz: np.ndarray, baza: dict, prag: float):
    # Za svakog studenta uzima minimalnu udaljenost od svih njegovih referentnih snimki
    # Bolje od prosjeka jer jedna dobra snimka moze "spasiti" prepoznavanje
    najbolji_student    = None
    najmanja_udaljenost = float("inf")
    emb = embedding_ulaz / np.linalg.norm(embedding_ulaz)
    for ime, referentni in baza.items():
        min_dist = min(float(1 - np.dot(emb, ref)) for ref in referentni)
        if min_dist < najmanja_udaljenost:
            najmanja_udaljenost = min_dist
            najbolji_student    = ime
    if najmanja_udaljenost <= prag:
        return najbolji_student, najmanja_udaljenost
    return None, najmanja_udaljenost


# ================================================================
# GLAVNI PROGRAM - orkestrira cijeli tok: baza -> snimke -> rezultati
# ================================================================
def main():
    os.makedirs(DIR_BAZA,   exist_ok=True)
    os.makedirs(DIR_SNIMKE, exist_ok=True)

    datum_vrijeme = datetime.now().strftime("%d.%m.%Y. u %H:%M:%S")

    print("=" * 50)
    print("Sustav za popisivanje studenata (ECAPA-TDNN)")
    print("=" * 50)

    # Ucitaj bazu
    print("\n[1] Ucitavanje baze govornika...")
    baza = ucitaj_bazu(DIR_BAZA)
    print(f"    Ukupno studenata u bazi: {len(baza)}")
    prag = izracunaj_prag(baza)

    prisutnost = {ime: False for ime in baza}
    uljezi     = []
    rezultati  = []

    # Obradi snimke - progress bar bez ispisa po svakoj snimci
    print("\n[2] Obrada ulaznih snimaka...")
    snimke = sorted([
        f for f in os.listdir(DIR_SNIMKE)
        if f.lower().endswith(PODRZANI_FORMATI) and not f.endswith("_konv.wav")
    ])

    if not snimke:
        print("  Nema snimki u folderu 'snimke'.")
    else:
        for naziv in tqdm(snimke, desc="  Napredak", leave=True):
            putanja       = os.path.join(DIR_SNIMKE, naziv)
            embedding     = izvuci_embedding(putanja)
            student, dist = identificiraj(embedding, baza, prag)

            if student:
                prisutnost[student] = True
                rezultati.append((naziv, student, dist, False))
            else:
                uljezi.append(naziv)
                rezultati.append((naziv, None, dist, True))

        # Ispisi sve rezultate nakon sto progress bar zavrsi
        print()
        for naziv, student, dist, je_uljez in rezultati:
            if not je_uljez:
                print(f"  {naziv:30s} -> PREPOZNAT:   {student} (dist={dist:.4f})")
            else:
                print(f"  {naziv:30s} -> NIJE U BAZI  (dist={dist:.4f})")

    # Popis prisutnosti
    print("\n" + "=" * 50)
    print("POPIS PRISUTNOSTI")
    print("=" * 50)
    for ime, prisutan in prisutnost.items():
        print(f"  [{'+'  if prisutan else '-'}] {ime}")

    # Spremi u datoteku
    prisutni  = [ime for ime, p in prisutnost.items() if p]
    nedostaju = [ime for ime, p in prisutnost.items() if not p]
    sep = "-" * 30 + "\n"

    with open("prisutnost.txt", "w", encoding="utf-8") as f:
        f.write(f"Analiza provedena: {datum_vrijeme}\n")
        f.write(sep)
        f.write(f"Ukupno prisutnih studenata (prepoznatih): {len(prisutni)}/{len(baza)}\n")
        if prisutni:
            for ime in prisutni:
                f.write(f"  + {ime}\n")
        else:
            f.write("  (nitko nije prisutan)\n")

        f.write(sep)
        f.write("Studenti koji nedostaju:\n")
        if nedostaju:
            for ime in nedostaju:
                f.write(f"  - {ime}\n")
        else:
            f.write("  SVI SU STUDENTI PRISUTNI\n")

        f.write(sep)
        f.write(f"Broj uljeza (neprepoznatih): {len(uljezi)}\n")
        if uljezi:
            for naziv in uljezi:
                f.write(f"  - {naziv}\n")
        else:
            f.write("  NEMA ULJEZA\n")

    print("\nRezultati spremljeni u: prisutnost.txt")


if __name__ == "__main__":
    main()