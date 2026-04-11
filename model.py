"""
ECAPA-TDNN model, embeddinzi, baza govornika i pragovi
=======================================================
Ekstrakcija glasovnih embeddinga, izgradnja i cachiranje baze,
te automatski izracun pragova prepoznavanja.
"""

import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel, logging as hf_logging

from predobrada import predobradi_signal, normaliziraj_segment

hf_logging.set_verbosity_error()

# Globalne varijable modela — inicijaliziraju se pozivom ucitaj_model()
feature_extractor = None
model             = None


# ================================================================
# UCITAVANJE MODELA
# ================================================================
def ucitaj_model(callback=None):
    """
    Ucitava ECAPA-TDNN model s HuggingFace.
    callback(poruka) se poziva za statusne poruke (korisno za GUI).
    """
    global feature_extractor, model

    _log("Ucitavanje ECAPA-TDNN modela (UniSpeech-SAT)...", callback)
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "microsoft/unispeech-sat-base-plus-sv"
    )
    model = AutoModel.from_pretrained("microsoft/unispeech-sat-base-plus-sv")
    model.eval()
    _log("Model ucitan.", callback)


def _log(poruka, callback):
    if callback:
        callback(poruka)
    else:
        print(poruka)


# ================================================================
# EKSTRAKCIJA EMBEDDINGA
# ================================================================
def izvuci_embedding_iz_signala(signal: np.ndarray, sr: int) -> np.ndarray:
    """Generira 768-dim embedding direktno iz numpy signala."""
    inputs = feature_extractor(signal, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return emb / np.linalg.norm(emb)


def izvuci_embedding_sa_segmentacijom(signal: np.ndarray, sr: int,
                                       trajanje: float, preklapanje: float) -> np.ndarray:
    """
    Segmentira procisceni signal i vraca prosjecni embedding.
    Signal koji dolazi ovdje je vec prociscen VAD-om i normaliziran
    po segmentu — dakle isti preprocessing kao za ulazne snimke.
    """
    segment_uzorci = int(trajanje * sr)
    korak          = segment_uzorci - int(preklapanje * sr)

    if len(signal) <= segment_uzorci:
        segmenti = [signal]
    else:
        segmenti, pocetak = [], 0
        while pocetak < len(signal):
            seg = signal[pocetak:pocetak + segment_uzorci]
            if len(seg) >= int(0.5 * sr):
                # Normalizacija po segmentu — konzistentno s predobradom
                segmenti.append(normaliziraj_segment(seg))
            pocetak += korak

    if not segmenti:
        segmenti = [signal]

    embeddinzi = [izvuci_embedding_iz_signala(s, sr) for s in segmenti]
    srednji = np.mean(embeddinzi, axis=0)
    return srednji / np.linalg.norm(srednji)


# ================================================================
# EKSTRAKCIJA EMBEDDINGA IZ DATOTEKE
# Isti pipeline za bazu i za ulazne snimke:
#   1. predobradi_signal (resample → denoising → VAD → norm po segmentu)
#   2. izvuci_embedding_sa_segmentacijom
# ================================================================
def izvuci_embedding_iz_datoteke(putanja: str, sr: int,
                                  trajanje: float, preklapanje: float,
                                  prop_decrease: float,
                                  vad_top_db: float, vad_min_duljina: float,
                                  vad_spajanje: float) -> np.ndarray:
    """
    Preprocessing pipeline isti kao za ulazne snimke:
    resample → opcionalni denoising → VAD → norm po segmentu → embedding
    """
    signal, _ = predobradi_signal(
        putanja, sr, prop_decrease,
        vad_top_db, vad_min_duljina, vad_spajanje
    )
    return izvuci_embedding_sa_segmentacijom(signal, sr, trajanje, preklapanje)


# ================================================================
# CACHE
# ================================================================
def spremi_cache(baza: dict, putanja: str):
    with open(putanja, "wb") as f:
        pickle.dump(baza, f)
    print(f"  Cache spremljen: {putanja}\n")


def ucitaj_cache(putanja: str) -> dict | None:
    if os.path.exists(putanja):
        with open(putanja, "rb") as f:
            return pickle.load(f)
    return None


# ================================================================
# IZGRADNJA BAZE
# ================================================================
def ucitaj_bazu(dir_baza: str, callback=None,
                sr: int = 16000,
                trajanje: float = 1.5,
                preklapanje: float = 0.3,
                prop_decrease: float = 0.0,
                vad_top_db: float = 25,
                vad_min_duljina: float = 0.3,
                vad_spajanje: float = 0.15,
                podrzani_formati: tuple = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".mp4"),
                cache_putanja: str = "baza_cache.pkl") -> dict:
    """
    Ucitava bazu govornika s istim preprocessingom kao ulazne snimke.
    Default prop_decrease=0.0 (denoising iskljucen).
    """
    cache = ucitaj_cache(cache_putanja)
    if cache is not None:
        _log(f"Ucitan cache s {len(cache)} studenata.", callback)
        return cache

    baza     = {}
    studenti = [s for s in os.listdir(dir_baza)
                if os.path.isdir(os.path.join(dir_baza, s))]

    for ime_studenta in tqdm(studenti, desc="  Izgradnja baze", leave=False):
        putanja_studenta = os.path.join(dir_baza, ime_studenta)
        embeddinzi = []
        for datoteka in os.listdir(putanja_studenta):
            if (datoteka.lower().endswith(podrzani_formati)
                    and not datoteka.endswith("_konv.wav")):
                putanja = os.path.join(putanja_studenta, datoteka)
                try:
                    emb = izvuci_embedding_iz_datoteke(
                        putanja, sr, trajanje, preklapanje,
                        prop_decrease, vad_top_db, vad_min_duljina, vad_spajanje
                    )
                    embeddinzi.append(emb)
                except Exception as e:
                    print(f"\n  UPOZORENJE: Ne mogu ucitati {datoteka} ({e})")

        if embeddinzi:
            baza[ime_studenta] = [e / np.linalg.norm(e) for e in embeddinzi]
            if callback:
                callback(ime_studenta, len(embeddinzi))
            else:
                print(f"  Ucitano: {ime_studenta}  ({len(embeddinzi)} snimki)")
        else:
            print(f"  UPOZORENJE: Nema snimki za '{ime_studenta}', preskacam.")

    spremi_cache(baza, cache_putanja)
    return baza


# ================================================================
# PRAGOVI
# ================================================================
def izracunaj_pragove(baza: dict, faktor: float = 1.8) -> tuple:
    """
    Dinamicki izracunava donji i gornji prag iz medusobnih distanci baze.
    Donji  = prosjek / 2
    Gornji = donji * faktor
    """
    imena   = list(baza.keys())
    srednji = {ime: np.mean(baza[ime], axis=0) for ime in imena}
    distance = []
    for i in range(len(imena)):
        for j in range(i + 1, len(imena)):
            dist = float(1 - np.dot(srednji[imena[i]], srednji[imena[j]]))
            distance.append(dist)
    if not distance:
        return 0.0, 0.0

    prag_donji  = np.mean(distance) / 2
    prag_gornji = prag_donji * faktor
    print(f"  Donji prag (siguran):   {prag_donji:.4f}")
    print(f"  Gornji prag (uljez):    {prag_gornji:.4f}")
    print(f"  (prosjek distanci baze: {np.mean(distance):.4f})")
    return prag_donji, prag_gornji