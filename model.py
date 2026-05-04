"""
ECAPA-TDNN model, embeddinzi, baza govornika i pragovi
=======================================================
Ekstrakcija glasovnih embeddinga, izgradnja i cachiranje baze,
te automatski izračun pragova prepoznavanja.
"""

import os
import pickle
import hashlib
import logging
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel, logging as hf_logging

from predobrada import predobradi_signal, normaliziraj_segment

hf_logging.set_verbosity_error()
log = logging.getLogger(__name__)

# Globalne varijable modela — inicijaliziraju se pozivom ucitaj_model()
feature_extractor = None
model             = None


# ================================================================
# UCITAVANJE MODELA
# ================================================================
def ucitaj_model(callback=None):
    """
    Učitava ECAPA-TDNN model s HuggingFace.
    callback(poruka) se poziva za statusne poruke (korisno za GUI).
    """
    global feature_extractor, model

    # Fine-tuned model ako postoji, inace originalni pretrained
    _FINETUNED_DIR = "finetuned_model"
    _FE_DIR        = os.path.join(_FINETUNED_DIR, "feature_extractor")

    if os.path.isdir(_FINETUNED_DIR) and os.path.isdir(_FE_DIR):
        _log("Ucitavanje fine-tuned ECAPA-TDNN modela...", callback)
        feature_extractor = AutoFeatureExtractor.from_pretrained(_FE_DIR)
        model = AutoModel.from_pretrained(_FINETUNED_DIR)
    else:
        _log("Ucitavanje ECAPA-TDNN modela (UniSpeech-SAT)...", callback)
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "microsoft/unispeech-sat-base-plus-sv"
        )
        model = AutoModel.from_pretrained("microsoft/unispeech-sat-base-plus-sv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    _log(f"Model ucitan. (device: {device})", callback)


def _log(poruka, callback):
    if callback:
        try:
            callback(poruka, None)
        except TypeError:
            callback(poruka)
    else:
        print(poruka)


# ================================================================
# EKSTRAKCIJA EMBEDDINGA
# ================================================================
def izvuci_embedding_iz_signala(signal: np.ndarray, sr: int) -> np.ndarray:
    """Generira 768-dim embedding direktno iz numpy signala."""
    device = next(model.parameters()).device
    inputs = feature_extractor(signal, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return emb / np.linalg.norm(emb)


def izvuci_embedding_sa_segmentacijom(signal: np.ndarray, sr: int,
                                       trajanje: float, preklapanje: float) -> np.ndarray:
    """
    Segmentira procišćeni signal i vraća prosječni embedding.
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
                segmenti.append(normaliziraj_segment(seg))
            pocetak += korak

    if not segmenti:
        segmenti = [signal]

    embeddinzi = [izvuci_embedding_iz_signala(s, sr) for s in segmenti]
    srednji = np.mean(embeddinzi, axis=0)
    return srednji / np.linalg.norm(srednji)


# ================================================================
# EKSTRAKCIJA EMBEDDINGA IZ DATOTEKE
# ================================================================
def izvuci_embedding_iz_datoteke(putanja: str, sr: int,
                                  trajanje: float, preklapanje: float,
                                  prop_decrease: float,
                                  vad_top_db: float, vad_min_duljina: float,
                                  vad_spajanje: float) -> np.ndarray:
    signal, _ = predobradi_signal(
        putanja, sr, prop_decrease,
        vad_top_db, vad_min_duljina, vad_spajanje
    )
    return izvuci_embedding_sa_segmentacijom(signal, sr, trajanje, preklapanje)


# ================================================================
# CACHE — s invalidacijom temeljenom na sadržaju baze
# ================================================================
def _izracunaj_hash_baze(dir_baza: str, podrzani_formati: tuple) -> str:
    """
    Izračunava MD5 hash nad imenima i veličinama svih audio datoteka u bazi.
    Ako se baza promijeni (nova snimka, obrisana snimka), hash se mijenja
    i cache se automatski invalidira.
    """
    stavke = []
    for student in sorted(os.listdir(dir_baza)):
        putanja_studenta = os.path.join(dir_baza, student)
        if not os.path.isdir(putanja_studenta):
            continue
        for datoteka in sorted(os.listdir(putanja_studenta)):
            if datoteka.lower().endswith(podrzani_formati) and not datoteka.endswith("_konv.wav"):
                puna_putanja = os.path.join(putanja_studenta, datoteka)
                velicina = os.path.getsize(puna_putanja)
                stavke.append(f"{student}/{datoteka}:{velicina}")
    sadrzaj = "\n".join(stavke).encode("utf-8")
    return hashlib.md5(sadrzaj).hexdigest()


def spremi_cache(baza: dict, putanja: str, hash_baze: str = ""):
    """Sprema bazu u cache zajedno s hashom baze za invalidaciju."""
    podaci = {"baza": baza, "hash": hash_baze}
    with open(putanja, "wb") as f:
        pickle.dump(podaci, f)
    print(f"  Cache spremljen: {putanja}\n")


def ucitaj_cache(putanja: str, ocekivani_hash: str = "") -> Optional[dict]:
    """
    Učitava cache ako postoji i ako je hash valjan.
    Vraća None ako cache ne postoji ili je zastario (baza se promijenila).
    """
    if not os.path.exists(putanja):
        return None
    try:
        with open(putanja, "rb") as f:
            podaci = pickle.load(f)

        # Podrška za stari format cachea (bez hasha)
        if isinstance(podaci, dict) and "baza" in podaci and "hash" in podaci:
            if ocekivani_hash and podaci["hash"] != ocekivani_hash:
                log.info("Cache zastario (baza se promijenila), rebuild...")
                print("  Cache zastario — baza se promijenila, rebuild...")
                return None
            return podaci["baza"]
        else:
            # Stari format — dict direktno (bez hasha), prihvati ali upozori
            if isinstance(podaci, dict):
                log.warning("Cache je u starom formatu (bez hasha). Preporučuje se rebuild.")
                print("  UPOZORENJE: Cache je u starom formatu. Preporučuje se rebuild baze.")
                return podaci
            return None
    except Exception as e:
        log.warning(f"Cache nije valjan ({e}), rebuild...")
        print(f"  Cache oštećen ({e}), rebuild...")
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
    Učitava bazu govornika s istim preprocessingom kao ulazne snimke.
    Cache se automatski invalidira ako se baza promijeni.
    """
    hash_baze = _izracunaj_hash_baze(dir_baza, podrzani_formati)
    cache = ucitaj_cache(cache_putanja, ocekivani_hash=hash_baze)
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

    spremi_cache(baza, cache_putanja, hash_baze=hash_baze)
    return baza


# ================================================================
# PRAGOVI
# Popravak: faktor se sada uvijek prosljeđuje iz pozivatelja
# ================================================================
def izracunaj_pragove(baza: dict, faktor: float = 2.0) -> tuple:
    """
    Dinamički izračunava donji i gornji prag iz međusobnih distanci baze.
    Donji  = prosjek / 2
    Gornji = donji * faktor

    Napomena: default faktor je 2.0, konzistentno s FAKTOR_GORNJEG_PRAGA u main.py.
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