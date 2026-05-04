"""
Fine-tuning UniSpeech-SAT za prepoznavanje govornika
=====================================================
Prilagođava pretrained model na glasove studenata iz baze.

Strategija (sprječavanje overfittinga na malom datasetu):
  Faza 1 — Linear Probing:
      Zamrznuti su svi Transformer slojevi.
      Trenira se samo projection head (768 → 256 → N_studenata).
      Niska šansa overfittinga, brzo konvergira.

  Faza 2 — Partial Unfreeze:
      Odmrzavaju se zadnja 2 Transformer bloka + projection head.
      Jako niska learning rate da ne uništi pretrained weighte.
      Early stopping sprječava overfitting.

Izlaz:
  finetuned_model/   — fine-tuned model spreman za učitavanje
  finetuned_model/feature_extractor/ — kopija feature extractora

Pokretanje:
  python finetune.py             → interaktivni mod
  python finetune.py --faza 1    → samo linear probing
  python finetune.py --faza 2    → samo partial unfreeze (treba fazu 1)
  python finetune.py --oba       → faza 1 + faza 2

Instalacija:
  pip install transformers torch soundfile librosa tqdm

VAŽNO: Napravi backup originalnog modela iz HuggingFace cachea
       prije pokretanja fine-tuninga!
"""

import os
import sys
import random
import argparse
import warnings
import logging

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, AutoModel, logging as hf_logging
from tqdm import tqdm

import main as cfg
from predobrada import predobradi_signal, normaliziraj_segment

hf_logging.set_verbosity_error()
logging.basicConfig(level=logging.WARNING)

# ================================================================
# POSTAVKE FINE-TUNINGA
# ================================================================
MODEL_NAZIV       = "microsoft/unispeech-sat-base-plus-sv"
IZLAZ_DIR         = "finetuned_model"

VAL_SPLIT         = 0.20          # 20% snimki po studentu za validaciju
SEED              = 42

# Faza 1 — Linear Probing
F1_EPOHE          = 30
F1_LR             = 3e-4
F1_BATCH          = 8
F1_PATIENCE       = 8            # Early stopping: čekaj N epoha bez poboljšanja

# Faza 2 — Partial Unfreeze
F2_EPOHE          = 20
F2_LR             = 5e-6         # Jako niska LR da ne uništimo pretrained weighte
F2_BATCH          = 4            # Manji batch zbog gradijenata kroz više slojeva
F2_PATIENCE       = 6
F2_ODMRZNI_BLOKOVA = 2           # Koliko zadnjih Transformer blokova odmrznuti

# Augmentacija za trening (blaga, samo za datasete)
AUG_SUM_RAZINA    = 0.003        # Gaussov šum
AUG_GLASNOCA_VAR  = 0.10         # ±10% glasnoća


# ================================================================
# DATASET
# ================================================================
class GlasovniDataset(Dataset):
    """
    Dataset koji učitava audio iz baze/ foldera.
    Svaki uzorak je (signal, student_idx).
    Podržava blagu online augmentaciju za train split.
    """

    def __init__(self, snimke: list, augmentacija: bool = False):
        """
        snimke = lista (putanja, student_idx)
        """
        self.snimke       = snimke
        self.augmentacija = augmentacija

    def __len__(self):
        return len(self.snimke)

    def __getitem__(self, idx):
        putanja, student_idx = self.snimke[idx]

        signal, _ = predobradi_signal(
            putanja,
            sr=cfg.SR,
            prop_decrease=0.0,
            vad_top_db=cfg.VAD_TOP_DB,
            vad_min_duljina=cfg.VAD_MIN_DULJINA,
            vad_spajanje=cfg.VAD_SPAJANJE,
        )

        # Normaliziraj
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val * 0.95

        # Online augmentacija samo za train
        if self.augmentacija:
            signal = self._augmentiraj(signal)

        # Izreži ili popuni na fiksnu duljinu (3 sekunde)
        ciljna_duljina = cfg.SR * 3
        if len(signal) >= ciljna_duljina:
            # Random crop
            pocetak = random.randint(0, len(signal) - ciljna_duljina)
            signal  = signal[pocetak:pocetak + ciljna_duljina]
        else:
            # Zero-pad
            signal = np.pad(signal, (0, ciljna_duljina - len(signal)))

        return torch.tensor(signal, dtype=torch.float32), student_idx

    def _augmentiraj(self, signal: np.ndarray) -> np.ndarray:
        """Blaga online augmentacija: šum + glasnoća."""
        # Gaussov šum
        if random.random() < 0.5:
            signal = signal + np.random.normal(0, AUG_SUM_RAZINA, len(signal))

        # Glasnoća ±10%
        if random.random() < 0.5:
            faktor = 1.0 + random.uniform(-AUG_GLASNOCA_VAR, AUG_GLASNOCA_VAR)
            signal = signal * faktor

        return signal.astype(np.float32)


# ================================================================
# MODEL S PROJECTION HEADOM
# ================================================================
class UniSpeechSATSpeaker(nn.Module):
    """
    UniSpeech-SAT + projection head za klasifikaciju govornika.

    Arhitektura:
      UniSpeech-SAT → mean pooling → LayerNorm → Linear(768→256) →
      ReLU → Dropout(0.3) → Linear(256→N) → logiti

    Projection head je mali MLP koji mapira embedding na identitet govornika.
    """

    def __init__(self, n_studenata: int, dropout: float = 0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_NAZIV)
        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_studenata)
        )

    def forward(self, input_values: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask
        )
        # Mean pooling po vremenskoj osi
        emb = outputs.last_hidden_state.mean(dim=1)
        return self.head(emb)

    def izvuci_embedding(self, input_values: torch.Tensor) -> torch.Tensor:
        """Vraća L2-normalizirani embedding (za inference)."""
        with torch.no_grad():
            outputs = self.backbone(input_values=input_values)
            emb = outputs.last_hidden_state.mean(dim=1)
            return F.normalize(emb, dim=-1)

    def zamrzni_backbone(self):
        """Zamrzava sve parametre backbonea."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  Backbone zamrznut.")

    def odmrzni_zadnje_blokove(self, n: int = 2):
        """
        Odmrzava zadnjih N Transformer blokova + final layer norm.
        Ostali slojevi ostaju zamrznuti.
        Generički pristup — radi s bilo kojom transformers arhitekturom.
        """
        # Pronađi encoder generičkim pristupom kroz sve children
        encoder = None
        for _, modul in self.backbone.named_children():
            for pod_naziv, pod_modul in modul.named_children():
                if pod_naziv == "encoder":
                    encoder = pod_modul
                    break
            if encoder is not None:
                break

        if encoder is None:
            print("  UPOZORENJE: Encoder nije pronađen, odmrzavam sve backbone parametre.")
            for param in self.backbone.parameters():
                param.requires_grad = True
        elif hasattr(encoder, "layers"):
            ukupno_blokova = len(encoder.layers)
            for i, blok in enumerate(encoder.layers):
                if i >= ukupno_blokova - n:
                    for param in blok.parameters():
                        param.requires_grad = True
            # Odmrzni final layer norm ako postoji
            if hasattr(encoder, "layer_norm"):
                for param in encoder.layer_norm.parameters():
                    param.requires_grad = True
        else:
            print("  UPOZORENJE: Encoder nema 'layers', odmrzavam cijeli encoder.")
            for param in encoder.parameters():
                param.requires_grad = True

        odmrznuti = sum(p.numel() for p in self.parameters() if p.requires_grad)
        ukupno    = sum(p.numel() for p in self.parameters())
        print(f"  Odmrznuto zadnjih {n} blokova.")
        print(f"  Parametara za trening: {odmrznuti:,} / {ukupno:,} "
              f"({100*odmrznuti/ukupno:.1f}%)")


# ================================================================
# UČITAVANJE PODATAKA
# ================================================================
def ucitaj_podatke(dir_baza: str) -> tuple:
    """
    Čita sve audio datoteke iz baze i dijeli ih na train/val.
    Vraća (train_snimke, val_snimke, student2idx, idx2student).
    """
    studenti = sorted([
        s for s in os.listdir(dir_baza)
        if os.path.isdir(os.path.join(dir_baza, s))
    ])

    student2idx = {ime: idx for idx, ime in enumerate(studenti)}
    idx2student = {idx: ime for ime, idx in student2idx.items()}

    train_snimke = []
    val_snimke   = []

    random.seed(SEED)

    for student in studenti:
        putanja_studenta = os.path.join(dir_baza, student)
        snimke = [
            os.path.join(putanja_studenta, f)
            for f in sorted(os.listdir(putanja_studenta))
            if f.lower().endswith(cfg.PODRZANI_FORMATI)
            and not f.endswith("_konv.wav")
            and not f.startswith("aug_")   # Samo originalne za val
        ]

        aug_snimke = [
            os.path.join(putanja_studenta, f)
            for f in sorted(os.listdir(putanja_studenta))
            if f.lower().endswith(".wav")
            and f.startswith("aug_")
        ]

        if not snimke:
            print(f"  UPOZORENJE: Nema snimki za '{student}', preskačem.")
            continue

        # Val split samo od originalnih snimki
        random.shuffle(snimke)
        n_val   = max(1, int(len(snimke) * VAL_SPLIT))
        val_sn  = snimke[:n_val]
        train_sn = snimke[n_val:] + aug_snimke   # Train = originalne + augmentacije

        idx = student2idx[student]
        train_snimke.extend([(p, idx) for p in train_sn])
        val_snimke.extend([(p, idx)   for p in val_sn])

        print(f"  {student}: {len(train_sn)} train, {len(val_sn)} val")

    return train_snimke, val_snimke, student2idx, idx2student


# ================================================================
# COLLATE FUNKCIJA
# ================================================================
def collate_fn(batch):
    """Batch s padding-om na jednaku duljinu."""
    signali, labele = zip(*batch)
    # Svi su iste duljine zbog fiksnog cropa u datasetu, ali budimo sigurni
    max_len = max(s.shape[0] for s in signali)
    padded  = torch.stack([
        F.pad(s, (0, max_len - s.shape[0])) for s in signali
    ])
    return padded, torch.tensor(labele, dtype=torch.long)


# ================================================================
# TRENING JEDNA EPOHA
# ================================================================
def treniraj_epohu(model, loader, optimizer, criterion, device) -> tuple:
    model.train()
    ukupno_loss = 0.0
    tocno       = 0
    ukupno      = 0

    for signali, labele in loader:
        signali = signali.to(device)
        labele  = labele.to(device)

        optimizer.zero_grad()
        logiti = model(signali)
        loss   = criterion(logiti, labele)
        loss.backward()

        # Gradient clipping — sprječava exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        ukupno_loss += loss.item() * len(labele)
        tocno       += (logiti.argmax(dim=1) == labele).sum().item()
        ukupno      += len(labele)

    return ukupno_loss / ukupno, tocno / ukupno


# ================================================================
# VALIDACIJA
# ================================================================
@torch.no_grad()
def validiraj(model, loader, criterion, device) -> tuple:
    model.eval()
    ukupno_loss = 0.0
    tocno       = 0
    ukupno      = 0

    for signali, labele in loader:
        signali = signali.to(device)
        labele  = labele.to(device)

        logiti = model(signali)
        loss   = criterion(logiti, labele)

        ukupno_loss += loss.item() * len(labele)
        tocno       += (logiti.argmax(dim=1) == labele).sum().item()
        ukupno      += len(labele)

    return ukupno_loss / ukupno, tocno / ukupno


# ================================================================
# TRENING PETLJA
# ================================================================
def treniraj(model, train_loader, val_loader, optimizer, scheduler,
             n_epoha: int, patience: int, device,
             naziv_faze: str) -> float:
    """
    Glavna trening petlja s early stoppingom.
    Vraća najbolji val accuracy.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc  = 0.0
    best_state    = None
    epohe_bez_poboljšanja = 0

    print(f"\n  {'Epoha':>6}  {'Train Loss':>11}  {'Train Acc':>10}"
          f"  {'Val Loss':>9}  {'Val Acc':>8}")
    print(f"  {'-'*55}")

    for epoha in range(1, n_epoha + 1):
        train_loss, train_acc = treniraj_epohu(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = validiraj(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step(val_loss)

        poboljsanje = val_acc > best_val_acc + 1e-4
        if poboljsanje:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            epohe_bez_poboljšanja = 0
            oznaka = " ★"
        else:
            epohe_bez_poboljšanja += 1
            oznaka = ""

        print(f"  {epoha:>6}  {train_loss:>11.4f}  {train_acc:>9.1%}"
              f"  {val_loss:>9.4f}  {val_acc:>7.1%}{oznaka}")

        if epohe_bez_poboljšanja >= patience:
            print(f"\n  Early stopping — {patience} epoha bez poboljšanja.")
            break

    # Vrati najbolji state
    if best_state:
        model.load_state_dict(best_state)
        print(f"\n  Vraćeni weighti najboljeg modela (val acc: {best_val_acc:.1%})")

    return best_val_acc


# ================================================================
# SPREMI MODEL
# ================================================================
def spremi_model(model, feature_extractor, izlaz_dir: str):
    """
    Sprema fine-tuned backbone (bez projection heada) u format
    kompatibilan s originalnim model.py učitavanjem.
    """
    os.makedirs(izlaz_dir, exist_ok=True)

    # Spremi samo backbone — isti format kao pretrained model
    model.backbone.save_pretrained(izlaz_dir)
    feature_extractor.save_pretrained(os.path.join(izlaz_dir, "feature_extractor"))

    print(f"\n  Model spremljen u: {izlaz_dir}/")
    print(f"  Za korištenje u sustavu, postavi u model.py:")
    print(f'      feature_extractor = AutoFeatureExtractor.from_pretrained("{izlaz_dir}/feature_extractor")')
    print(f'      model = AutoModel.from_pretrained("{izlaz_dir}")')


# ================================================================
# FAZA 1 — LINEAR PROBING
# ================================================================
def pokreni_fazu_1(model, train_loader, val_loader, device) -> float:
    print("\n" + "="*55)
    print("  FAZA 1 — Linear Probing")
    print("  (backbone zamrznut, trenira se samo projection head)")
    print("="*55)

    model.zamrzni_backbone()

    # Samo head parametri
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=F1_LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    best_acc = treniraj(
        model, train_loader, val_loader,
        optimizer, scheduler,
        n_epoha=F1_EPOHE, patience=F1_PATIENCE,
        device=device, naziv_faze="Faza 1"
    )

    print(f"\n  Faza 1 završena. Najbolji val accuracy: {best_acc:.1%}")
    return best_acc


# ================================================================
# FAZA 2 — PARTIAL UNFREEZE
# ================================================================
def pokreni_fazu_2(model, train_loader, val_loader, device) -> float:
    print("\n" + "="*55)
    print(f"  FAZA 2 — Partial Unfreeze (zadnjih {F2_ODMRZNI_BLOKOVA} blokova)")
    print("="*55)

    model.odmrzni_zadnje_blokove(F2_ODMRZNI_BLOKOVA)

    # Različite LR za backbone i head
    backbone_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "head" not in n
    ]
    head_params = list(model.head.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": F2_LR},
        {"params": head_params,     "lr": F2_LR * 10},  # Head može ići brže
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )

    best_acc = treniraj(
        model, train_loader, val_loader,
        optimizer, scheduler,
        n_epoha=F2_EPOHE, patience=F2_PATIENCE,
        device=device, naziv_faze="Faza 2"
    )

    print(f"\n  Faza 2 završena. Najbolji val accuracy: {best_acc:.1%}")
    return best_acc


# ================================================================
# GLAVNI PROGRAM
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Fine-tuning UniSpeech-SAT")
    parser.add_argument("--faza", type=int, choices=[1, 2],
                        help="Pokreni samo fazu 1 ili 2")
    parser.add_argument("--oba", action="store_true",
                        help="Pokreni fazu 1 pa fazu 2")
    args = parser.parse_args()

    # Interaktivni mod ako nema argumenata
    if not args.faza and not args.oba:
        print("="*55)
        print("  Fine-tuning UniSpeech-SAT")
        print("="*55)
        print()
        print("  Odaberi mod:")
        print("  [1] Samo Faza 1 — Linear Probing (sigurno, brzo)")
        print("  [2] Samo Faza 2 — Partial Unfreeze (treba fazu 1)")
        print("  [3] Faza 1 + Faza 2 (preporučeno)")
        odabir = input("  Odabir (1/2/3, Enter = 3): ").strip() or "3"
        if odabir == "1":
            args.faza = 1
        elif odabir == "2":
            args.faza = 2
        else:
            args.oba = True

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cpu":
        print("  NAPOMENA: CPU trening je spor. Preporuča se GPU.")

    # Učitaj podatke
    print(f"\n  Učitavanje podataka iz '{cfg.DIR_BAZA}'...")
    train_snimke, val_snimke, student2idx, idx2student = ucitaj_podatke(cfg.DIR_BAZA)

    n_studenata = len(student2idx)
    print(f"\n  Studenata: {n_studenata}")
    print(f"  Train uzoraka: {len(train_snimke)}")
    print(f"  Val uzoraka:   {len(val_snimke)}")

    if len(val_snimke) == 0:
        print("  GREŠKA: Nema validacijskih uzoraka.")
        return

    # DataLoaderi
    train_ds = GlasovniDataset(train_snimke, augmentacija=True)
    val_ds   = GlasovniDataset(val_snimke,   augmentacija=False)

    train_loader = DataLoader(
        train_ds, batch_size=F1_BATCH, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=F1_BATCH, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    # Inicijalizacija modela
    print(f"\n  Inicijalizacija modela...")
    ft_model = UniSpeechSATSpeaker(n_studenata=n_studenata).to(device)
    ukupno_params = sum(p.numel() for p in ft_model.parameters())
    print(f"  Ukupno parametara: {ukupno_params:,}")

    # Feature extractor (za spremanje)
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAZIV)

    # Pokretanje faza
    if args.faza == 1 or args.oba:
        val_loader_f1 = DataLoader(
            val_ds, batch_size=F1_BATCH, shuffle=False,
            collate_fn=collate_fn, num_workers=0
        )
        pokreni_fazu_1(ft_model, train_loader, val_loader_f1, device)

    if args.faza == 2 or args.oba:
        train_loader_f2 = DataLoader(
            train_ds, batch_size=F2_BATCH, shuffle=True,
            collate_fn=collate_fn, num_workers=0
        )
        val_loader_f2 = DataLoader(
            val_ds, batch_size=F2_BATCH, shuffle=False,
            collate_fn=collate_fn, num_workers=0
        )
        pokreni_fazu_2(ft_model, train_loader_f2, val_loader_f2, device)

    # Spremi model
    print("\n" + "="*55)
    spremi_model(ft_model, feature_extractor, IZLAZ_DIR)

    # Upute za korištenje
    print("\n  Za testiranje fine-tuned modela:")
    print("  1. Obriši baza_cache.pkl")
    print("  2. U model.py promijeni putanje na finetuned_model/")
    print("  3. Pokreni main.py i usporedi rezultate")
    print("\n  Za povratak na originalni model:")
    print("  1. Vrati originalne putanje u model.py")
    print("  2. Obriši baza_cache.pkl")
    print("="*55)


if __name__ == "__main__":
    main()