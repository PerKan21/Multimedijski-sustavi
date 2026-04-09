# Multimedijski sustavi
Konstrukcijske vježbe (ak. god. 2025. / 2026.)

---

## Opis zadatka

**Sustav za popisivanje studenata zasnovan na obradi audio snimke.**

U početnom dijelu zadatka potrebno je analizirati najmanje dva postojeća algoritma za prepoznavanje osoba na temelju audio zapisa govora. Nakon toga potrebno je osmisliti i napraviti računalni algoritam za popisivanje studenata prisutnih na nastavi, na temelju obrade audio snimke postavljene na ulazu u predavaonicu, gdje svaki student koji ulazi u predavaonicu izgovara ime kolegija na čiju je nastavu došao.

Na ulasku u predavaonicu potrebno je snimati audio zapis svakoga tko ulazi, a zvučni zapis ne treba biti trajanja duljeg od jedne minute. Potrebno je snimiti 10 različitih audio signala ulaska u predavaonicu različitim redoslijedom. Ukupno na nastavi treba biti prisutno 10 studenata koji se popisuju, s tim što neće biti svi prisutni u svim snimkama. Na snimkama se ponekad trebaju čuti i osobe koje nisu među 10 onih koje je potrebno popisati.

Usporedbom zvukova iz zapisa s ulaza u predavaonicu sa zvukovima studenata iz baze studenata, algoritam upisuje `+` za prisustvo onom studentu kojeg prepozna. Algoritam može:

- ✅ uspješno prepoznati osobu,
- ❌ pogrešno prepoznati osobu,
- ❓ ne prepoznati osobu uopće (javiti da osoba nije u bazi studenata).

---

## Preduvjeti

- **Python 3.10 ili više** → [python.org/downloads](https://www.python.org/downloads/)
  > ⚠️ Tijekom instalacije označi **"Add Python to PATH"** prije nego klikneš *Install Now*.
  > 🔄 Nakon instalacije Pythona **preporučuje se restartati računalo** prije nego nastaviš.
- **PyCharm Community Edition** → [jetbrains.com/pycharm/download](https://www.jetbrains.com/pycharm/download/)
- **ffmpeg** → potreban za konverziju audio formata (m4a, mp3...)
  > 💡 Najlakša instalacija — otvori **cmd kao administrator** i pokreni:
  > ```
  > winget install ffmpeg
  > ```
  > Nakon instalacije restartaj PyCharm.
  > 
  > Alternativno: ručna instalacija s [ffmpeg.org/download.html](https://ffmpeg.org/download.html) — raspakiraj i dodaj `ffmpeg/bin` u System PATH.

---

## Postavljanje

### 1. Stvori novi projekt u PyCharmu

1. Pokreni PyCharm → **File → New Project**
2. Odaberi lokaciju ili ostavi sve na defaultu
3. Klikni **Create**

### 2. Dodaj datoteke projekta

1. U lijevom panelu desni klik na naziv projekta → **Open In → Explorer**
2. Otvori mapu svog projekta i u nju zalijepi sve datoteke preuzete s ovog repozitorija

Nakon toga bi u lijevom panelu PyCharma trebao vidjeti sljedeću strukturu:

```
📁 moj_projekt/
├── 📄 main.py
├── 📄 gui.py
├── 📄 model.py
├── 📄 predobrada.py
├── 📄 analiza.py
├── 📄 augmentacija.py
└── 📄 requirements.txt
```

> Ako datoteke nisu vidljive odmah, desni klik na naziv projekta → **Reload from Disk**.

### 3. Instaliraj potrebne pakete

Otvori terminal unutar PyCharma (**View → Tool Windows → Terminal**) i pokreni:

```bash
pip install -r requirements.txt
```

---

## Struktura projekta

Nakon postavljanja, u mapi projekta potrebno je ručno stvoriti još 2 foldera:

- **`baza/`** — za svakog studenta stvori podfolder s njegovim imenom i u njega stavi njegove audio snimke (3 ili više)
- **`snimke/`** — ovdje stavi snimke s ulaza u predavaonicu koje algoritam treba analizirati

Rezultati analize automatski se spremaju u folder **`rezultati/`** koji se kreira automatski.

Konačna struktura trebala bi izgledati ovako:

```
📁 moj_projekt/
├── 📄 main.py
├── 📄 gui.py
├── 📄 model.py
├── 📄 predobrada.py
├── 📄 analiza.py
├── 📄 augmentacija.py
├── 📄 requirements.txt
├── 📁 baza/
│   ├── 📁 Ime Studenta 1/
│   │   ├── 🔊 snimka1.wav
│   │   └── 🔊 snimka2.wav
│   ├── 📁 Ime Studenta 2/
│   │   ├── 🔊 snimka1.wav
│   │   └── 🔊 snimka2.wav
│   └── 📁 Ime Studenta N/
│       ├── 🔊 snimka1.wav
│       └── 🔊 snimka2.wav
├── 📁 snimke/
│   ├── 🔊 ulaz1.wav
│   ├── 🔊 ulaz2.wav
│   └── 🔊 ulazN.wav
└── 📁 rezultati/
    ├── 📄 prisutnost_2026-04-09_14-35-22.txt
    └── 📊 prisutnost_2026-04-09_14-35-22.xlsx
```

---

## Pokretanje

### Grafičko sučelje (GUI)

Desni klik na `gui.py` u lijevom panelu PyCharma → **Run 'gui'**, ili zelena strelica ▶️ ako je `gui.py` odabrana datoteka.

Alternativno u terminalu:
```bash
python gui.py
```

![GUI sučelje](https://i.imgur.com/kVyHUM8.png)

GUI omogućuje:
- Odabir mape baze i ulaznih snimki
- Izgradnju baze govornika s progress barom
- Učitavanje postojećeg cachea (brže pokretanje)
- Pokretanje analize s prikazom napretka
- Prikaz popisa prisutnosti (zeleno = prisutan, crveno = odsutan)
- Spremanje rezultata kao `.txt` ili `.xlsx` (Excel tablica)

### Terminalni mod

Desni klik na `main.py` u lijevom panelu PyCharma → **Run 'main'**, ili zelena strelica ▶️ ako je `main.py` odabrana datoteka.

Alternativno u terminalu:
```bash
python main.py
```

Na kraju analize program pita za format spremanja:
```
  [0] Nemoj spremati, samo izađi
  [1] Tekstualna datoteka (.txt)
  [2] Excel tablica (.xlsx)
```

---

## Augmentacija podataka

Skripta `augmentacija.py` proširuje referentne snimke u bazi primjenom 11 audio transformacija (pitch shift, time stretch, Gaussov šum, glasnoća, reverb). Svako pokretanje dodaje novi sloj augmentacije:

- **1. pokretanje** → `aug_*.wav`
- **2. pokretanje** → `2aug_*.wav`
- **3. pokretanje** → `3aug_*.wav`

```bash
python augmentacija.py
```

> ⚠️ Nakon augmentacije obriši `baza_cache.pkl` da se baza ažurira s novim snimkama.

---

## Konfiguracija

Sve konfiguracijske varijable nalaze se u `main.py` pod sekcijom **POSTAVKE**:

| Varijabla | Opis | Default |
|---|---|---|
| `SEGMENT_TRAJANJE` | Duljina segmenta za ekstrakciju embeddinga (s) | `1.5` |
| `VAD_TOP_DB` | Prag energije ispod kojeg se smatra tišinom (dB) | `25` |
| `VAD_MIN_DULJINA` | Minimalna duljina govornog segmenta (s) | `0.3` |
| `VAD_SPAJANJE` | Spoji segmente bliže od ovoga (s) | `0.15` |
| `FAKTOR_GORNJEG_PRAGA` | Gornji prag = donji × faktor | `1.8` |
| `FIKSNI_PRAG_DONJI` | Fiksni donji prag (0.0–1.0) ili `None` za dinamički | `None` |
| `FIKSNI_PRAG_GORNJI` | Fiksni gornji prag (0.0–1.0) ili `None` za dinamički | `None` |
| `CLUSTERING_PRAG` | Prag za procjenu broja govornika | `0.25` |

---

<div align="right">
<sub>Zadnja izmjena: 09.04.2026. u 20:00h</sub>
</div>
