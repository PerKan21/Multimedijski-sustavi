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
- **ffmpeg** → potreban za konverziju audio formata (m4a, mp3, mp4, ogg...)
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
├── 📄 stitch.py
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

Nakon postavljanja, u mapi projekta potrebno je ručno stvoriti potrebne foldere:

- **`baza/`** — za svakog studenta stvori podfolder s njegovim imenom i u njega stavi njegove audio snimke (3 ili više)
- **`snimke/`** — ovdje stavi snimke s ulaza u predavaonicu koje algoritam treba analizirati
- **`stitch/snimke/`** — individualne snimke govornika za generiranje testnih snimki
- **`stitch/generirano/`** — generirane spojene snimke (kreira se automatski)
- **`stitch/log/`** — tekstualni logovi redoslijeda za svaku generiranu snimku (kreira se automatski)

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
├── 📄 stitch.py
├── 📄 requirements.txt
├── 📁 baza/
│   ├── 📁 Ime Studenta 1/
│   │   ├── 🔊 snimka1.wav
│   │   └── 🔊 snimka2.wav
│   └── 📁 Ime Studenta N/
│       └── 🔊 snimka1.wav
├── 📁 snimke/
│   ├── 🔊 ulaz1.wav
│   └── 🔊 ulazN.wav
├── 📁 rezultati/
│   ├── 📄 prisutnost_2026-04-09_14-35-22.txt
│   └── 📊 prisutnost_2026-04-09_14-35-22.xlsx
└── 📁 stitch/
    ├── 📁 snimke/       ← individualne snimke govornika
    ├── 📁 generirano/   ← generirane spojene snimke
    └── 📁 log/          ← tekstualni logovi redoslijeda
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
- Izgradnju baze govornika s progress barom i postotkom napretka
- Učitavanje postojećeg cachea (brže pokretanje bez ponovne izgradnje)
- Pokretanje analize s prikazom napretka po snimci
- Prikaz popisa prisutnosti s dinamičkim bojama prema broju prisutnosti kroz sve snimke
- Spremanje rezultata kao `.txt` ili `.xlsx` (Excel tablica s matricom prisutnosti)

### Terminalni mod

Desni klik na `main.py` u lijevom panelu PyCharma → **Run 'main'**, ili zelena strelica ▶️ ako je `main.py` odabrana datoteka.

Alternativno u terminalu:
```bash
python main.py
```

Na početku programa pita se za brisanje cachea (ako postoji), a na kraju analize program pita za format spremanja:
```
  [0] Nemoj spremati, samo izađi
  [1] Tekstualna datoteka (.txt)
  [2] Excel tablica (.xlsx)
```

---

## Augmentacija podataka

Skripta `augmentacija.py` proširuje referentne snimke u bazi primjenom 10 audio transformacija kako bi se povećao broj uzoraka po studentu i poboljšala robustnost modela. Augmentacije koje se primjenjuju su time stretch (±5%), Gaussov šum, glasnoća, reverb, telefonski filtar, MP3 kompresija i slučajni crop.

Svako pokretanje dodaje novi sloj augmentacije:

- **1. pokretanje** → `aug_*.wav`
- **2. pokretanje** → `2aug_*.wav`
- **3. pokretanje** → `3aug_*.wav`

Desni klik na `augmentacija.py` → **Run 'augmentacija'**, ili u terminalu:
```bash
python augmentacija.py
```

> ⚠️ Nakon augmentacije obriši `baza_cache.pkl` da se baza ažurira s novim snimkama.

---

## Generiranje testnih snimki (stitch)

Skripta `stitch.py` spaja individualne snimke govornika iz `stitch/snimke/` u jednu snimku s realističnim razmacima između govornika, simulirajući stvarno snimanje s ulaza u predavaonicu.

Desni klik na `stitch.py` → **Run 'stitch'**, ili u terminalu:
```bash
python stitch.py
```

Program interaktivno pita:
- **Koliko snimki generirati** — svaka snimka ima drugačiji nasumični redoslijed govornika
- **Koliko studenata izostaviti** — `0` (nitko), broj (fiksno), ili `r` (nasumično, distribuirano oko 3)
- **Generirati log datoteku** — `.txt` datoteka s redoslijedom i izostavljenima za svaku snimku

Razmaci između govornika variraju realistično: 30% studenata dođe brzo (1–1.8s), 50% normalno (oko 2.5s), 20% malo kasni (3.5–5s).

Generirane snimke idu u `stitch/generirano/`, a logovi u `stitch/log/`.

---

## Preprocessing pipeline

Isti preprocessing koristi se i za izgradnju baze i za analizu ulaznih snimki, što osigurava konzistentnost embeddinga:

1. **Resample** na 16 kHz, mono
2. **Opcionalni denoising** — isključen po defaultu (`prop_decrease=0.0`), jer VAD je prioritet
3. **VAD segmentacija** — uklanja tišinu, spaja bliske segmente, filtrira prekratke
4. **Normalizacija po segmentu** — svaki VAD segment normalizira se zasebno
5. **ECAPA-TDNN** — ekstrakcija 768-dimenzionalnog glasovnog embeddinga

---

## Konfiguracija

Sve konfiguracijske varijable nalaze se u `main.py` pod sekcijom **POSTAVKE**:

| Varijabla | Opis | Default |
|---|---|---|
| `SEGMENT_TRAJANJE` | Duljina segmenta za ekstrakciju embeddinga (s) | `1.5` |
| `SEGMENT_PREKLAPANJE` | Preklapanje između segmenata (s) | `0.3` |
| `VAD_TOP_DB` | Prag energije ispod kojeg se smatra tišinom (dB) | `25` |
| `VAD_MIN_DULJINA` | Minimalna duljina govornog segmenta (s) | `0.3` |
| `VAD_SPAJANJE` | Spoji segmente bliže od ovoga (s) | `0.15` |
| `SUM_PROP_DECREASE` | Agresivnost redukcije šuma (0 = isključeno) | `0.75` |
| `FAKTOR_GORNJEG_PRAGA` | Gornji prag = donji × faktor | `2.0` |
| `FIKSNI_PRAG_DONJI` | Fiksni donji prag (0.0–1.0) ili `None` za dinamički | `None` |
| `FIKSNI_PRAG_GORNJI` | Fiksni gornji prag (0.0–1.0) ili `None` za dinamički | `None` |

### Pragovi prepoznavanja

Sustav koristi dva praga za klasifikaciju segmenata:

- `dist ≤ prag_donji` → **[+] SIGURAN** — prepoznata osoba
- `prag_donji < dist ≤ prag_gornji` → **[?] NESIGURAN** — vjerojatno prepoznata osoba
- `dist > prag_gornji` → **[!] ULJEZ** — nepoznata osoba

Ako `FIKSNI_PRAG_DONJI` nije postavljen (`None`), pragovi se automatski izračunavaju iz međusobnih distanci embeddinga u bazi. Gornji prag se uvijek računa kao `donji × FAKTOR_GORNJEG_PRAGA`, osim ako je i gornji eksplicitno postavljen.

---

<div align="right">
<sub>Zadnja izmjena: 11.04.2026. u 14:30h</sub>
</div>
