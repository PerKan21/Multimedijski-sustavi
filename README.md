# Multimedijski sustavi
Konstrukcijske vježbe (ak. god. 2025. / 2026.)

---

## Opis zadatka

**Sustav za popisivanje studenata zasnovan na obradi audio snimke.**

U početnom dijelu zadatka potrebno je analizirati najmanje dva postojeća algoritma za prepoznavanje osoba na temelju audio zapisa govora. Nakon toga potrebno je osmisliti i napraviti računalni algoritam za popisivanje studenata prisutnih na nastavi, na temelju obrade audio snimke postavljene na ulazu u predavaonicu, gdje svaki student koji ulazi u predavaonicu izgovara ime kolegija na čiju je nastavu došao.

Na ulasku u predavaonicu potrebno je snimati audio zapis svakoga tko ulazi, a zvučni zapis ne treba biti trajanja duljeg od jedne minute. Potrebno je snimiti 10 različitih audio signala ulaska u predavaonicu različitim redoslijedom. Ukupno na nastavi treba biti prisutno 10 studenata koji se popisuju, s tim što neće biti svi prisutni u svim snimkama. Na snimkama se ponekad trebaju čuti i osobe koje nisu među 10 onih koje je potrebno popisati.

Usporedbom zvukova iz zapisa s ulaza u predavaonicu sa zvukovima studenata iz baze studenata, algoritam upisuje `+` za prisustvo onom studentu kojeg prepozna. Treba analizirati osnovne značajke na temelju kojih se prepoznaje određena osoba prema govoru te analizirati uspješnost predloženog algoritma. Algoritam može:
- ✅ uspješno prepoznati osobu,
- ❌ pogrešno prepoznati osobu,
- ❓ ne prepoznati osobu uopće (javiti da osoba nije u bazi studenata).

---

## Preduvjeti

- **Python 3.10 ili više** → [python.org/downloads](https://www.python.org/downloads/)
  > ⚠️ Tijekom instalacije označi **"Add Python to PATH"** prije nego klikneš *Install Now*.
  > 🔄 Nakon instalacije Pythona **preporučuje se restartati računalo** prije nego nastaviš.
- **PyCharm Community Edition** → [jetbrains.com/pycharm/download](https://www.jetbrains.com/pycharm/download/)

---

## Postavljanje

### 1. Stvori novi projekt u PyCharmu

1. Pokreni PyCharm → **File → New Project**
2. Odaberi lokaciju ili ostavi sve na defaultu
3. Klikni **Create**

### 2. Dodaj datoteke projekta

1. U lijevom panelu desni klik na naziv projekta → **Open In → Explorer**
2. Otvori mapu svog projekta i u nju zalijepi sve 3 datoteke preuzete s ovog repozitorija: `main.py`, `augmentacija.py` i `requirements.txt`
   > Ako te pita za zamjenu postojećih datoteka — klikni **Da**

Nakon toga bi u lijevom panelu PyCharma trebao vidjeti sljedeću strukturu:

```
📁 moj_projekt/
├── 📄 main.py
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

- **`baza/`** — za svakog studenta stvori podfolder s njegovim imenom i u njega stavi njegove audio snimke
- **`snimke/`** — ovdje stavi snimke s ulaza u predavaonicu koje algoritam treba analizirati

Konačna struktura trebala bi izgledati ovako:

```
📁 moj_projekt/
├── 📄 main.py
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
└── 📁 snimke/
    ├── 🔊 ulaz1.wav
    ├── 🔊 ulaz2.wav
    └── 🔊 ulazN.wav
```

---

## Pokretanje koda

Klikni zelenu strelicu ▶️ u gornjem desnom kutu PyCharma, ili desni klik na `main.py` → **Run 'main'**.

---

<div align="right">
<sub>Zadnja izmjena: 28.03.2026. u 20:23h</sub>
</div>
