"""
GUI - Sustav za popisivanje studenata
======================================
Moderno Tkinter sučelje. Pokretanje: python gui.py
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import main as cfg
from model import ucitaj_model, ucitaj_bazu, izracunaj_pragove
from analiza import obradi_snimku, spremi_rezultate, spremi_excel

# ================================================================
# PALETA BOJA
# ================================================================
BG         = "#1e1e1e"
SURFACE    = "#2a2a2a"
SURFACE2   = "#333333"
BORDER     = "#3d3d3d"
TEXT       = "#e8e8e8"
TEXT_MUTE  = "#888888"
SUCCESS    = "#3d7a42"
SUCCESS_FG = "#7ecf85"
WARN_1     = "#7a4a1a"   # Prisutan na malo snimki
WARN_1_FG  = "#e8943a"
WARN_2     = "#6a6a10"   # Prisutan na otprilike pola
WARN_2_FG  = "#d4d44a"
WARN_3     = "#2a6a2a"   # Prisutan na vecini
WARN_3_FG  = "#7ecf85"
DANGER     = "#7a3d3d"
DANGER_FG  = "#cf7e7e"
FONT       = "Cambria"


# ================================================================
# POMOCNI WIDGETI
# ================================================================
def kartica(roditelj, **kw) -> tk.Frame:
    return tk.Frame(roditelj, bg=SURFACE,
                    highlightbackground=BORDER,
                    highlightthickness=1, **kw)


def label(roditelj, tekst="", boja=TEXT, vel=11, bold=False, bg=None, **kw) -> tk.Label:
    tezina = "bold" if bold else "normal"
    return tk.Label(roditelj, text=tekst, fg=boja,
                    bg=bg if bg else SURFACE,
                    font=(FONT, vel, tezina), **kw)


def gumb(roditelj, tekst, naredba, **kw) -> tk.Button:
    return tk.Button(
        roditelj, text=tekst, command=naredba,
        bg=SURFACE2, fg=TEXT,
        activebackground=BORDER, activeforeground=TEXT,
        relief="flat", cursor="hand2",
        font=(FONT, 10),
        padx=14, pady=5,
        **kw
    )


def unos(roditelj, varijabla, **kw) -> tk.Entry:
    return tk.Entry(
        roditelj, textvariable=varijabla,
        bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
        relief="flat", font=(FONT, 10),
        highlightbackground=BORDER, highlightthickness=1,
        **kw
    )


# ================================================================
# PROGRESS BAR KOMPONENTA
# ================================================================
class ProgressBar:
    def __init__(self, roditelj, boja=TEXT):
        self._boja_default = boja
        self.canvas = tk.Canvas(roditelj, height=4, bg=SURFACE2, highlightthickness=0)
        self.rect = self.canvas.create_rectangle(0, 0, 0, 4, fill=boja, outline="")

    def pack(self, **kw):
        self.canvas.pack(**kw)

    def postavi(self, napredak: float, boja=None):
        if boja:
            self.canvas.itemconfig(self.rect, fill=boja)
        self.canvas.update_idletasks()
        sirina = self.canvas.winfo_width()
        self.canvas.coords(self.rect, 0, 0, sirina * max(0.0, min(1.0, napredak)), 4)

    def reset(self):
        self.canvas.itemconfig(self.rect, fill=self._boja_default)
        self.postavi(0.0)

    def zavrseno(self):
        """Postavi na 100% i promijeni boju u zelenu."""
        self.postavi(1.0, boja=SUCCESS_FG)


# ================================================================
# GLAVNA APLIKACIJA
# ================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sustav za popisivanje studenata")
        self.geometry("760x920")
        self.configure(bg=BG)
        self.resizable(False, False)

        self.baza               = {}
        self.prag_d             = 0.0
        self.prag_g             = 0.0
        self.prisutnost         = {}
        self.svi_rezultati      = {}
        self.timestamp_analize  = None
        self._ukupno_studenata  = 0
        self._ucitano_studenata = 0
        self.var_format         = tk.StringVar(value="txt")

        self._izgradnja_ui()
        self._log("Pokretanje — učitavanje modela...")
        threading.Thread(target=self._ucitaj_model, daemon=True).start()

    # ----------------------------------------------------------------
    # IZGRADNJA UI
    # ----------------------------------------------------------------
    def _izgradnja_ui(self):
        zaglavlje = tk.Frame(self, bg=BG)
        zaglavlje.pack(fill="x", padx=20, pady=(20, 0))
        label(zaglavlje, "Sustav za popisivanje studenata",
              vel=17, bold=True, bg=BG).pack(anchor="center")

        sep = tk.Frame(self, bg=BORDER, height=1)
        sep.pack(fill="x", padx=20, pady=(10, 16))

        self._sekcija_baza()
        self._sekcija_snimke()
        self._sekcija_pokretanje()
        self._sekcija_rezultati()
        self._sekcija_log()

    def _sekcija_baza(self):
        k = tk.Frame(self, bg=SURFACE,
                     highlightbackground=BORDER, highlightthickness=1)
        k.pack(fill="x", padx=20, pady=(0, 10))

        unutar = tk.Frame(k, bg=SURFACE)
        unutar.pack(fill="x", padx=14, pady=12)

        label(unutar, "Baza govornika", bold=True, vel=11).pack(anchor="w", pady=(0, 8))

        red = tk.Frame(unutar, bg=SURFACE)
        red.pack(fill="x")
        self.var_baza = tk.StringVar(value=cfg.DIR_BAZA)
        unos(red, self.var_baza).pack(side="left", fill="x", expand=True, padx=(0, 8))
        gumb(red, "Odaberi", self._odaberi_bazu).pack(side="left", padx=(0, 4))
        gumb(red, "Izgradi", self._izgradi_bazu).pack(side="left", padx=(0, 4))
        # TODO: maknuti kad ne bude potrebno
        gumb(red, "Učitaj cache", self._ucitaj_cache).pack(side="left")

        self.lbl_studenti = label(unutar, "Studenti: —", boja=TEXT_MUTE, vel=9)
        self.lbl_studenti.pack(anchor="w", pady=(6, 2))

        self.lbl_pragovi = label(unutar, "Pragovi: —", boja=TEXT_MUTE, vel=9)
        self.lbl_pragovi.pack(anchor="w", pady=(0, 6))

        self.prog_baza = ProgressBar(unutar)
        self.prog_baza.pack(fill="x", pady=(0, 4))
        self.prog_baza.canvas.pack_forget()

        self.lbl_baza_status = label(unutar, "", boja=TEXT_MUTE, vel=11)
        self.lbl_baza_status.pack(anchor="w")

    def _sekcija_snimke(self):
        k = tk.Frame(self, bg=SURFACE,
                     highlightbackground=BORDER, highlightthickness=1)
        k.pack(fill="x", padx=20, pady=(0, 10))

        unutar = tk.Frame(k, bg=SURFACE)
        unutar.pack(fill="x", padx=14, pady=12)

        label(unutar, "Ulazne snimke", bold=True, vel=11).pack(anchor="w", pady=(0, 8))

        red = tk.Frame(unutar, bg=SURFACE)
        red.pack(fill="x")
        self.var_snimke = tk.StringVar(value=cfg.DIR_SNIMKE)
        unos(red, self.var_snimke).pack(side="left", fill="x", expand=True, padx=(0, 8))
        gumb(red, "Odaberi", self._odaberi_snimke).pack(side="left")

        self.lbl_snimke = label(unutar, "Snimki: —", boja=TEXT_MUTE, vel=9)
        self.lbl_snimke.pack(anchor="w", pady=(6, 0))

    def _sekcija_pokretanje(self):
        k = tk.Frame(self, bg=SURFACE,
                     highlightbackground=BORDER, highlightthickness=1)
        k.pack(fill="x", padx=20, pady=(0, 10))

        unutar = tk.Frame(k, bg=SURFACE)
        unutar.pack(fill="x", padx=14, pady=12)

        self.btn_run = gumb(unutar, "▶   Pokreni analizu", self._pokreni, state="disabled")
        self.btn_run.config(font=(FONT, 12, "bold"), pady=12)
        self.btn_run.pack(fill="x", pady=(0, 10))

        self.prog_analiza = ProgressBar(unutar)
        self.prog_analiza.pack(fill="x", pady=(0, 6))

        self.lbl_status = label(unutar, "—", boja=TEXT_MUTE, vel=11)
        self.lbl_status.pack(anchor="w")

    def _sekcija_rezultati(self):
        k = tk.Frame(self, bg=SURFACE,
                     highlightbackground=BORDER, highlightthickness=1)
        k.pack(fill="x", padx=20, pady=(0, 10))

        unutar = tk.Frame(k, bg=SURFACE)
        unutar.pack(fill="x", padx=14, pady=12)

        label(unutar, "Popis prisutnosti", bold=True, vel=11).pack(anchor="w", pady=(0, 8))

        self.frame_rez = tk.Frame(unutar, bg=SURFACE)
        self.frame_rez.pack(fill="x")

        # Format odabir
        dno = tk.Frame(unutar, bg=SURFACE)
        dno.pack(fill="x", pady=(10, 0))

        label(dno, "Spremi kao:", vel=10, boja=TEXT_MUTE).pack(side="left", padx=(0, 8))

        for tekst, vrijednost in [("Tekstualna datoteka (.txt)", "txt"),
                                   ("Excel tablica (.xlsx)", "xlsx")]:
            tk.Radiobutton(
                dno, text=tekst, variable=self.var_format, value=vrijednost,
                bg=SURFACE, fg=TEXT, selectcolor=SURFACE2,
                activebackground=SURFACE, activeforeground=TEXT,
                font=(FONT, 10), cursor="hand2"
            ).pack(side="left", padx=(0, 10))

        # Gumb iste veličine kao Pokreni analizu — pady=12, bez height
        self.btn_spremi = gumb(unutar, "💾   Spremi rezultate", self._spremi)
        self.btn_spremi.config(font=(FONT, 12, "bold"), pady=12)
        self.btn_spremi.pack(fill="x", pady=(10, 0))

    def _sekcija_log(self):
        k = tk.Frame(self, bg=SURFACE,
                     highlightbackground=BORDER, highlightthickness=1)
        k.pack(fill="x", padx=20, pady=(0, 16))

        self.txt_log = tk.Text(
            k, height=4, bg=SURFACE2, fg=TEXT_MUTE,
            font=("Consolas", 10), relief="flat",
            padx=10, pady=8, state="disabled"
        )
        self.txt_log.pack(fill="x")

    # ----------------------------------------------------------------
    # LOG I STATUS
    # ----------------------------------------------------------------
    def _log(self, poruka: str):
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", poruka + "\n")
        self.txt_log.configure(state="disabled")
        self.txt_log.see("end")
        self.txt_log.update_idletasks()

    def _status(self, poruka: str, napredak: float = None, success: bool = False):
        boja = SUCCESS_FG if success else TEXT_MUTE
        self.lbl_status.configure(text=poruka, fg=boja)
        if napredak is not None:
            if success:
                self.prog_analiza.zavrseno()
            else:
                self.prog_analiza.postavi(napredak)
        self.update_idletasks()

    # ----------------------------------------------------------------
    # MODEL
    # ----------------------------------------------------------------
    def _ucitaj_model(self):
        try:
            ucitaj_model(callback=lambda msg: self.after(0, self._log, msg))
            self.after(0, self._model_spreman)
        except Exception as e:
            self.after(0, messagebox.showerror, "Greška", f"Model nije učitan:\n{e}")

    def _model_spreman(self):
        self.btn_run.configure(state="normal")
        self._log("Model spreman.")

    # ----------------------------------------------------------------
    # ODABIR MAPA
    # ----------------------------------------------------------------
    def _odaberi_bazu(self):
        mapa = filedialog.askdirectory(title="Odaberi mapu baze")
        if mapa:
            self.var_baza.set(mapa)

    def _odaberi_snimke(self):
        mapa = filedialog.askdirectory(title="Odaberi mapu snimki")
        if mapa:
            self.var_snimke.set(mapa)
            snimke = [f for f in os.listdir(mapa)
                      if f.lower().endswith(cfg.PODRZANI_FORMATI)]
            self.lbl_snimke.configure(text=f"Snimki pronađeno: {len(snimke)}")

    # ----------------------------------------------------------------
    # CACHE I IZGRADNJA BAZE
    # ----------------------------------------------------------------
    def _ucitaj_cache(self):
        cache = os.path.join(os.getcwd(), cfg.CACHE_PUTANJA)
        if not os.path.exists(cache):
            messagebox.showwarning("Upozorenje",
                                   f"Cache ne postoji:\n{cfg.CACHE_PUTANJA}\n\nPokreni 'Izgradi' prvo.")
            return

        self.btn_run.configure(state="disabled")
        self.lbl_baza_status.configure(text="Učitavanje cachea...", fg=TEXT_MUTE)
        self._log("Učitavanje cachea...")

        def _rad():
            from model import ucitaj_cache as _ucitaj_cache_fn
            baza = _ucitaj_cache_fn(cfg.CACHE_PUTANJA)
            if baza is None:
                self.after(0, messagebox.showerror, "Greška", "Cache nije valjan.")
                self.after(0, self.btn_run.configure, {"state": "normal"})
                return
            prag_d, prag_g = izracunaj_pragove(baza, cfg.FAKTOR_GORNJEG_PRAGA)
            self.after(0, self._baza_gotova, baza, prag_d, prag_g, "✓  Učitana baza iz cachea")

        threading.Thread(target=_rad, daemon=True).start()

    def _izgradi_bazu(self):
        dir_baza = self.var_baza.get()
        if not os.path.isdir(dir_baza):
            messagebox.showerror("Greška", f"Mapa ne postoji:\n{dir_baza}")
            return

        studenti = [s for s in os.listdir(dir_baza)
                    if os.path.isdir(os.path.join(dir_baza, s))]
        self._ukupno_studenata  = max(len(studenti), 1)
        self._ucitano_studenata = 0

        cache = os.path.join(os.getcwd(), cfg.CACHE_PUTANJA)
        if os.path.exists(cache):
            os.remove(cache)

        self.btn_run.configure(state="disabled")
        self.prog_baza.canvas.pack(fill="x", pady=(0, 4))
        self.prog_baza.reset()
        self.lbl_baza_status.configure(text="Izgradnja baze...", fg=TEXT_MUTE)
        self._log("Izgradnja baze...")

        def _rad():
            def _cb(ime, n):
                if n is None:
                    self.after(0, self._log, ime)
                else:
                    self._ucitano_studenata += 1
                    napredak = self._ucitano_studenata / self._ukupno_studenata
                    postotak = int(napredak * 100)
                    self.after(0, self.prog_baza.postavi, napredak)
                    self.after(0, self.lbl_baza_status.configure,
                               {"text": f"Izgradnja baze...  {self._ucitano_studenata}/{self._ukupno_studenata}  ({postotak}%)",
                                "fg": TEXT_MUTE})
                    self.after(0, self._log, f"  {ime} — {n} snimki")

            baza = ucitaj_bazu(
                dir_baza, callback=_cb,
                sr=cfg.SR,
                trajanje=cfg.SEGMENT_TRAJANJE,
                preklapanje=cfg.SEGMENT_PREKLAPANJE,
                prop_decrease=cfg.SUM_PROP_DECREASE,
                vad_top_db=cfg.VAD_TOP_DB,
                vad_min_duljina=cfg.VAD_MIN_DULJINA,
                vad_spajanje=cfg.VAD_SPAJANJE,
                podrzani_formati=cfg.PODRZANI_FORMATI,
                cache_putanja=cfg.CACHE_PUTANJA
            )
            prag_d, prag_g = izracunaj_pragove(baza, cfg.FAKTOR_GORNJEG_PRAGA)
            self.after(0, self._baza_gotova, baza, prag_d, prag_g)

        threading.Thread(target=_rad, daemon=True).start()

    def _baza_gotova(self, baza, prag_d, prag_g, poruka="✓  Baza izgrađena"):
        self.baza   = baza

        # Primijeni fiksne pragove ako su postavljeni
        if isinstance(cfg.FIKSNI_PRAG_DONJI, float) and 0.0 < cfg.FIKSNI_PRAG_DONJI < 1.0:
            prag_d = cfg.FIKSNI_PRAG_DONJI
            prag_g = prag_d * cfg.FAKTOR_GORNJEG_PRAGA  # Preračunaj iz fiksnog donjeg
        if isinstance(cfg.FIKSNI_PRAG_GORNJI, float) and 0.0 < cfg.FIKSNI_PRAG_GORNJI < 1.0:
            prag_g = cfg.FIKSNI_PRAG_GORNJI

        self.prag_d = prag_d
        self.prag_g = prag_g
        self.prog_baza.zavrseno()
        self.lbl_baza_status.configure(text=poruka, fg=SUCCESS_FG)
        self.lbl_studenti.configure(
            text=f"Studenti ({len(baza)}): {', '.join(sorted(baza.keys()))}"
        )
        self.lbl_pragovi.configure(
            text=f"Donji prag: {prag_d:.4f}   Gornji prag: {prag_g:.4f}"
        )
        self.btn_run.configure(state="normal")
        self._log("Baza izgrađena.")

    # ----------------------------------------------------------------
    # POKRETANJE ANALIZE
    # ----------------------------------------------------------------
    def _pokreni(self):
        if not self.baza:
            messagebox.showwarning("Upozorenje", "Prvo izgradi bazu!")
            return

        dir_snimke = self.var_snimke.get()
        if not os.path.isdir(dir_snimke):
            messagebox.showerror("Greška", f"Mapa ne postoji:\n{dir_snimke}")
            return

        snimke = sorted([
            f for f in os.listdir(dir_snimke)
            if f.lower().endswith(cfg.PODRZANI_FORMATI)
            and not f.endswith("_konv.wav")
        ])
        if not snimke:
            messagebox.showwarning("Upozorenje", "Nema snimki u odabranoj mapi!")
            return

        self.btn_run.configure(state="disabled")
        self._ocisti_rezultate()
        self._status("Pokretanje...", 0.0)
        self._log(f"Analiza {len(snimke)} snimki...")

        def _rad():
            prisutnost    = {ime: False for ime in self.baza}
            svi_rezultati = {}
            ukupno = len(snimke)

            for i, naziv in enumerate(snimke):
                putanja = os.path.join(dir_snimke, naziv)
                prepoznati, segmenti, uljezi, n_gov = obradi_snimku(
                    putanja, self.baza, self.prag_d, self.prag_g,
                    sr=cfg.SR,
                    trajanje=cfg.SEGMENT_TRAJANJE,
                    preklapanje=cfg.SEGMENT_PREKLAPANJE,
                    prop_decrease=cfg.SUM_PROP_DECREASE,
                    vad_top_db=cfg.VAD_TOP_DB,
                    vad_min_duljina=cfg.VAD_MIN_DULJINA,
                    vad_spajanje=cfg.VAD_SPAJANJE,
                    clustering_prag=cfg.CLUSTERING_PRAG
                )
                for student in prepoznati:
                    prisutnost[student] = True
                svi_rezultati[naziv] = (prepoznati, segmenti, uljezi, n_gov)
                napredak = (i + 1) / ukupno
                self.after(0, self._status,
                           f"{naziv}  ({i+1}/{ukupno})", napredak, False)

            self.after(0, self._analiza_gotova, prisutnost, svi_rezultati)

        threading.Thread(target=_rad, daemon=True).start()

    def _analiza_gotova(self, prisutnost, svi_rezultati):
        from datetime import datetime
        self.prisutnost        = prisutnost
        self.svi_rezultati     = svi_rezultati
        self.timestamp_analize = datetime.now().strftime("%d.%m.%Y. u %H:%M:%S")
        self._prikazi_rezultate(prisutnost)
        self._status("✓  Analiza završena.", 1.0, success=True)
        self.btn_run.configure(state="normal")
        self._log("Analiza završena.")

    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # PRIKAZ REZULTATA - grid, abecedno, lijeva na desno
    # ----------------------------------------------------------------
    def _ocisti_rezultate(self):
        for w in self.frame_rez.winfo_children():
            w.destroy()

    def _boja_prisutnosti(self, n: int, ukupno: int) -> tuple:
        """Vraća (bg, fg) boju ovisno o omjeru prisutnosti."""
        if ukupno == 0 or n == 0:
            return DANGER, DANGER_FG
        omjer = n / ukupno
        if omjer == 1.0:
            return SUCCESS, SUCCESS_FG
        elif omjer >= 0.66:
            return WARN_3, WARN_3_FG
        elif omjer >= 0.33:
            return WARN_2, WARN_2_FG
        else:
            return WARN_1, WARN_1_FG

    def _prikazi_rezultate(self, prisutnost: dict):
        self._ocisti_rezultate()

        # Izbroji koliko puta je svaki student prisutan kroz sve snimke
        ukupno_snimki = len(self.svi_rezultati)
        broj_prisutnosti = {}
        for ime in prisutnost:
            broj = sum(
                1 for _, (prepoznati_sn, _, _, _) in self.svi_rezultati.items()
                if ime in prepoznati_sn
            )
            broj_prisutnosti[ime] = broj

        sortirani = sorted(prisutnost.keys())
        stupci    = 4

        for i, ime in enumerate(sortirani):
            n       = broj_prisutnosti[ime]
            bg_boja, fg_boja = self._boja_prisutnosti(n, ukupno_snimki)
            red, col = divmod(i, stupci)

            okvir = tk.Frame(self.frame_rez, bg=bg_boja,
                             highlightbackground=fg_boja,
                             highlightthickness=1)
            okvir.grid(row=red, column=col, padx=3, pady=3, sticky="ew")

            tk.Label(
                okvir,
                text=f"{ime}",
                bg=bg_boja, fg=fg_boja,
                font=(FONT, 9, "bold"),
                anchor="center", pady=4
            ).pack(fill="x", padx=4)

            tk.Label(
                okvir,
                text=f"{n}/{ukupno_snimki}",
                bg=bg_boja, fg=fg_boja,
                font=(FONT, 8),
                anchor="center", pady=2
            ).pack(fill="x", padx=4)

        for col in range(stupci):
            self.frame_rez.columnconfigure(col, weight=1, uniform="col")

    # ----------------------------------------------------------------
    # SPREMANJE
    # ----------------------------------------------------------------
    def _spremi(self):
        if not self.prisutnost:
            messagebox.showwarning("Upozorenje", "Nema rezultata za spremanje!")
            return

        fmt = self.var_format.get()
        if self.timestamp_analize:
            from datetime import datetime
            dt = datetime.strptime(self.timestamp_analize, "%d.%m.%Y. u %H:%M:%S")
            ts = dt.strftime("%Y-%m-%d_%H-%M-%S")
        else:
            ts = "prisutnost"

        os.makedirs("rezultati", exist_ok=True)

        if fmt == "xlsx":
            putanja = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel datoteka", "*.xlsx")],
                initialdir=os.path.join(os.getcwd(), "rezultati"),
                initialfile=f"prisutnost_{ts}.xlsx"
            )
            if putanja:
                spremi_excel(self.prisutnost, self.svi_rezultati, putanja,
                             timestamp=self.timestamp_analize)
                self._log(f"Spremljeno: {putanja}")
                messagebox.showinfo("Spremljeno", f"Excel tablica spremljena u:\n{putanja}")
        else:
            putanja = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text datoteka", "*.txt")],
                initialdir=os.path.join(os.getcwd(), "rezultati"),
                initialfile=f"prisutnost_{ts}.txt"
            )
            if putanja:
                spremi_rezultate(self.prisutnost, self.svi_rezultati, putanja,
                                 timestamp=self.timestamp_analize)
                self._log(f"Spremljeno: {putanja}")
                messagebox.showinfo("Spremljeno", f"Rezultati spremljeni u:\n{putanja}")


# ================================================================
# POKRETANJE
# ================================================================
if __name__ == "__main__":
    app = App()
    app.mainloop()