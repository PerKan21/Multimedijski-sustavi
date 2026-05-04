"""
GUI - Sustav za popisivanje studenata
======================================
Moderno Tkinter sučelje. Pokretanje: python gui.py

Drag & drop podrška zahtijeva:
    pip install tkinterdnd2
Ako paket nije dostupan, zona radi samo na klik.

Ova verzija zadržava drag & drop zonu, uklanja donji debug log,
dodaje scroll cijele aplikacije i automatski smanjuje višak praznog prostora.
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import main as cfg
from model import ucitaj_model, ucitaj_bazu, izracunaj_pragove
from analiza import obradi_snimku, spremi_rezultate, spremi_excel

# ================================================================
# DRAG & DROP — opcionalni import, graceful fallback
# ================================================================
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    DND_DOSTUPAN = True
except ImportError:
    DND_DOSTUPAN = False


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
WARN_1     = "#7a4a1a"
WARN_1_FG  = "#e8943a"
WARN_2     = "#6a6a10"
WARN_2_FG  = "#d4d44a"
WARN_3     = "#2a6a2a"
WARN_3_FG  = "#7ecf85"
DANGER     = "#7a3d3d"
DANGER_FG  = "#cf7e7e"
DND_HOVER  = "#1a3a4a"
DND_BORDER = "#4a9abf"
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
        self.postavi(1.0, boja=SUCCESS_FG)


# ================================================================
# UNIFIED ZONA — drag & drop + klik
# ================================================================
class SnimkeZona:

    _TEKST_PRAZAN = "⬇  Povuci datoteke / mapu ovdje\n ili klikni za odabir"
    _TEKST_HOVER  = "⬇  Ispusti ovdje"

    def __init__(self, roditelj, callback_promjena):
        self._callback  = callback_promjena
        self._datoteke  = []
        self._popup     = None

        self.frame = tk.Frame(
            roditelj,
            bg=SURFACE2,
            highlightbackground=DND_BORDER,
            highlightthickness=1,
            cursor="hand2"
        )

        self._zona = tk.Frame(self.frame, bg=SURFACE2, cursor="hand2")
        self._zona.pack(fill="x")

        self._ikona = tk.Label(
            self._zona, text="📂", bg=SURFACE2, fg=TEXT,
            font=(FONT, 17), pady=4, cursor="hand2"
        )
        self._ikona.pack()

        self._lbl_glavni = tk.Label(
            self._zona,
            text=self._TEKST_PRAZAN,
            bg=SURFACE2, fg=TEXT_MUTE,
            font=(FONT, 10),
            justify="center", pady=2,
            cursor="hand2"
        )
        self._lbl_glavni.pack()

        if DND_DOSTUPAN:
            hint = "Klikni za odabir  ·  ili povuci datoteke/mapu"
        else:
            hint = "Klikni za odabir"

        self._lbl_hint = tk.Label(
            self._zona,
            text=hint,
            bg=SURFACE2, fg=TEXT_MUTE,
            font=(FONT, 8), pady=3,
            cursor="hand2"
        )
        self._lbl_hint.pack()

        tk.Frame(self.frame, bg=BORDER, height=1).pack(fill="x")

        self._dno = tk.Frame(self.frame, bg=SURFACE2)
        self._dno.pack(fill="x", padx=10, pady=4)

        self._lbl_info = tk.Label(
            self._dno,
            text="Nema odabranih snimki.",
            bg=SURFACE2, fg=TEXT_MUTE,
            font=(FONT, 9),
            anchor="w", justify="left"
        )
        self._lbl_info.pack(side="left", fill="x", expand=True)

        self._btn_ocisti = tk.Button(
            self._dno, text="✕  Očisti",
            command=self.ocisti,
            bg=SURFACE2, fg=TEXT_MUTE,
            activebackground=BORDER, activeforeground=TEXT,
            relief="flat", cursor="hand2",
            font=(FONT, 8), padx=8, pady=2
        )
        self._btn_ocisti.pack(side="right")
        self._btn_ocisti.pack_forget()

        self._klikabilni = [self._zona, self._ikona, self._lbl_glavni, self._lbl_hint]
        for w in self._klikabilni:
            w.bind("<Button-1>", self._otvori_popup)
            w.bind("<Enter>",    self._hover_enter)
            w.bind("<Leave>",    self._hover_leave)

        if DND_DOSTUPAN:
            for w in self._klikabilni + [self.frame]:
                w.drop_target_register(DND_FILES)
                w.dnd_bind("<<DropEnter>>", self._dnd_enter)
                w.dnd_bind("<<DropLeave>>", self._dnd_leave)
                w.dnd_bind("<<Drop>>",      self._dnd_drop)

    def pack(self, **kw):
        self.frame.pack(**kw)

    def _otvori_popup(self, event=None):
        if self._popup and self._popup.winfo_exists():
            self._popup.destroy()
            self._popup = None
            return

        popup = tk.Toplevel(self.frame)
        popup.overrideredirect(True)
        popup.configure(bg=BORDER)
        popup.attributes("-topmost", True)
        self._popup = popup

        self.frame.update_idletasks()
        x = self.frame.winfo_rootx()
        y = self.frame.winfo_rooty() + self.frame.winfo_height() + 2
        popup.geometry(f"+{x}+{y}")

        def _stil_gumba(tekst, naredba):
            b = tk.Button(
                popup, text=tekst, command=naredba,
                bg=SURFACE2, fg=TEXT,
                activebackground=BORDER, activeforeground=SUCCESS_FG,
                relief="flat", cursor="hand2",
                font=(FONT, 10),
                anchor="w", padx=16, pady=8,
                width=28
            )
            b.pack(fill="x", padx=1, pady=1)
            return b

        _stil_gumba("🎵   Odaberi datoteke",  self._popup_datoteke)
        _stil_gumba("📁   Odaberi mapu",      self._popup_mapa)

        popup.bind("<FocusOut>", lambda e: self._zatvori_popup())
        popup.focus_set()

    def _zatvori_popup(self):
        if self._popup and self._popup.winfo_exists():
            self._popup.destroy()
        self._popup = None

    def _popup_datoteke(self):
        self._zatvori_popup()
        tipovi = [
            ("Audio datoteke", " ".join(f"*{e}" for e in cfg.PODRZANI_FORMATI)),
            ("Sve datoteke", "*.*")
        ]
        odabrano = filedialog.askopenfilenames(
            title="Odaberi audio datoteke",
            filetypes=tipovi
        )
        if odabrano:
            self._dodaj_datoteke(list(odabrano))

    def _popup_mapa(self):
        self._zatvori_popup()
        mapa = filedialog.askdirectory(title="Odaberi mapu sa snimkama")
        if mapa:
            self._dodaj_iz_mape(mapa)

    def _hover_enter(self, event=None):
        self.frame.configure(highlightbackground=SUCCESS_FG)

    def _hover_leave(self, event=None):
        self.frame.configure(highlightbackground=DND_BORDER)

    def _dnd_enter(self, event=None):
        self.frame.configure(bg=DND_HOVER, highlightbackground=SUCCESS_FG)
        self._zona.configure(bg=DND_HOVER)
        self._ikona.configure(bg=DND_HOVER)
        self._lbl_glavni.configure(bg=DND_HOVER, fg=SUCCESS_FG,
                                   text=self._TEKST_HOVER)
        self._lbl_hint.configure(bg=DND_HOVER)

    def _dnd_leave(self, event=None):
        self._resetiraj_izgled()

    def _dnd_drop(self, event):
        self._resetiraj_izgled()
        putanje = self._parsiraj_putanje(event.data)
        if not putanje:
            return

        if len(putanje) == 1 and os.path.isdir(putanje[0]):
            self._dodaj_iz_mape(putanje[0])
            return

        audio = []
        dodana_mapa = False
        for p in putanje:
            if (os.path.isfile(p)
                    and p.lower().endswith(cfg.PODRZANI_FORMATI)
                    and not p.endswith("_konv.wav")):
                audio.append(p)
            elif os.path.isdir(p):
                dodana_mapa = True
                self._dodaj_iz_mape(p, tiho=True)

        if audio:
            self._dodaj_datoteke(audio)
        elif not dodana_mapa:
            messagebox.showwarning(
                "Drag & Drop",
                "Nisu pronađene podržane audio datoteke.\n"
                f"Podržani formati: {', '.join(cfg.PODRZANI_FORMATI)}"
            )

    def _resetiraj_izgled(self):
        self.frame.configure(bg=SURFACE2, highlightbackground=DND_BORDER)
        self._zona.configure(bg=SURFACE2)
        self._ikona.configure(bg=SURFACE2)
        self._lbl_glavni.configure(bg=SURFACE2, fg=TEXT_MUTE,
                                   text=self._TEKST_PRAZAN)
        self._lbl_hint.configure(bg=SURFACE2)

    def _dodaj_iz_mape(self, mapa: str, tiho: bool = False):
        nove = [
            os.path.join(mapa, f)
            for f in sorted(os.listdir(mapa))
            if f.lower().endswith(cfg.PODRZANI_FORMATI)
            and not f.endswith("_konv.wav")
        ]
        if not nove and not tiho:
            messagebox.showwarning(
                "Odabir mape",
                f"Mapa ne sadrži podržane audio datoteke:\n{mapa}"
            )
            return
        self._dodaj_datoteke(nove)

    def _dodaj_datoteke(self, putanje: list):
        for p in putanje:
            if p not in self._datoteke:
                self._datoteke.append(p)
        self._datoteke.sort()
        self._azuriraj_prikaz()
        self._callback()

    def ocisti(self):
        self._datoteke = []
        self._azuriraj_prikaz()
        self._callback()

    def _azuriraj_prikaz(self):
        n = len(self._datoteke)
        if n == 0:
            self._lbl_info.configure(text="Nema odabranih snimki.", fg=TEXT_MUTE)
            self._lbl_glavni.configure(text=self._TEKST_PRAZAN)
            self._ikona.configure(text="📂", fg=TEXT)
            self._btn_ocisti.pack_forget()
        else:
            nazivi = [os.path.basename(p) for p in self._datoteke]
            if n <= 2:
                prikaz = ",  ".join(nazivi)
            else:
                prikaz = f"{nazivi[0]},  {nazivi[1]}  … (+{n - 2} više)"
            self._lbl_info.configure(
                text=f"🎵  {n} {'snimka' if n == 1 else 'snimki'}:  {prikaz}",
                fg=SUCCESS_FG
            )
            self._lbl_glavni.configure(
                text="⬇  Povuci još datoteka / mapu\n ili klikni za dodavanje"
            )
            self._ikona.configure(text=f"🎵  ×{n}", fg=SUCCESS_FG)
            self._btn_ocisti.pack(side="right")

    @property
    def datoteke(self) -> list:
        return self._datoteke[:]

    @property
    def ima_datoteke(self) -> bool:
        return bool(self._datoteke)

    @staticmethod
    def _parsiraj_putanje(data: str) -> list:
        putanje = []
        data = data.strip()
        i = 0
        while i < len(data):
            if data[i] == "{":
                kraj = data.find("}", i)
                if kraj == -1:
                    putanje.append(data[i + 1:].strip())
                    break
                putanje.append(data[i + 1:kraj])
                i = kraj + 1
            elif data[i] == " ":
                i += 1
            else:
                kraj = data.find(" ", i)
                if kraj == -1:
                    putanje.append(data[i:])
                    break
                putanje.append(data[i:kraj])
                i = kraj + 1
        return [p.strip() for p in putanje if p.strip()]


# ================================================================
# GLAVNA APLIKACIJA
# ================================================================
_BaseClass = TkinterDnD.Tk if DND_DOSTUPAN else tk.Tk


class App(_BaseClass):
    def __init__(self):
        super().__init__()
        self.title("Sustav za popisivanje studenata")
        self.geometry("780x820")
        self.minsize(760, 600)
        self.configure(bg=BG)
        self.resizable(True, True)

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
        self.lbl_model.configure(text="⏳ Učitavanje modela...")
        self.after(120, self._prilagodi_visinu_prozora)
        threading.Thread(target=self._ucitaj_model, daemon=True).start()

    # ----------------------------------------------------------------
    # IZGRADNJA UI
    # ----------------------------------------------------------------
    def _izgradnja_ui(self):
        # Glavni scroll container: kad je prozor niži od sadržaja,
        # pojavi se scrollbar za cijelu aplikaciju. Kad sve stane,
        # scrollbar se automatski sakrije.
        self.scroll_canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        self.scroll_canvas.pack(side="left", fill="both", expand=True)

        self.scrollbar = tk.Scrollbar(
            self,
            orient="vertical",
            command=self.scroll_canvas.yview
        )
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.main_frame = tk.Frame(self.scroll_canvas, bg=BG)
        self._main_window = self.scroll_canvas.create_window(
            (0, 0),
            window=self.main_frame,
            anchor="nw"
        )

        self.main_frame.bind("<Configure>", self._azuriraj_scroll_region)
        self.scroll_canvas.bind("<Configure>", self._azuriraj_canvas_sirinu)
        self.bind_all("<MouseWheel>", self._mousewheel_scroll)

        zaglavlje = tk.Frame(self.main_frame, bg=BG)
        zaglavlje.pack(fill="x", padx=18, pady=(12, 0))
        label(zaglavlje, "Sustav za popisivanje studenata",
              vel=17, bold=True, bg=BG).pack(anchor="center")

        sep = tk.Frame(self.main_frame, bg=BORDER, height=1)
        sep.pack(fill="x", padx=18, pady=(8, 10))

        self._sekcija_baza()
        self._sekcija_snimke()
        self._sekcija_pokretanje()
        self._sekcija_rezultati()

    def _azuriraj_scroll_region(self, event=None):
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))
        self._azuriraj_scrollbar_vidljivost()

    def _azuriraj_canvas_sirinu(self, event=None):
        if event is not None:
            self.scroll_canvas.itemconfig(self._main_window, width=event.width)
        self._azuriraj_scrollbar_vidljivost()

    def _azuriraj_scrollbar_vidljivost(self):
        self.update_idletasks()
        bbox = self.scroll_canvas.bbox("all")
        if not bbox:
            return

        sadrzaj_visina = bbox[3] - bbox[1]
        canvas_visina = self.scroll_canvas.winfo_height()

        if sadrzaj_visina > canvas_visina + 2:
            if not self.scrollbar.winfo_ismapped():
                self.scrollbar.pack(side="right", fill="y")
            self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)
        else:
            if self.scrollbar.winfo_ismapped():
                self.scrollbar.pack_forget()
            self.scroll_canvas.yview_moveto(0)
            self.scroll_canvas.configure(yscrollcommand=lambda *args: None)

    def _mousewheel_scroll(self, event):
        bbox = self.scroll_canvas.bbox("all")
        if not bbox:
            return
        sadrzaj_visina = bbox[3] - bbox[1]
        canvas_visina = self.scroll_canvas.winfo_height()
        if sadrzaj_visina <= canvas_visina:
            return

        # Windows: event.delta je najčešće ±120
        self.scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _prilagodi_visinu_prozora(self):
        """
        Uklanja veliki prazni prostor ispod zadnje sekcije kad je prozor
        viši od stvarnog sadržaja. Ako sadržaj ne stane na ekran, prozor
        ostaje unutar ekrana i koristi se scrollbar cijele aplikacije.
        """
        try:
            self.update_idletasks()
            if self.state() != "normal":
                return

            bbox = self.scroll_canvas.bbox("all")
            if not bbox:
                return

            sadrzaj_visina = bbox[3] - bbox[1]
            # Malo prostora za title bar / rubove prozora.
            zeljena_visina = sadrzaj_visina + 44
            ekran_visina = self.winfo_screenheight()
            zeljena_visina = max(600, min(zeljena_visina, ekran_visina - 80))

            trenutna_sirina = max(self.winfo_width(), 760)
            trenutna_visina = self.winfo_height()

            # Ne diraj prozor zbog par piksela, samo kad je razlika očita.
            if abs(trenutna_visina - zeljena_visina) > 28:
                self.geometry(f"{trenutna_sirina}x{int(zeljena_visina)}")

            self.after(60, self._azuriraj_scrollbar_vidljivost)
        except Exception:
            pass

    def _sekcija_baza(self):
        k = tk.Frame(self.main_frame, bg=SURFACE,
                     highlightbackground=BORDER, highlightthickness=1)
        k.pack(fill="x", padx=18, pady=(0, 8))

        unutar = tk.Frame(k, bg=SURFACE)
        unutar.pack(fill="x", padx=12, pady=10)

        label(unutar, "Baza govornika", bold=True, vel=11).pack(anchor="w", pady=(0, 8))

        red = tk.Frame(unutar, bg=SURFACE)
        red.pack(fill="x")
        self.var_baza = tk.StringVar(value=cfg.DIR_BAZA)
        unos(red, self.var_baza).pack(side="left", fill="x", expand=True, padx=(0, 8))
        gumb(red, "Odaberi", self._odaberi_bazu).pack(side="left", padx=(0, 4))
        gumb(red, "Izgradi", self._izgradi_bazu).pack(side="left", padx=(0, 4))
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
        k = tk.Frame(self.main_frame, bg=SURFACE,
                     highlightbackground=BORDER, highlightthickness=1)
        k.pack(fill="x", padx=18, pady=(0, 8))

        unutar = tk.Frame(k, bg=SURFACE)
        unutar.pack(fill="x", padx=12, pady=10)

        label(unutar, "Ulazne snimke", bold=True, vel=11).pack(anchor="w", pady=(0, 8))

        self.snimke_zona = SnimkeZona(unutar, callback_promjena=self._snimke_promijenjene)
        self.snimke_zona.pack(fill="x")

    def _sekcija_pokretanje(self):
        k = tk.Frame(self.main_frame, bg=SURFACE,
                     highlightbackground=BORDER, highlightthickness=1)
        k.pack(fill="x", padx=18, pady=(0, 8))

        unutar = tk.Frame(k, bg=SURFACE)
        unutar.pack(fill="x", padx=12, pady=10)

        self.lbl_model = label(unutar, "", boja=TEXT_MUTE, vel=10)
        self.lbl_model.pack(anchor="w", pady=(0, 6))

        self.btn_run = gumb(unutar, "▶   Pokreni analizu", self._pokreni, state="disabled")
        self.btn_run.config(font=(FONT, 12, "bold"), pady=10)
        self.btn_run.pack(fill="x", pady=(0, 8))

        self.prog_analiza = ProgressBar(unutar)
        self.prog_analiza.pack(fill="x", pady=(0, 6))

        self.lbl_status = label(unutar, "—", boja=TEXT_MUTE, vel=11)
        self.lbl_status.pack(anchor="w")

    def _sekcija_rezultati(self):
        k = tk.Frame(self.main_frame, bg=SURFACE,
                     highlightbackground=BORDER, highlightthickness=1)
        k.pack(fill="x", padx=18, pady=(0, 8))

        unutar = tk.Frame(k, bg=SURFACE)
        unutar.pack(fill="x", padx=12, pady=10)

        label(unutar, "Popis prisutnosti", bold=True, vel=11).pack(anchor="w", pady=(0, 6))

        # Rezultati su običan frame bez scrollanja.
        # Za 8 studenata stanu u 2 reda × 4 stupca, a prozor je malo viši.
        self.frame_rez = tk.Frame(unutar, bg=SURFACE)
        self.frame_rez.pack(fill="x")

        dno = tk.Frame(unutar, bg=SURFACE)
        dno.pack(fill="x", pady=(8, 0))

        label(dno, "Spremi kao:", vel=10, boja=TEXT_MUTE).pack(side="left", padx=(0, 8))

        for tekst, vrijednost in [("Tekstualna datoteka (.txt)", "txt"),
                                   ("Excel tablica (.xlsx)", "xlsx")]:
            tk.Radiobutton(
                dno, text=tekst, variable=self.var_format, value=vrijednost,
                bg=SURFACE, fg=TEXT, selectcolor=SURFACE2,
                activebackground=SURFACE, activeforeground=TEXT,
                font=(FONT, 10), cursor="hand2"
            ).pack(side="left", padx=(0, 10))

        self.btn_spremi = gumb(unutar, "💾   Spremi rezultate", self._spremi)
        self.btn_spremi.config(font=(FONT, 12, "bold"), pady=12)
        self.btn_spremi.pack(fill="x", pady=(8, 0))

    def _sekcija_log(self):
        """Debug log je namjerno skriven u ovoj verziji GUI-ja."""
        pass

    # ----------------------------------------------------------------
    # CALLBACK
    # ----------------------------------------------------------------
    def _snimke_promijenjene(self):
        n = len(self.snimke_zona.datoteke)
        if n > 0:
            self._log(f"Snimke ažurirane: {n} {'datoteka' if n == 1 else 'datoteka'} odabrano.")

    # ----------------------------------------------------------------
    # LOG I STATUS
    # ----------------------------------------------------------------
    def _log(self, poruka: str):
        # Debug poruke više ne prikazujemo u GUI-ju da se ne zauzima prostor.
        # Ostavljamo metodu da postojeći pozivi u kodu ne pucaju.
        print(poruka)

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
            ucitaj_model()
            self.after(0, self._model_spreman)
        except Exception as e:
            self.after(0, self.lbl_model.configure, {"text": "✗ Model nije učitan", "fg": DANGER_FG})
            self.after(0, messagebox.showerror, "Greška", f"Model nije učitan:\n{e}")

    def _model_spreman(self):
        self.btn_run.configure(state="normal")
        self.lbl_model.configure(text="✓ Sustav spreman za analizu", fg=SUCCESS_FG)
        self.after(80, self._prilagodi_visinu_prozora)

    # ----------------------------------------------------------------
    # ODABIR BAZE
    # ----------------------------------------------------------------
    def _odaberi_bazu(self):
        mapa = filedialog.askdirectory(title="Odaberi mapu baze")
        if mapa:
            self.var_baza.set(mapa)

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
            try:
                from model import ucitaj_cache as _ucitaj_cache_fn
                baza = _ucitaj_cache_fn(cfg.CACHE_PUTANJA)
                if baza is None:
                    self.after(0, messagebox.showerror, "Greška", "Cache nije valjan.")
                    self.after(0, self.btn_run.configure, {"state": "normal"})
                    return
                prag_d, prag_g = izracunaj_pragove(baza, cfg.FAKTOR_GORNJEG_PRAGA)
                self.after(0, self._baza_gotova, baza, prag_d, prag_g, "✓  Učitana baza iz cachea")
            except Exception as e:
                self.after(0, messagebox.showerror, "Greška", f"Učitavanje cachea nije uspjelo:\n{e}")
                self.after(0, self.btn_run.configure, {"state": "normal"})

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
            try:
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
            except Exception as e:
                self.after(0, messagebox.showerror, "Greška", f"Izgradnja baze nije uspjela:\n{e}")
                self.after(0, self.btn_run.configure, {"state": "normal"})

        threading.Thread(target=_rad, daemon=True).start()

    def _baza_gotova(self, baza, prag_d, prag_g, poruka="✓  Baza izgrađena"):
        self.baza = baza

        if isinstance(cfg.FIKSNI_PRAG_DONJI, float) and 0.0 < cfg.FIKSNI_PRAG_DONJI < 1.0:
            prag_d = cfg.FIKSNI_PRAG_DONJI
            prag_g = prag_d * cfg.FAKTOR_GORNJEG_PRAGA
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
        self.after(80, self._prilagodi_visinu_prozora)

    # ----------------------------------------------------------------
    # POKRETANJE ANALIZE
    # ----------------------------------------------------------------
    def _pokreni(self):
        if not self.baza:
            messagebox.showwarning("Upozorenje", "Prvo izgradi bazu!")
            return

        if not self.snimke_zona.ima_datoteke:
            messagebox.showwarning("Upozorenje",
                                   "Nema odabranih snimki!\n"
                                   "Klikni na zonu ili povuci datoteke/mapu.")
            return

        snimke = [(os.path.basename(p), p) for p in self.snimke_zona.datoteke]

        self.btn_run.configure(state="disabled")
        self._ocisti_rezultate()
        self._status("Pokretanje...", 0.0)
        self._log(f"Analiza {len(snimke)} snimki...")

        def _rad():
            try:
                prisutnost    = {ime: False for ime in self.baza}
                svi_rezultati = {}
                ukupno = len(snimke)

                for i, (naziv, putanja) in enumerate(snimke):
                    prepoznati, segmenti, uljezi, n_gov = obradi_snimku(
                        putanja, self.baza, self.prag_d, self.prag_g,
                        sr=cfg.SR,
                        trajanje=cfg.SEGMENT_TRAJANJE,
                        preklapanje=cfg.SEGMENT_PREKLAPANJE,
                        prop_decrease=cfg.SUM_PROP_DECREASE,
                        vad_top_db=cfg.VAD_TOP_DB,
                        vad_min_duljina=cfg.VAD_MIN_DULJINA,
                        vad_spajanje=cfg.VAD_SPAJANJE,
                    )
                    for student in prepoznati:
                        prisutnost[student] = True
                    svi_rezultati[naziv] = (prepoznati, segmenti, uljezi, n_gov)
                    napredak = (i + 1) / ukupno
                    self.after(0, self._status,
                               f"{naziv}  ({i+1}/{ukupno})", napredak, False)

                self.after(0, self._analiza_gotova, prisutnost, svi_rezultati)
            except Exception as e:
                self.after(0, messagebox.showerror, "Greška", f"Analiza nije uspjela:\n{e}")
                self.after(0, self.btn_run.configure, {"state": "normal"})

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
        self.after(80, self._prilagodi_visinu_prozora)

    # ----------------------------------------------------------------
    # PRIKAZ REZULTATA
    # ----------------------------------------------------------------
    def _ocisti_rezultate(self):
        for w in self.frame_rez.winfo_children():
            w.destroy()

    def _boja_prisutnosti(self, n: int, ukupno: int) -> tuple:
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
            n = broj_prisutnosti[ime]
            bg_boja, fg_boja = self._boja_prisutnosti(n, ukupno_snimki)
            red, col = divmod(i, stupci)

            okvir = tk.Frame(self.frame_rez, bg=bg_boja,
                             highlightbackground=fg_boja, highlightthickness=1)
            okvir.grid(row=red, column=col, padx=3, pady=3, sticky="ew")

            tk.Label(okvir, text=f"{ime}", bg=bg_boja, fg=fg_boja,
                     font=(FONT, 9, "bold"), anchor="center", pady=2
                     ).pack(fill="x", padx=4)
            tk.Label(okvir, text=f"{n}/{ukupno_snimki}", bg=bg_boja, fg=fg_boja,
                     font=(FONT, 8), anchor="center", pady=1
                     ).pack(fill="x", padx=4)

        for col in range(stupci):
            self.frame_rez.columnconfigure(col, weight=1, uniform="col")

        self.after(50, self._prilagodi_visinu_prozora)


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
