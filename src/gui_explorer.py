import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.processing import filtrar_datos, clasificar_datos_ttc
from src.plotting import graficar_analisis

class DataExplorerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tyre Data Explorer")
        
        # Cargar y clasificar datos
        df_raw = pd.read_parquet('results/dataset_clasificado.parquet')
        self.df = clasificar_datos_ttc(df_raw)
        
        # UI Elements
        self._setup_widgets()
        
        # Plot area
        self.fig_frame = tk.Frame(self.root)
        self.fig_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
    def _setup_widgets(self):
        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Dropdowns
        ttk.Label(frame, text="FZ_nom:").pack()
        self.fz_dropdown = ttk.Combobox(frame, values=sorted(self.df['FZ_nom'].unique()), state="readonly")
        self.fz_dropdown.pack()
        self.fz_dropdown.bind("<<ComboboxSelected>>", self.update_dependent_dropdowns)
        
        ttk.Label(frame, text="P_nom:").pack()
        self.p_dropdown = ttk.Combobox(frame, state="readonly")
        self.p_dropdown.pack()
        self.p_dropdown.bind("<<ComboboxSelected>>", self.update_ia_dropdown)
        
        ttk.Label(frame, text="IA_nom:").pack()
        self.ia_dropdown = ttk.Combobox(frame, state="readonly")
        self.ia_dropdown.pack()
        
        ttk.Label(frame, text="Variable Y:").pack()
        self.force_dropdown = ttk.Combobox(frame, values=["FY", "FX"], state="readonly")
        self.force_dropdown.set("FY")
        self.force_dropdown.pack()
        
        ttk.Button(frame, text="Actualizar Plot", command=self.update_plot).pack(pady=10)

    def update_dependent_dropdowns(self, event):
        fz = float(self.fz_dropdown.get())
        df_fz = self.df[self.df['FZ_nom'] == fz]
        
        self.p_dropdown['values'] = sorted(df_fz['P_nom'].unique())
        self.p_dropdown.set('')
        self.ia_dropdown.set('')
        self.ia_dropdown['values'] = []

    def update_ia_dropdown(self, event):
        fz = float(self.fz_dropdown.get())
        p = float(self.p_dropdown.get())
        df_filt = self.df[(self.df['FZ_nom'] == fz) & (self.df['P_nom'] == p)]
        
        self.ia_dropdown['values'] = sorted(df_filt['IA_nom'].unique())
        self.ia_dropdown.set('')

    def update_plot(self):
        try:
            criterios = {
                'FZ_nom': float(self.fz_dropdown.get()),
                'P_nom': float(self.p_dropdown.get()),
                'IA_nom': float(self.ia_dropdown.get())
            }
            force_type = self.force_dropdown.get()
        except ValueError:
            print("Selecciona valores válidos.")
            return
        
        df_filt = filtrar_datos(self.df, criterios)
        
        if df_filt.empty:
            return
        
        col_x = "SL" if force_type == "FX" else "SA"
        fig = graficar_analisis(df_filt, col_x, force_type)
        
        for widget in self.fig_frame.winfo_children():
            widget.destroy()
            
        canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = DataExplorerGUI(root)
    root.mainloop()
