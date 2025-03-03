import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class GrayscaleConverter:
    def __init__(self, master):
        self.master = master
        self.master.title("Convertidor a Escala de Grises")
        self.master.geometry("300x200")

        self.original_image = None
        self.grayscale_image = None

        self.open_button = tk.Button(self.master, text="Abrir", command=self.open_image)
        self.open_button.pack(pady=20)

        self.save_button = tk.Button(self.master, text="Guardar", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(pady=20)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if file_path:
            self.original_image = Image.open(file_path)
            self.grayscale_image = self.original_image.convert('L')
            self.save_button.config(state=tk.NORMAL)
            messagebox.showinfo("Éxito", "Imagen convertida a escala de grises.")

    def save_image(self):
        if self.grayscale_image:
            save_path = filedialog.asksaveasfilename(defaultextension=".png", 
                                                     filetypes=[("PNG files", "*.png"), 
                                                                ("JPEG files", "*.jpg"),
                                                                ("All files", "*.*")])
            if save_path:
                self.grayscale_image.save(save_path)
                messagebox.showinfo("Éxito", f"Imagen guardada en {save_path}")
        else:
            messagebox.showerror("Error", "No hay imagen para guardar.")

if __name__ == "__main__":
    root = tk.Tk()
    app = GrayscaleConverter(root)
    root.mainloop()
