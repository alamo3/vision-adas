import os
import tkinter
import tkintermapview
import customtkinter


class DataMapVisualization(customtkinter.CTk):
    WIDTH = 1280
    HEIGHT = 720

    def __init__(self):
        super().__init__()

        self.title("Field experiment visualization")
        self.geometry(f"{DataMapVisualization.WIDTH}x{DataMapVisualization.HEIGHT}")

        self.grid_columnconfigure(0, weight=0)
        self.grid_rowconfigure(0, weight=1)

        self.frame_disp = customtkinter.CTkFrame(master=self, width=DataMapVisualization.WIDTH, corner_radius=0)
        self.frame_disp.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

        self.map_widget = tkintermapview.TkinterMapView(self.frame_disp, width=DataMapVisualization.WIDTH, height=DataMapVisualization.HEIGHT, corner_radius=0)
        self.map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}&s=Ga", max_zoom=22)
        self.map_widget.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)
        self.map_widget.set_address("1276 Vanrose Street Mississauga")


if __name__ == "__main__":
    viz = DataMapVisualization()
    viz.mainloop()
