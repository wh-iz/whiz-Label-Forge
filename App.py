import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
import sv_ttk
import subprocess
import threading
import cv2
from PIL import Image, ImageTk
import yaml
import tempfile
import time
import shutil
import random
from LabelEditor import LabelEditor
from NullGeneratorTab import NullGeneratorTab

CONFIG_FILE = "config.json"

class ToolTip:
    """Creates a tooltip for a given widget"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
    
    def show_tooltip(self, event=None):
        if self.tooltip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert") if hasattr(self.widget, 'bbox') else (0, 0, 0, 0)
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(tw, text=self.text, justify='left',
                        background="#2b2b2b", foreground="#ffffff",
                        relief='solid', borderwidth=1,
                        font=("Segoe UI", 9), padx=8, pady=6)
        label.pack()
    
    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("whiz label forge")
        self.geometry("1100x780")
        self.minsize(950, 700)

        sv_ttk.set_theme("dark")

        self.model_paths = {}

        # --- Main Layout ---
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)  # Notebook
        self.rowconfigure(1, weight=0)  # Footer (Logs/Status)

        # Notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 5))

        self.tab_dashboard = ttk.Frame(self.notebook, padding=10)
        self.tab_settings = ttk.Frame(self.notebook, padding=10)
        self.tab_train = ttk.Frame(self.notebook, padding=10)
        self.tab_tools = ttk.Frame(self.notebook, padding=10)
        self.tab_editor = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.tab_dashboard, text="Dashboard")
        self.notebook.add(self.tab_settings, text="Settings")
        self.notebook.add(self.tab_train, text="Training")
        
        self.tab_null = NullGeneratorTab(self.notebook, self)
        self.notebook.add(self.tab_null, text="Null Generator")

        self.notebook.add(self.tab_tools, text="Tools")
        self.notebook.add(self.tab_editor, text="Label Editor")

        # Variables
        self.video_path = tk.StringVar()
        self.youtube_link = tk.StringVar()
        self.output_path = tk.StringVar()
        self.conf_threshold = tk.DoubleVar(value=0.5)
        self.iou_threshold = tk.DoubleVar(value=0.45)
        self.frame_step = tk.IntVar(value=1)
        self.merge_iou = tk.DoubleVar(value=0.6)
        self.use_gpu = tk.BooleanVar(value=True)
        self.similarity_threshold = tk.IntVar(value=5)
        self.history_size = tk.IntVar(value=30)
        self.input_size = tk.IntVar(value=640)
        self.target_class_id = tk.IntVar(value=-1)
        self.save_raw = tk.BooleanVar(value=True)
        self.save_empty = tk.BooleanVar(value=False)
        self.save_annotated = tk.BooleanVar(value=True)
        
        self.train_epochs = tk.IntVar(value=50)
        self.train_batch = tk.IntVar(value=16)
        self.train_imgsz = tk.IntVar(value=640)
        self.train_device = tk.StringVar(value="auto")
        self.train_model = tk.StringVar(value="yolov8n.pt")
        
        self.cancel_flag = False
        self.pause_flag = False
        self.last_saved_paths = None

        self.setup_dashboard_tab()
        self.setup_settings_tab()
        self.setup_train_tab()
        self.setup_tools_tab()
        self.setup_editor_tab()
        self.setup_footer()
        
        self.load_config()

    def setup_editor_tab(self):
        self.label_editor = LabelEditor(self.tab_editor, app_instance=self)
        self.label_editor.pack(fill="both", expand=True)
        # Bind tab change to refresh file list
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

    def on_tab_change(self, event):
        selected_tab = event.widget.select()
        tab_text = event.widget.tab(selected_tab, "text")
        if tab_text == "Label Editor":
            self.label_editor.refresh_file_list()

    def setup_dashboard_tab(self):
        self.tab_dashboard.columnconfigure(0, weight=4) # Input/Config column
        self.tab_dashboard.columnconfigure(1, weight=6) # Preview column
        self.tab_dashboard.rowconfigure(0, weight=1)

        # --- Left Column ---
        left_col = ttk.Frame(self.tab_dashboard)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_col.columnconfigure(0, weight=1)

        # Input Source
        input_group = ttk.LabelFrame(left_col, text="Input Source", padding=10)
        input_group.pack(fill="x", pady=(0, 10))
        
        ttk.Label(input_group, text="Video File:").pack(anchor="w")
        vid_frame = ttk.Frame(input_group)
        vid_frame.pack(fill="x", pady=(5, 10))
        ttk.Entry(vid_frame, textvariable=self.video_path).pack(side="left", fill="x", expand=True)
        ttk.Button(vid_frame, text="Browse", command=self.browse_video).pack(side="right", padx=(5, 0))

        ttk.Label(input_group, text="YouTube URL:").pack(anchor="w")
        ttk.Entry(input_group, textvariable=self.youtube_link).pack(fill="x", pady=(5, 10))

        ttk.Label(input_group, text="YT Res:").pack(anchor="w")
        self.yt_res = ttk.Combobox(input_group, values=["360p", "480p", "720p", "1080p", "1440p", "2160p"])
        self.yt_res.current(3)
        self.yt_res.pack(fill="x", pady=(5, 0))

        # Configuration (Moved to Settings)
        
        # Actions
        action_group = ttk.LabelFrame(left_col, text="Actions", padding=10)
        action_group.pack(fill="x", pady=(0, 10))

        
        self.run_btn = ttk.Button(action_group, text="▶ RUN DETECTION", command=self.run_detection)
        self.run_btn.pack(fill="x", pady=(0, 5))
        
        self.pause_button = ttk.Button(action_group, text="Pause", command=self.toggle_pause)
        self.pause_button.pack(fill="x", pady=(0, 5))
        
        self.cancel_button = ttk.Button(action_group, text="Cancel", command=self.cancel_detection)
        self.cancel_button.pack(fill="x")
        self.cancel_button.pack_forget()

        # --- Right Column: Preview ---
        right_col = ttk.Frame(self.tab_dashboard)
        right_col.grid(row=0, column=1, sticky="nsew")
        
        preview_group = ttk.LabelFrame(right_col, text="Preview (Scroll to Zoom)", padding=5)
        preview_group.pack(fill="both", expand=True)
        
        self.preview_image_label = ttk.Label(preview_group, anchor="center")
        self.preview_image_label.pack(fill="both", expand=True)

        ctrl_frame = ttk.Frame(preview_group)
        ctrl_frame.pack(fill="x", pady=5)
        # Placeholder for previous/next/delete logic if I had a viewer mode, 
        # but for now just "Delete Last"
        ttk.Button(ctrl_frame, text="Delete Last Saved", command=self.delete_last_saved).pack(side="right")

    def setup_settings_tab(self):
        self.tab_settings.columnconfigure(0, weight=1)
        self.tab_settings.columnconfigure(1, weight=1)
        self.tab_settings.rowconfigure(0, weight=1)

        # --- Left Column: Configuration & System ---
        left_col = ttk.Frame(self.tab_settings)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Paths & Models
        config_group = ttk.LabelFrame(left_col, text="Paths & Models", padding=15)
        config_group.pack(fill="x", pady=(0, 15))

        ttk.Label(config_group, text="Output Folder:").pack(anchor="w")
        out_frame = ttk.Frame(config_group)
        out_frame.pack(fill="x", pady=(5, 10))
        ttk.Entry(out_frame, textvariable=self.output_path).pack(side="left", fill="x", expand=True)
        ttk.Button(out_frame, text="Open", command=self.open_output_folder).pack(side="right", padx=(5, 0))
        ttk.Button(out_frame, text="...", width=3, command=self.browse_output).pack(side="right", padx=(5, 0))

        ttk.Label(config_group, text="Models:").pack(anchor="w")
        list_frame = ttk.Frame(config_group)
        list_frame.pack(fill="x", pady=(5, 5))
        self.models_listbox = tk.Listbox(list_frame, height=5, selectmode="multiple", bg="#1e1e1e", fg="#f0f0f0", borderwidth=0, highlightthickness=1)
        self.models_listbox.pack(side="left", fill="x", expand=True)
        ttk.Scrollbar(list_frame, command=self.models_listbox.yview).pack(side="right", fill="y")

        btn_frame = ttk.Frame(config_group)
        btn_frame.pack(fill="x", pady=(5, 10))
        ttk.Button(btn_frame, text="+ Add", command=self.add_model).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(btn_frame, text="- Remove", command=self.remove_model).pack(side="right", fill="x", expand=True)

        ttk.Label(config_group, text="Output Resolution:").pack(anchor="w")
        self.out_res = ttk.Combobox(config_group, values=["Off", "640x640", "320x320", "1280x1280"])
        self.out_res.current(0)
        self.out_res.pack(fill="x", pady=(5, 0))

        # System
        sys_group = ttk.LabelFrame(left_col, text="System", padding=15)
        sys_group.pack(fill="x")

        ttk.Checkbutton(sys_group, text="Use GPU (CUDA/DirectML)", variable=self.use_gpu).pack(anchor="w", pady=2)
        ttk.Checkbutton(sys_group, text="Save Raw Images", variable=self.save_raw).pack(anchor="w", pady=2)
        ttk.Checkbutton(sys_group, text="Save Empty Frames", variable=self.save_empty).pack(anchor="w", pady=2)
        ttk.Checkbutton(sys_group, text="Save Annotated Images", variable=self.save_annotated).pack(anchor="w", pady=2)
        
        ttk.Button(sys_group, text="Save Configuration", command=self.save_config).pack(fill="x", pady=(10, 0))

        # --- Right Column: Parameters ---
        right_col = ttk.Frame(self.tab_settings)
        right_col.grid(row=0, column=1, sticky="nsew")

        # Detection Params
        det_group = ttk.LabelFrame(right_col, text="Detection Parameters", padding=15)
        det_group.pack(fill="x", pady=(0, 15))

        for label, var, vmin, vmax, step, tip in [
            ("Confidence Thresh", self.conf_threshold, 0.01, 1.0, 0.01, "Minimum confidence score to detect an object"),
            ("IoU Thresh", self.iou_threshold, 0.01, 1.0, 0.01, "Intersection over Union threshold for NMS"),
            ("Merge IoU", self.merge_iou, 0.01, 1.0, 0.01, "IoU threshold for merging overlapping boxes from ensemble")
        ]:
            row = ttk.Frame(det_group)
            row.pack(fill="x", pady=5)
            ttk.Label(row, text=label, width=20).pack(side="left")
            scale = ttk.Scale(row, variable=var, from_=vmin, to=vmax)
            scale.pack(side="left", fill="x", expand=True, padx=10)
            lbl = ttk.Label(row, text=f"{var.get():.2f}", width=5)
            lbl.pack(side="right")
            scale.configure(command=lambda v, l=lbl: l.config(text=f"{float(v):.2f}"))
            ToolTip(scale, tip)

        # Advanced Settings
        adv_group = ttk.LabelFrame(right_col, text="Advanced Settings", padding=15)
        adv_group.pack(fill="x")

        grid = ttk.Frame(adv_group)
        grid.pack(fill="x")
        
        row = 0
        for label, var, vmin, vmax, inc, tip in [
            ("Frame Step", self.frame_step, 1, 60, 1, "Process every Nth frame (1 = every frame)"),
            ("History Size", self.history_size, 5, 100, 1, "Number of frames to keep in history for duplicate detection"),
            ("Sim. Threshold", self.similarity_threshold, 0, 64, 1, "Perceptual hash hamming distance threshold (0=identical, 64=diff)"),
            ("Input Size", self.input_size, 256, 1536, 32, "Resize input frame to this size for inference"),
            ("Target Class ID", self.target_class_id, -1, 80, 1, "Override all detected classes to this ID (-1 to disable)")
        ]:
            ttk.Label(grid, text=label).grid(row=row, column=0, sticky="w", pady=5, padx=(0, 10))
            spin = ttk.Spinbox(grid, from_=vmin, to=vmax, increment=inc, textvariable=var, width=10)
            spin.grid(row=row, column=1, sticky="w", pady=5)
            ToolTip(spin, tip)
            row += 1

    def setup_train_tab(self):
        container = ttk.Frame(self.tab_train)
        container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Training Configuration Group
        train_group = ttk.LabelFrame(container, text="Training Configuration", padding=20)
        train_group.place(relx=0.5, rely=0.5, anchor="center")
        
        grid_frame = ttk.Frame(train_group)
        grid_frame.pack(fill="both", expand=True)

        ttk.Label(grid_frame, text="Base Model").grid(row=0, column=0, sticky="e", pady=5, padx=5)
        models = [
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
            "yolov9t.pt", "yolov9s.pt", "yolov9m.pt", "yolov9c.pt", "yolov9e.pt",
            "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10b.pt", "yolov10l.pt", "yolov10x.pt",
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"
        ]
        ttk.Combobox(grid_frame, values=models, textvariable=self.train_model).grid(row=0, column=1, sticky="w", pady=5, padx=5)

        ttk.Label(grid_frame, text="Epochs").grid(row=1, column=0, sticky="e", pady=5, padx=5)
        ttk.Spinbox(grid_frame, from_=1, to=1000, textvariable=self.train_epochs).grid(row=1, column=1, sticky="w", pady=5, padx=5)

        ttk.Label(grid_frame, text="Batch Size").grid(row=2, column=0, sticky="e", pady=5, padx=5)
        ttk.Spinbox(grid_frame, from_=1, to=128, textvariable=self.train_batch).grid(row=2, column=1, sticky="w", pady=5, padx=5)

        ttk.Label(grid_frame, text="Image Size").grid(row=3, column=0, sticky="e", pady=5, padx=5)
        ttk.Spinbox(grid_frame, from_=256, to=1536, increment=32, textvariable=self.train_imgsz).grid(row=3, column=1, sticky="w", pady=5, padx=5)

        ttk.Label(grid_frame, text="Device").grid(row=4, column=0, sticky="e", pady=5, padx=5)
        ttk.Combobox(grid_frame, values=["auto","cpu","cuda"], textvariable=self.train_device).grid(row=4, column=1, sticky="w", pady=5, padx=5)

        ttk.Button(train_group, text="Start Training", command=self.run_training).pack(pady=(20, 0), fill="x")

    def setup_tools_tab(self):
        self.tab_tools.columnconfigure(0, weight=1)

        # Dataset Tools
        ds_group = ttk.LabelFrame(self.tab_tools, text="Dataset Tools", padding=20)
        ds_group.pack(fill="x", pady=20, padx=20)
        
        ttk.Button(ds_group, text="Split Train/Val (70/30)", command=self.split_dataset_logic).pack(fill="x", pady=(0, 5))
        ttk.Label(ds_group, text="Automatically splits images/labels into train/val folders and generates data.yaml", foreground="gray").pack()

        # Stats
        stat_group = ttk.LabelFrame(self.tab_tools, text="Dataset Statistics", padding=20)
        stat_group.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        ttk.Button(stat_group, text="Refresh Stats", command=self.refresh_stats_logic).pack(anchor="w")
        
        self.stats_label = ttk.Label(stat_group, text="No labels found", anchor="center", font=("Segoe UI", 12))
        self.stats_label.pack(fill="both", expand=True, pady=20)

    def setup_footer(self):
        footer = ttk.Frame(self)
        footer.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        
        # Progress Bars
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(footer, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", pady=(0, 5))
        self.progress_bar.pack_forget()

        self.detect_progress_var = tk.DoubleVar()
        self.detect_progress_bar = ttk.Progressbar(footer, variable=self.detect_progress_var, maximum=100)
        self.detect_progress_bar.pack(fill="x", pady=(0, 5))
        self.detect_progress_bar.pack_forget()

        # Status & ETA
        status_frame = ttk.Frame(footer)
        status_frame.pack(fill="x")
        self.status_label = ttk.Label(status_frame, text="🟢 Ready", foreground="#9FEF9F", font=("Segoe UI", 10, "bold"))
        self.status_label.pack(side="left")
        self.eta_label = ttk.Label(status_frame, text="")
        self.eta_label.pack(side="right")

        # Logs
        log_frame = ttk.LabelFrame(footer, text="Logs", padding=5)
        log_frame.pack(fill="x", pady=(5, 0))
        self.log_text = tk.Text(log_frame, height=6, bg="#1e1e1e", fg="#f0f0f0", borderwidth=0)
        self.log_text.pack(fill="both", expand=True)

    # ---------- Logic ----------
    def browse_video(self):
        file_path = filedialog.askopenfilename(title="Select Video", filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
        if file_path:
            self.video_path.set(file_path)

    def browse_output(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_path.set(folder)

    def open_output_folder(self):
        path = self.output_path.get()
        if path and os.path.exists(path):
            os.startfile(path)
        else:
            messagebox.showwarning("Warning", "Output path does not exist.")

    def add_model(self):
        files = filedialog.askopenfilenames(title="Select ONNX Models", filetypes=[("ONNX Model", "*.onnx")])
        for f in files:
            filename = os.path.basename(f)
            self.models_listbox.insert("end", filename)
            if not hasattr(self, 'model_paths'):
                self.model_paths = {}
            self.model_paths[filename] = f

    def remove_model(self):
        selected = self.models_listbox.curselection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select one or more models to remove.")
            return
        for index in reversed(selected):
            filename = self.models_listbox.get(index)
            self.models_listbox.delete(index)
            if filename in self.model_paths:
                del self.model_paths[filename]

    def update_status(self, text, color):
        self.status_label.config(text=text, foreground=color)
        self.status_label.update_idletasks()

    def update_progress(self, percent):
        try:
            self.progress_var.set(percent)
            self.progress_bar.update_idletasks()
        except:
            pass

    def cancel_detection(self):
        self.cancel_flag = True
        self.update_status("🛑 Canceling...", "#FF7F7F")

    def update_detection_progress(self, percent, eta=None):
        try:
            if percent is not None:
                self.detect_progress_var.set(percent)
                self.detect_progress_bar.update_idletasks()
            if eta:
                self.eta_label.config(text=f"ETA: {eta:.1f}s remaining")
            else:
                self.eta_label.config(text="")
        except:
            pass

    def update_preview(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            # Dynamic resize based on label size
            w = self.preview_image_label.winfo_width()
            h = self.preview_image_label.winfo_height()
            if w > 10 and h > 10:
                img.thumbnail((w, h))
            else:
                img.thumbnail((400, 300))
                
            photo = ImageTk.PhotoImage(img)
            self.preview_image_label.configure(image=photo)
            self.preview_image_label.image = photo
        except:
            pass

    def toggle_pause(self):
        self.pause_flag = not self.pause_flag
        if self.pause_flag:
            self.update_status("⏸️ Paused", "#FFD966")
            self.pause_button.config(text="Resume")
        else:
            self.update_status("🟡 Running...", "#FFD966")
            self.pause_button.config(text="Pause")

    def on_saved(self, img_path, label_path, ann_path):
        self.last_saved_paths = (img_path, label_path, ann_path)

    def delete_last_saved(self):
        paths = self.last_saved_paths
        if not paths:
            return
        for p in paths:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except:
                pass
        self.last_saved_paths = None
        self.update_status("Deleted last entry", "#FFFFFF")

    # ---------- Tools Logic ----------
    def split_dataset_logic(self):
        root = self.output_path.get()
        if not root or not os.path.exists(root):
            messagebox.showerror("Error", "Please select a valid Output Folder first.")
            return

        images_dir = os.path.join(root, "images")
        labels_dir = os.path.join(root, "labels")
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            messagebox.showerror("Error", "Output folder must contain 'images' and 'labels' subfolders.\nRun Detection first to generate data.")
            return

        # Check if already split
        if os.path.exists(os.path.join(root, "train")):
            if not messagebox.askyesno("Warning", "Dataset seems to be already split (train/val folders exist).\nRe-splitting will move files again. Continue?"):
                return

        # Create train/val structure
        for split in ["train", "val"]:
            os.makedirs(os.path.join(root, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(root, split, "labels"), exist_ok=True)

        all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not all_images:
            messagebox.showinfo("Info", "No images found in 'images' folder to split.")
            return

        random.shuffle(all_images)
        split_idx = int(len(all_images) * 0.7)
        train_imgs = all_images[:split_idx]
        val_imgs = all_images[split_idx:]

        def move_files(file_list, split_name):
            moved_count = 0
            for img_name in file_list:
                try:
                    # Move image
                    src_img = os.path.join(images_dir, img_name)
                    dst_img = os.path.join(root, split_name, "images", img_name)
                    shutil.move(src_img, dst_img)
                    
                    # Move corresponding label
                    label_name = os.path.splitext(img_name)[0] + ".txt"
                    src_label = os.path.join(labels_dir, label_name)
                    dst_label = os.path.join(root, split_name, "labels", label_name)
                    if os.path.exists(src_label):
                        shutil.move(src_label, dst_label)
                    moved_count += 1
                except Exception as e:
                    print(f"Error moving {img_name}: {e}")
            return moved_count

        t_count = move_files(train_imgs, "train")
        v_count = move_files(val_imgs, "val")

        # Create data.yaml
        yaml_path = os.path.join(root, "data.yaml")
        data = {
            "path": root,
            "train": "train/images",
            "val": "val/images",
            "names": {0: "object"} # Placeholder, ideally scan classes
        }
        
        # Scan for classes
        classes = set()
        for split in ["train", "val"]:
            lbl_path = os.path.join(root, split, "labels")
            if os.path.exists(lbl_path):
                for f in os.listdir(lbl_path):
                    if f.endswith(".txt"):
                        with open(os.path.join(lbl_path, f), "r") as lf:
                            for line in lf:
                                try:
                                    classes.add(int(line.split()[0]))
                                except: pass
        if classes:
            data["names"] = {c: f"class_{c}" for c in sorted(classes)}

        with open(yaml_path, "w") as f:
            yaml.dump(data, f)

        messagebox.showinfo("Success", f"Split complete!\nTrain: {len(train_imgs)}\nVal: {len(val_imgs)}\ndata.yaml created.")

    def refresh_stats_logic(self):
        root = self.output_path.get()
        if not root or not os.path.exists(root):
            self.stats_label.config(text="Invalid Output Folder")
            return

        total_images = 0
        total_labels = 0
        class_counts = {}

        # Scan root images/labels AND train/val subfolders if they exist
        dirs_to_scan = [root]
        if os.path.exists(os.path.join(root, "train")):
            dirs_to_scan.append(os.path.join(root, "train"))
        if os.path.exists(os.path.join(root, "val")):
            dirs_to_scan.append(os.path.join(root, "val"))

        for d in dirs_to_scan:
            img_d = os.path.join(d, "images")
            lbl_d = os.path.join(d, "labels")
            
            if os.path.exists(img_d):
                total_images += len([f for f in os.listdir(img_d) if f.lower().endswith(('.jpg', '.png'))])
            
            if os.path.exists(lbl_d):
                for f in os.listdir(lbl_d):
                    if f.endswith(".txt"):
                        total_labels += 1
                        with open(os.path.join(lbl_d, f), "r") as lf:
                            for line in lf:
                                try:
                                    c = int(line.split()[0])
                                    class_counts[c] = class_counts.get(c, 0) + 1
                                except: pass

        stats_text = f"Total Images: {total_images}\nTotal Label Files: {total_labels}\n\nClass Distribution:\n"
        if class_counts:
            for c, count in sorted(class_counts.items()):
                stats_text += f"Class {c}: {count}\n"
        else:
            stats_text += "No objects found."
            
        self.stats_label.config(text=stats_text)

    # ---------- Config Management ----------
    def save_config(self):
        model_filenames = list(self.models_listbox.get(0, "end"))
        full_model_paths = [self.model_paths.get(fn, fn) for fn in model_filenames]
        
        config = {
            "video_path": self.video_path.get(),
            "youtube_link": self.youtube_link.get(),
            "yt_res": self.yt_res.get(),
            "models": full_model_paths,
            "output_path": self.output_path.get(),
            "conf_threshold": self.conf_threshold.get(),
            "iou_threshold": self.iou_threshold.get(),
            "frame_step": self.frame_step.get(),
            "merge_iou": self.merge_iou.get(),
            "use_gpu": self.use_gpu.get(),
            "similarity_threshold": self.similarity_threshold.get(),
            "history_size": self.history_size.get(),
            "input_size": self.input_size.get(),
            "target_class_id": self.target_class_id.get(),
            "save_raw": self.save_raw.get(),
            "save_empty": self.save_empty.get(),
            "save_annotated": self.save_annotated.get(),
            "train_model": self.train_model.get(),
            "train_epochs": self.train_epochs.get(),
            "train_batch": self.train_batch.get(),
            "train_imgsz": self.train_imgsz.get(),
            "train_device": self.train_device.get(),
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
        messagebox.showinfo("Saved", "Configuration saved successfully!")

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
            self.video_path.set(config.get("video_path", ""))
            self.youtube_link.set(config.get("youtube_link", ""))
            self.yt_res.set(config.get("yt_res", "1080p"))
            self.out_res.set(config.get("out_res", "off"))
            self.models_listbox.delete(0, "end")
            self.model_paths = {}
            for full_path in config.get("models", []):
                filename = os.path.basename(full_path)
                self.models_listbox.insert("end", filename)
                self.model_paths[filename] = full_path
            self.output_path.set(config.get("output_path", ""))
            self.conf_threshold.set(config.get("conf_threshold", 0.5))
            self.iou_threshold.set(config.get("iou_threshold", 0.45))
            self.frame_step.set(config.get("frame_step", 1))
            self.merge_iou.set(config.get("merge_iou", 0.6))
            self.use_gpu.set(config.get("use_gpu", True))
            self.similarity_threshold.set(config.get("similarity_threshold", 5))
            self.history_size.set(config.get("history_size", 30))
            self.input_size.set(config.get("input_size", 640))
            self.target_class_id.set(config.get("target_class_id", -1))
            self.save_raw.set(config.get("save_raw", True))
            self.save_empty.set(config.get("save_empty", False))
            self.save_annotated.set(config.get("save_annotated", True))
            self.train_model.set(config.get("train_model", "yolov8n.pt"))
            self.train_epochs.set(config.get("train_epochs", 50))
            self.train_batch.set(config.get("train_batch", 16))
            self.train_imgsz.set(config.get("train_imgsz", 640))
            self.train_device.set(config.get("train_device", "auto"))

    class TextRedirector:
        """Redirects print() output to a Tkinter Text widget in real time."""
        def __init__(self, text_widget):
            self.text_widget = text_widget

        def write(self, message):
            self.text_widget.insert("end", message)
            self.text_widget.see("end")
            self.text_widget.update_idletasks()

        def flush(self):
            pass

    def run_detection(self):
        # 1. Validation
        if not self.validate_detection_inputs():
            return

        model_filenames = list(self.models_listbox.get(0, "end"))
        models = [self.model_paths.get(fn, fn) for fn in model_filenames]
        
        self.cancel_flag = False
        self.detect_progress_bar.pack(fill="x", pady=(0, 5))
        self.progress_bar.pack(fill="x", pady=(0, 5))
        self.detect_progress_var.set(0)
        self.eta_label.config(text="")
        self.cancel_button.pack(fill="x")
        self.run_button_state(False)
        self.update_status("🟡 Running...", "#FFD966")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("end", "Starting detection...\n")
        self.log_text.see("end")

        # Prepare arguments
        args = {
            "video": self.video_path.get() or None,
            "youtube": self.youtube_link.get() or None,
            "yt_res": self.yt_res.get(),
            "out_res": self.out_res.get() if self.out_res.get().lower() != "off" else None,
            "models": models,
            "out": self.output_path.get() or "./output",
            "conf": self.conf_threshold.get(),
            "iou": self.iou_threshold.get(),
            "frame_step": self.frame_step.get(),
            "merge_iou": self.merge_iou.get(),
            "cpu": not self.use_gpu.get(),
            "similarity_threshold": self.similarity_threshold.get(),
            "history_size": self.history_size.get(),
            "input_size": self.input_size.get(),
            "target_class_id": self.target_class_id.get(),
            "save_raw": self.save_raw.get(),
            "save_empty": self.save_empty.get(),
            "save_annotated": self.save_annotated.get(),
        }

        threading.Thread(target=self.run_internal_main, args=(args,), daemon=True).start()

    def validate_detection_inputs(self):
        # Check models
        if self.models_listbox.size() == 0:
            messagebox.showerror("Error", "Please add at least one ONNX model in Settings.")
            return False
            
        # Check input (Video or YouTube)
        video = self.video_path.get()
        youtube = self.youtube_link.get()
        
        if not video and not youtube:
            messagebox.showerror("Error", "Please select a Video File or enter a YouTube Link.")
            return False
            
        if video and not os.path.exists(video):
            messagebox.showerror("Error", "Selected video file does not exist.")
            return False
            
        if youtube and "youtube.com" not in youtube and "youtu.be" not in youtube:
             messagebox.showerror("Error", "Invalid YouTube link.")
             return False

        # Check Output
        out = self.output_path.get()
        if not out:
             messagebox.showerror("Error", "Please select an Output Folder in Settings.")
             return False
             
        return True

    def yt_progress_callback(self, text=None, percent=None):
        if text:
            self.log_text.insert("end", text + "\n")
            self.log_text.see("end")

        if percent is not None:
            self.progress_bar.pack(fill="x", pady=(0, 5))
            self.update_progress(percent)

        if text and "Download complete" in text:
            self.progress_var.set(100)

    def run_internal_main(self, args):
        self.log_text.delete("1.0", "end")
        self.log_text.insert("end", "Starting detection...\n\n")

        try:
            import sys
            import Main

            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = self.TextRedirector(self.log_text)

            self.update_status("🟡 Running...", "#FFD966")

            Main.main(
                args=args,
                progress_callback=self.yt_progress_callback,
                detect_progress_callback=self.update_detection_progress,
                cancel_check=lambda: self.cancel_flag,
                preview_callback=self.update_preview,
                pause_check=lambda: self.pause_flag,
                save_callback=self.on_saved
            )

            self.update_status("🟢 Done", "#9FEF9F")

        except Exception as e:
            print(f"\n[ERROR] {e}")
            self.update_status("🔴 Error", "#FF7F7F")

        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            self.run_button_state(True)
            print("\nProcess finished.\n")

    def run_button_state(self, enabled: bool):
        if enabled:
            self.run_btn.config(state="normal")
            self.cancel_button.pack_forget()
        else:
            self.run_btn.config(state="disabled")

    def scan_classes(self, labels_dir):
        classes = set()
        if os.path.isdir(labels_dir):
            for name in os.listdir(labels_dir):
                if name.endswith(".txt"):
                    try:
                        with open(os.path.join(labels_dir, name), "r") as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    classes.add(int(float(parts[0])))
                    except:
                        pass
        if not classes:
            return ["0"]
        m = max(classes)
        return [str(i) for i in range(m + 1)]

    def build_dataset_yaml(self, root):
        names = self.scan_classes(os.path.join(root, "labels"))
        data = {
            "path": root,
            "train": "images",
            "val": "images",
            "nc": len(names),
            "names": names,
        }
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        with open(tmp.name, "w") as f:
            yaml.safe_dump(data, f)
        return tmp.name

    def run_training(self):
        root = self.output_path.get() or ""
        # Check if train/val split exists
        if os.path.isdir(os.path.join(root, "train")) and os.path.isdir(os.path.join(root, "val")):
            yaml_path = os.path.join(root, "data.yaml")
            if not os.path.exists(yaml_path):
                # Fallback build
                yaml_path = self.build_dataset_yaml(root)
        elif os.path.isdir(os.path.join(root, "images")) and os.path.isdir(os.path.join(root, "labels")):
             yaml_path = self.build_dataset_yaml(root)
        else:
            messagebox.showerror("Error", "Select an output folder with images/labels or train/val structure")
            return
            
        weights = self.train_model.get()
        epochs = self.train_epochs.get()
        imgsz = self.train_imgsz.get()
        batch = self.train_batch.get()
        device = self.train_device.get()
        threading.Thread(target=self.training_worker, args=(yaml_path, weights, epochs, imgsz, batch, device), daemon=True).start()

    def training_worker(self, yaml_path, weights, epochs, imgsz, batch, device):
        self.update_status("🟡 Training...", "#FFD966")
        try:
            from ultralytics import YOLO
        except Exception as e:
            messagebox.showerror("Missing Dependency", "Install ultralytics and torch to train:\n\npip install ultralytics\npip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
            self.update_status("🔴 Error", "#FF7F7F")
            return
        try:
            model = YOLO(weights)
            dev = None if device == "auto" else device
            model.train(data=yaml_path, epochs=int(epochs), imgsz=int(imgsz), batch=int(batch), device=dev)
            
            self.update_status("� Exporting ONNX...", "#FFD966")
            model.export(format="onnx")

            self.update_status("�🟢 Trained & Exported", "#9FEF9F")
        except Exception as e:
            messagebox.showerror("Training Failed", str(e))
            self.update_status("🔴 Error", "#FF7F7F")

if __name__ == "__main__":
    app = App()
    app.mainloop()
