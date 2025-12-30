import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import os
import threading

class NullGeneratorTab(ttk.Frame):
    def __init__(self, parent, app_instance=None):
        super().__init__(parent)
        self.app = app_instance
        self.pack(fill="both", expand=True)
        
        # Variables
        self.image_folder = tk.StringVar()
        self.label_folder = tk.StringVar()
        self.output_folder = tk.StringVar(value=os.path.join(os.getcwd(), "null_generator_output"))
        self.expansion_factor = tk.DoubleVar(value=0.25)
        self.radius = tk.IntVar(value=5)
        self.soft_mask = tk.BooleanVar(value=False)
        self.convert_format = tk.StringVar(value="None")
        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="Ready")
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main layout container
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill="both", expand=True)
        
        # Split into two columns
        left_col = ttk.Frame(main_frame)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        right_col = ttk.Frame(main_frame)
        right_col.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        # --- Left Column: Inputs ---
        input_group = ttk.LabelFrame(left_col, text="Input & Output Paths", padding=15)
        input_group.pack(fill="x", pady=(0, 15))
        
        # Image Folder
        ttk.Label(input_group, text="Image Folder:").pack(anchor="w", pady=(0, 5))
        img_frame = ttk.Frame(input_group)
        img_frame.pack(fill="x", pady=(0, 10))
        ttk.Entry(img_frame, textvariable=self.image_folder).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(img_frame, text="Browse", command=self.browse_images).pack(side="right")
        
        # Label Folder
        ttk.Label(input_group, text="Label Folder:").pack(anchor="w", pady=(0, 5))
        lbl_frame = ttk.Frame(input_group)
        lbl_frame.pack(fill="x", pady=(0, 10))
        ttk.Entry(lbl_frame, textvariable=self.label_folder).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(lbl_frame, text="Browse", command=self.browse_labels).pack(side="right")

        # Output Folder
        ttk.Label(input_group, text="Output Folder:").pack(anchor="w", pady=(0, 5))
        out_frame = ttk.Frame(input_group)
        out_frame.pack(fill="x", pady=(0, 10))
        ttk.Entry(out_frame, textvariable=self.output_folder).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(out_frame, text="Browse", command=self.browse_output).pack(side="right")

        # --- Right Column: Settings ---
        settings_group = ttk.LabelFrame(right_col, text="Processing Parameters", padding=15)
        settings_group.pack(fill="x", pady=(0, 15))
        
        # Expansion Factor
        ttk.Label(settings_group, text="Mask Expansion Factor (0.0 - 1.0):").pack(anchor="w", pady=(0, 5))
        exp_frame = ttk.Frame(settings_group)
        exp_frame.pack(fill="x", pady=(0, 10))
        self.exp_scale = ttk.Scale(exp_frame, from_=0.0, to=1.0, variable=self.expansion_factor)
        self.exp_scale.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.exp_lbl = ttk.Label(exp_frame, text=f"{self.expansion_factor.get():.2f}", width=5)
        self.exp_lbl.pack(side="right")
        self.exp_scale.configure(command=lambda v: self.exp_lbl.config(text=f"{float(v):.2f}"))
        
        # Inpaint Radius
        ttk.Label(settings_group, text="Inpaint Radius (1 - 20 px):").pack(anchor="w", pady=(0, 5))
        rad_frame = ttk.Frame(settings_group)
        rad_frame.pack(fill="x", pady=(0, 10))
        self.rad_scale = ttk.Scale(rad_frame, from_=1, to=20, variable=self.radius)
        self.rad_scale.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.rad_lbl = ttk.Label(rad_frame, text=f"{self.radius.get():d}", width=5)
        self.rad_lbl.pack(side="right")
        self.rad_scale.configure(command=lambda v: self.rad_lbl.config(text=f"{int(float(v))}"))
        
        # Additional Options
        opts_group = ttk.LabelFrame(right_col, text="Advanced Options", padding=15)
        opts_group.pack(fill="x")
        
        ttk.Checkbutton(opts_group, text="Use Soft Mask (Gaussian Blur)", variable=self.soft_mask).pack(anchor="w", pady=(0, 10))
        
        ttk.Label(opts_group, text="Convert Output Format:").pack(anchor="w", pady=(0, 5))
        ttk.Combobox(opts_group, textvariable=self.convert_format, values=["None", "PNG", "JPEG", "JPG"], state="readonly").pack(fill="x")
        
        # --- Bottom Area: Action & Status ---
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(side="bottom", fill="x", pady=(20, 0))
        
        self.btn_start = ttk.Button(action_frame, text="Start Null Generation", command=self.start_processing_thread)
        self.btn_start.pack(fill="x", ipady=10, pady=(0, 10))
        
        status_frame = ttk.Frame(action_frame)
        status_frame.pack(fill="x")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side="left")
        ttk.Label(status_frame, text="Progress:").pack(side="right")
        
        ttk.Progressbar(action_frame, variable=self.progress_var, maximum=100).pack(fill="x", pady=(5, 0))

    def browse_images(self):
        path = filedialog.askdirectory(title="Select Image Folder")
        if path: self.image_folder.set(path)

    def browse_labels(self):
        path = filedialog.askdirectory(title="Select Label Folder")
        if path: self.label_folder.set(path)

    def browse_output(self):
        path = filedialog.askdirectory(title="Select Output Folder")
        if path: self.output_folder.set(path)

    def create_mask_from_yolo(self, image, label_path, expansion_factor):
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        try:
            if not os.path.exists(label_path): return mask
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                data = line.strip().split()
                if len(data) >= 5:
                    x_center = float(data[1]) * width
                    y_center = float(data[2]) * height
                    box_width = float(data[3]) * width
                    box_height = float(data[4]) * height
                    
                    expanded_width = box_width * (1 + expansion_factor)
                    expanded_height = box_height * (1 + expansion_factor)
                    
                    x1 = max(0, int(x_center - expanded_width/2))
                    y1 = max(0, int(y_center - expanded_height/2))
                    x2 = min(width - 1, int(x_center + expanded_width/2))
                    y2 = min(height - 1, int(y_center + expanded_height/2))
                    
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            return mask
        except Exception as e:
            print(f"Error processing annotations: {str(e)}")
            return np.zeros((height, width), dtype=np.uint8)

    def start_processing_thread(self):
        threading.Thread(target=self.process, daemon=True).start()

    def process(self):
        img_dir = self.image_folder.get()
        lbl_dir = self.label_folder.get()
        out_dir = self.output_folder.get()
        
        if not img_dir or not lbl_dir or not out_dir:
            messagebox.showerror("Error", "Please select all folders.")
            return

        self.btn_start.config(state="disabled")
        self.status_var.set("Processing...")
        self.progress_var.set(0)
        
        try:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total = len(image_files)
            
            for i, img_file in enumerate(image_files):
                img_path = os.path.join(img_dir, img_file)
                base_name = os.path.splitext(img_file)[0]
                label_path = os.path.join(lbl_dir, base_name + ".txt")
                
                # Check if label exists, if not, just copy image or skip? 
                # If the goal is "null generator" (remove objects), if no objects, image remains same.
                
                image = cv2.imread(img_path)
                if image is None: continue
                
                if os.path.exists(label_path):
                    mask = self.create_mask_from_yolo(image, label_path, self.expansion_factor.get())
                    
                    if self.soft_mask.get():
                        mask = cv2.GaussianBlur(mask, (7,7), 0)
                    
                    radius = int(self.radius.get())
                    result = cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)
                else:
                    result = image # No labels, no inpainting
                
                # Save
                ext = os.path.splitext(img_file)[1].lower()
                target_fmt = self.convert_format.get()
                if target_fmt == "JPEG" or target_fmt == "JPG":
                    ext = ".jpg"
                elif target_fmt == "PNG":
                    ext = ".png"
                    
                save_path = os.path.join(out_dir, base_name + ext)
                cv2.imwrite(save_path, result)
                
                self.progress_var.set(((i + 1) / total) * 100)
                self.status_var.set(f"Processed {i+1}/{total}")
                
            messagebox.showinfo("Done", "Null Generation Complete!")
            self.status_var.set("Complete")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            self.status_var.set("Error")
        finally:
            self.btn_start.config(state="normal")
