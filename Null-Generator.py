import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import os
import threading

# ----------------- YOLO Mask Creation -----------------
def create_mask_from_yolo(image, label_path, expansion_factor):
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    try:
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

# ----------------- Inpainting -----------------
def inpaint_from_yolo(image_path, label_path, expansion_factor=0.25, radius=3, method=cv2.INPAINT_TELEA, soft_mask=False):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    mask = create_mask_from_yolo(image, label_path, expansion_factor)
    
    if soft_mask:
        mask = cv2.GaussianBlur(mask, (7,7), 0)  # Soft edges
    
    result = cv2.inpaint(image, mask, radius, method)
    return image, mask, result

# ----------------- GUI Functions -----------------
def load_images():
    folder_selected = filedialog.askdirectory(title="Select Image Folder")
    if folder_selected:
        image_paths = [os.path.join(folder_selected, f) for f in os.listdir(folder_selected) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_listbox.delete(0, tk.END)
        for path in image_paths:
            image_listbox.insert(tk.END, path)

def load_labels():
    folder_selected = filedialog.askdirectory(title="Select Label Folder")
    if folder_selected:
        label_files = [os.path.join(folder_selected, f) for f in os.listdir(folder_selected) if f.endswith('.txt')]
        label_file_listbox.delete(0, tk.END)
        for path in label_files:
            label_file_listbox.insert(tk.END, path)

# ----------------- Processing -----------------
def start_processing_thread():
    thread = threading.Thread(target=start_processing)
    thread.start()

def start_processing():
    try:
        image_paths = image_listbox.get(0, tk.END)
        label_files = label_file_listbox.get(0, tk.END)

        if not image_paths or not label_files:
            messagebox.showerror("Missing Information", "Please load both images and labels first.")
            return

        output_dir = "processed_output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        label_dict = {os.path.splitext(os.path.basename(label_path))[0]: label_path for label_path in label_files}

        total_images = len(image_paths)
        for idx, image_path in enumerate(image_paths):
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            ext = os.path.splitext(image_path)[1].lower()
            label_path = label_dict.get(image_name)

            if label_path:
                try:
                    _, _, inpainted = inpaint_from_yolo(
                        image_path, label_path,
                        expansion_factor=expansion_scale.get(),
                        radius=radius_scale.get(),
                        method=cv2.INPAINT_TELEA,
                        soft_mask=soft_mask_var.get()
                    )

                    # Format conversion
                    if convert_var.get() != "None":
                        target_format = convert_var.get()
                        if target_format in ["JPEG", "JPG"]:
                            save_ext = ".jpg"
                        elif target_format == "PNG":
                            save_ext = ".png"
                        save_path = os.path.join(output_dir, f"{image_name}_processed{save_ext}")
                    else:
                        save_path = os.path.join(output_dir, f"{image_name}_processed{ext}")

                    cv2.imwrite(save_path, inpainted)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

            # Update progress bar
            progress_percent = ((idx + 1) / total_images) * 100
            progress_var.set(progress_percent)

        messagebox.showinfo("Processing Complete", f"{total_images} images processed.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# ----------------- Tkinter GUI -----------------
root = tk.Tk()
root.title("YOLO Image Inpainting Tool - Dark Theme")

# ----------------- Dark Theme -----------------
bg_color = "#2E2E2E"
fg_color = "#FFFFFF"
button_color = "#4B4B4B"
listbox_bg = "#3C3C3C"
highlight_color = "#5A5A5A"
accent_color = "#00BFFF"

root.configure(bg=bg_color)

style = ttk.Style()
style.theme_use('default')
style.configure("TProgressbar", troughcolor=listbox_bg, background=accent_color, bordercolor=listbox_bg,
                lightcolor=accent_color, darkcolor=accent_color)
style.configure("TCombobox", fieldbackground=listbox_bg, background=listbox_bg, foreground=fg_color)

# Image listbox
tk.Label(root, text="Loaded Images:", bg=bg_color, fg=fg_color).grid(row=0, column=0, padx=10, pady=5)
image_listbox = tk.Listbox(root, height=10, width=50, bg=listbox_bg, fg=fg_color, selectbackground=highlight_color)
image_listbox.grid(row=1, column=0, padx=10, pady=5)
tk.Button(root, text="Load Images", command=load_images, bg=button_color, fg=fg_color).grid(row=2, column=0, padx=10, pady=5)

# Label listbox
tk.Label(root, text="Loaded Label Files:", bg=bg_color, fg=fg_color).grid(row=0, column=1, padx=10, pady=5)
label_file_listbox = tk.Listbox(root, height=10, width=50, bg=listbox_bg, fg=fg_color, selectbackground=highlight_color)
label_file_listbox.grid(row=1, column=1, padx=10, pady=5)
tk.Button(root, text="Load Labels", command=load_labels, bg=button_color, fg=fg_color).grid(row=2, column=1, padx=10, pady=5)

# Format conversion
convert_var = tk.StringVar(value="None")
tk.Label(root, text="Convert Format:", bg=bg_color, fg=fg_color).grid(row=3, column=0, padx=10, pady=5)
format_options = ttk.Combobox(root, textvariable=convert_var, values=["None", "PNG", "JPEG", "JPG"], state="readonly", width=47)
format_options.grid(row=3, column=1, padx=10, pady=5)

# Expansion factor slider
tk.Label(root, text="Expansion Factor:", bg=bg_color, fg=fg_color).grid(row=4, column=0, padx=10, pady=5)
expansion_scale = tk.DoubleVar(value=0.25)
tk.Scale(root, from_=0.01, to=1.0, resolution=0.01, variable=expansion_scale, orient=tk.HORIZONTAL, length=300, bg=bg_color, fg=fg_color, troughcolor=listbox_bg, highlightbackground=bg_color).grid(row=4, column=1, padx=10, pady=5)

# Inpainting radius slider
tk.Label(root, text="Inpainting Radius:", bg=bg_color, fg=fg_color).grid(row=5, column=0, padx=10, pady=5)
radius_scale = tk.IntVar(value=5)
tk.Scale(root, from_=0.01, to=1.0, variable=radius_scale, orient=tk.HORIZONTAL, length=300, bg=bg_color, fg=fg_color, troughcolor=listbox_bg, highlightbackground=bg_color).grid(row=5, column=1, padx=10, pady=5)

# Soft mask checkbox
soft_mask_var = tk.BooleanVar(value=False)
tk.Checkbutton(root, text="Use Soft Mask (Gaussian Blur)", variable=soft_mask_var, bg=bg_color, fg=fg_color, selectcolor=bg_color, activebackground=bg_color).grid(row=6, column=0, columnspan=2, padx=10, pady=5)

# Start button
tk.Button(root, text="Start Processing", command=start_processing_thread, bg=button_color, fg=fg_color).grid(row=7, column=0, columnspan=2, padx=10, pady=10)

# Progress bar
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
progress_bar.grid(row=8, column=0, columnspan=2, padx=10, pady=10, sticky="we")

root.mainloop()
