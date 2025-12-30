import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import os

class LabelEditor(ttk.Frame):
    def __init__(self, parent, app_instance=None):
        super().__init__(parent)
        self.app = app_instance
        self.pack(fill="both", expand=True)
        
        self.current_image_path = None
        self.current_label_path = None
        self.image_list = []
        self.current_index = -1
        
        self.raw_image = None   # PIL Image
        self.tk_image = None    # ImageTk
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Labels: list of [class_id, x_center, y_center, w, h] (normalized)
        self.labels = []
        
        # Drawing state
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        self.selected_box_index = -1
        
        # Interaction state
        self.interaction_mode = 'idle' # idle, draw, move, resize
        self.active_handle = None
        self.drag_start_pos = None
        self.initial_box_state = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Layout: Left Sidebar (File List), Center (Canvas), Top (Toolbar)
        
        # Main PanedWindow
        self.paned = ttk.PanedWindow(self, orient="horizontal")
        self.paned.pack(fill="both", expand=True)
        
        # --- Left Sidebar ---
        self.sidebar = ttk.Frame(self.paned, width=200)
        self.paned.add(self.sidebar, weight=0)
        
        ttk.Label(self.sidebar, text="Images").pack(pady=5)
        
        self.file_listbox = tk.Listbox(self.sidebar, selectmode="single", bg="#1e1e1e", fg="#ffffff", highlightthickness=0, borderwidth=0)
        self.file_listbox.pack(fill="both", expand=True, padx=5, pady=5)
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_select)
        
        # --- Right Area ---
        self.right_frame = ttk.Frame(self.paned)
        self.paned.add(self.right_frame, weight=3)
        
        # Toolbar
        self.toolbar = ttk.Frame(self.right_frame)
        self.toolbar.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(self.toolbar, text="Class ID:").pack(side="left")
        self.class_var = tk.StringVar(value="0")
        self.class_entry = ttk.Entry(self.toolbar, textvariable=self.class_var, width=10)
        self.class_entry.pack(side="left", padx=5)
        
        ttk.Button(self.toolbar, text="Save (Ctrl+S)", command=self.save_labels).pack(side="left", padx=10)
        ttk.Button(self.toolbar, text="Delete Box (Del)", command=self.delete_selected).pack(side="left", padx=5)
        ttk.Button(self.toolbar, text="Delete Image", command=self.delete_current_image).pack(side="left", padx=5)
        ttk.Button(self.toolbar, text="Refresh", command=self.refresh_file_list).pack(side="left", padx=5)
        
        self.lbl_status = ttk.Label(self.toolbar, text="No image loaded")
        self.lbl_status.pack(side="right", padx=10)

        # Canvas Container (for centering)
        self.canvas_container = ttk.Frame(self.right_frame)
        self.canvas_container.pack(fill="both", expand=True)
        
        self.canvas = tk.Canvas(self.canvas_container, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        # Events
        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        # Right click to select/delete? Let's use left click to select, Del key to delete
        self.canvas.bind("<ButtonPress-3>", self.on_right_click) # Right click delete
        
        # Key bindings (global to app usually, but we bind to canvas focus)
        self.canvas.bind("<Delete>", lambda e: self.delete_selected())
        self.canvas.bind("<Control-s>", lambda e: self.save_labels())
        
        # Focus canvas on enter
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())

    def refresh_file_list(self):
        if not self.app: return
        
        output_path = self.app.output_path.get()
        if not output_path or not os.path.exists(output_path):
            self.lbl_status.config(text="Output path invalid")
            return
            
        img_dir = os.path.join(output_path, "images")
        if not os.path.exists(img_dir):
            self.lbl_status.config(text="No 'images' folder found")
            return
            
        self.image_list = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        self.file_listbox.delete(0, "end")
        for f in self.image_list:
            self.file_listbox.insert("end", f)
            
        self.lbl_status.config(text=f"Found {len(self.image_list)} images")

    def on_file_select(self, event):
        sel = self.file_listbox.curselection()
        if not sel: return
        
        idx = sel[0]
        self.load_image_by_index(idx)

    def load_image_by_index(self, index):
        if index < 0 or index >= len(self.image_list): return
        
        self.current_index = index
        filename = self.image_list[index]
        output_path = self.app.output_path.get()
        
        self.current_image_path = os.path.join(output_path, "images", filename)
        
        # Load Labels
        label_name = os.path.splitext(filename)[0] + ".txt"
        self.current_label_path = os.path.join(output_path, "labels", label_name)
        
        self.load_labels()
        self.load_image_file()
        self.redraw()

    def load_labels(self):
        self.labels = []
        if os.path.exists(self.current_label_path):
            with open(self.current_label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = parts[0]
                        cx = float(parts[1])
                        cy = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        self.labels.append([cls, cx, cy, w, h])

    def save_labels(self):
        if not self.current_label_path: return
        
        # Ensure labels dir exists
        os.makedirs(os.path.dirname(self.current_label_path), exist_ok=True)
        
        with open(self.current_label_path, "w") as f:
            for lbl in self.labels:
                f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")
        
        self.lbl_status.config(text="Saved!")
        self.after(2000, lambda: self.lbl_status.config(text=""))

    def load_image_file(self):
        try:
            self.raw_image = Image.open(self.current_image_path)
        except Exception as e:
            print(f"Error loading image: {e}")
            self.raw_image = None

    def on_resize(self, event):
        if self.raw_image:
            self.redraw()

    def redraw(self):
        if not self.raw_image: return
        
        # Calculate scale to fit canvas
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        iw, ih = self.raw_image.size
        
        if cw <= 1 or ch <= 1: return # Not ready
        
        # Scale to fit
        scale_w = cw / iw
        scale_h = ch / ih
        self.scale_factor = min(scale_w, scale_h, 1.0) # Don't upscale if smaller? Actually better to just fit
        self.scale_factor = min(scale_w, scale_h) * 0.95 # Leave some margin
        
        nw = int(iw * self.scale_factor)
        nh = int(ih * self.scale_factor)
        
        self.offset_x = (cw - nw) // 2
        self.offset_y = (ch - nh) // 2
        
        resized = self.raw_image.resize((nw, nh), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)
        
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_image)
        
        # Draw labels
        self.draw_boxes()

    def get_box_pixel_coords(self, index):
        if index < 0 or index >= len(self.labels): return None
        cls, cx, cy, w, h = self.labels[index]
        iw, ih = self.raw_image.size
        
        img_x1 = (cx - w/2) * iw
        img_y1 = (cy - h/2) * ih
        img_x2 = (cx + w/2) * iw
        img_y2 = (cy + h/2) * ih
        
        x1 = img_x1 * self.scale_factor + self.offset_x
        y1 = img_y1 * self.scale_factor + self.offset_y
        x2 = img_x2 * self.scale_factor + self.offset_x
        y2 = img_y2 * self.scale_factor + self.offset_y
        
        return x1, y1, x2, y2

    def draw_boxes(self):
        self.canvas.delete("box")
        self.canvas.delete("handle")
        if not self.raw_image: return
        
        for i, (cls, cx, cy, w, h) in enumerate(self.labels):
            coords = self.get_box_pixel_coords(i)
            if not coords: continue
            x1, y1, x2, y2 = coords
            
            color = "#00ff00" if i != self.selected_box_index else "#ffff00"
            width = 2 if i != self.selected_box_index else 3
            
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width, tags=("box", f"box_{i}"))
            self.canvas.create_text(x1, y1-10, text=f"{cls}", fill=color, anchor="sw", tags=("box", f"txt_{i}"))
            
            # Draw handles if selected
            if i == self.selected_box_index:
                self.draw_handles(x1, y1, x2, y2)

    def draw_handles(self, x1, y1, x2, y2):
        handles = self.get_handle_coords(x1, y1, x2, y2)
        for name, (hx1, hy1, hx2, hy2) in handles.items():
            self.canvas.create_rectangle(hx1, hy1, hx2, hy2, fill="cyan", outline="black", tags=("handle", f"handle_{name}"))

    def get_handle_coords(self, x1, y1, x2, y2):
        s = 8 # handle size
        hw = s/2
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2
        
        return {
            'nw': (x1-hw, y1-hw, x1+hw, y1+hw),
            'n' : (xm-hw, y1-hw, xm+hw, y1+hw),
            'ne': (x2-hw, y1-hw, x2+hw, y1+hw),
            'e' : (x2-hw, ym-hw, x2+hw, ym+hw),
            'se': (x2-hw, y2-hw, x2+hw, y2+hw),
            's' : (xm-hw, y2-hw, xm+hw, y2+hw),
            'sw': (x1-hw, y2-hw, x1+hw, y2+hw),
            'w' : (x1-hw, ym-hw, x1+hw, ym+hw)
        }

    def get_handle_at(self, x, y):
        if self.selected_box_index == -1: return None
        coords = self.get_box_pixel_coords(self.selected_box_index)
        if not coords: return None
        x1, y1, x2, y2 = coords
        
        handles = self.get_handle_coords(x1, y1, x2, y2)
        
        for name, (hx1, hy1, hx2, hy2) in handles.items():
            if hx1 <= x <= hx2 and hy1 <= y <= hy2:
                return name
        return None

    def get_canvas_coords(self, event):
        return event.x, event.y

    def on_mouse_down(self, event):
        if not self.raw_image: return
        x, y = self.get_canvas_coords(event)
        
        # 1. Check handles of selected box
        handle = self.get_handle_at(x, y)
        if handle:
            self.interaction_mode = 'resize'
            self.active_handle = handle
            self.drag_start_pos = (x, y)
            self.initial_box_state = list(self.labels[self.selected_box_index])
            return

        # 2. Check if inside a box
        clicked_box = -1
        # Iterate reverse
        for i in range(len(self.labels)-1, -1, -1):
            coords = self.get_box_pixel_coords(i)
            if not coords: continue
            x1, y1, x2, y2 = coords
            
            if x1 <= x <= x2 and y1 <= y <= y2:
                clicked_box = i
                break
        
        if clicked_box != -1:
            self.selected_box_index = clicked_box
            self.class_var.set(str(self.labels[clicked_box][0]))
            self.draw_boxes()
            
            self.interaction_mode = 'move'
            self.drag_start_pos = (x, y)
            self.initial_box_state = list(self.labels[clicked_box])
            return
        
        # 3. Background -> Start Drawing
        self.selected_box_index = -1
        self.draw_boxes()
        self.interaction_mode = 'draw'
        self.start_x = x
        self.start_y = y
        self.current_rect = self.canvas.create_rectangle(x, y, x, y, outline="cyan", width=2)

    def on_mouse_drag(self, event):
        x, y = self.get_canvas_coords(event)
        
        if self.interaction_mode == 'draw':
            if self.start_x is not None:
                self.canvas.coords(self.current_rect, self.start_x, self.start_y, x, y)
            
        elif self.interaction_mode == 'move':
            if not self.drag_start_pos: return
            dx = x - self.drag_start_pos[0]
            dy = y - self.drag_start_pos[1]
            
            # Convert px delta to normalized delta
            iw, ih = self.raw_image.size
            norm_dx = dx / self.scale_factor / iw
            norm_dy = dy / self.scale_factor / ih
            
            cls, cx, cy, w, h = self.initial_box_state
            self.labels[self.selected_box_index] = [cls, cx + norm_dx, cy + norm_dy, w, h]
            self.draw_boxes()
            
        elif self.interaction_mode == 'resize':
            if not self.drag_start_pos: return
            
            iw, ih = self.raw_image.size
            cls, cx, cy, w, h = self.initial_box_state
            
            # Initial pixel coords (relative to image)
            img_x1 = (cx - w/2) * iw
            img_y1 = (cy - h/2) * ih
            img_x2 = (cx + w/2) * iw
            img_y2 = (cy + h/2) * ih
            
            # Delta in image pixels
            dx = (x - self.drag_start_pos[0]) / self.scale_factor
            dy = (y - self.drag_start_pos[1]) / self.scale_factor
            
            # Update edges based on handle
            if 'w' in self.active_handle: img_x1 += dx
            if 'e' in self.active_handle: img_x2 += dx
            if 'n' in self.active_handle: img_y1 += dy
            if 's' in self.active_handle: img_y2 += dy
            
            # Recalculate normalized
            # Don't clamp here, allow free resize until mouse up or just clamp visually
            # Actually better to handle min size here to avoid crash
            if img_x2 < img_x1: img_x2 = img_x1 + 1
            if img_y2 < img_y1: img_y2 = img_y1 + 1
            
            new_w = (img_x2 - img_x1)
            new_h = (img_y2 - img_y1)
            new_cx = img_x1 + new_w/2
            new_cy = img_y1 + new_h/2
            
            self.labels[self.selected_box_index] = [cls, new_cx/iw, new_cy/ih, new_w/iw, new_h/ih]
            self.draw_boxes()

    def on_mouse_up(self, event):
        if self.interaction_mode == 'draw':
            if self.start_x is not None:
                end_x, end_y = self.get_canvas_coords(event)
                
                x1 = min(self.start_x, end_x) - self.offset_x
                y1 = min(self.start_y, end_y) - self.offset_y
                x2 = max(self.start_x, end_x) - self.offset_x
                y2 = max(self.start_y, end_y) - self.offset_y
                
                self.canvas.delete(self.current_rect)
                
                if (x2 - x1) >= 5 and (y2 - y1) >= 5:
                    iw, ih = self.raw_image.size
                    x1 = max(0, min(x1, iw * self.scale_factor))
                    y1 = max(0, min(y1, ih * self.scale_factor))
                    x2 = max(0, min(x2, iw * self.scale_factor))
                    y2 = max(0, min(y2, ih * self.scale_factor))
                    
                    raw_x1 = x1 / self.scale_factor
                    raw_y1 = y1 / self.scale_factor
                    raw_x2 = x2 / self.scale_factor
                    raw_y2 = y2 / self.scale_factor
                    
                    w = raw_x2 - raw_x1
                    h = raw_y2 - raw_y1
                    cx = raw_x1 + w/2
                    cy = raw_y1 + h/2
                    
                    cls_id = self.class_var.get() or "0"
                    self.labels.append([cls_id, cx/iw, cy/ih, w/iw, h/ih])
                    self.selected_box_index = len(self.labels) - 1
            
            self.start_x = None
            self.draw_boxes()
            
        self.interaction_mode = 'idle'
        self.active_handle = None
        self.drag_start_pos = None

    def delete_selected(self):
        if self.selected_box_index != -1:
            del self.labels[self.selected_box_index]
            self.selected_box_index = -1
            self.draw_boxes()

    def delete_current_image(self):
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            return

        if not messagebox.askyesno("Confirm Delete", "Are you sure you want to delete this image and its labels?"):
            return

        try:
            # Delete Image
            os.remove(self.current_image_path)
            
            # Delete Label if exists
            if self.current_label_path and os.path.exists(self.current_label_path):
                os.remove(self.current_label_path)
            
            # Update List
            del self.image_list[self.current_index]
            self.file_listbox.delete(self.current_index)
            
            # Load next or previous
            if self.current_index >= len(self.image_list):
                self.current_index = len(self.image_list) - 1
            
            if self.current_index >= 0:
                self.file_listbox.selection_clear(0, "end")
                self.file_listbox.selection_set(self.current_index)
                self.load_image_by_index(self.current_index)
            else:
                # No images left
                self.current_image_path = None
                self.current_label_path = None
                self.labels = []
                self.raw_image = None
                self.canvas.delete("all")
                self.lbl_status.config(text="No images")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete: {e}")

    def on_right_click(self, event):
        # Quick delete
        self.on_mouse_down(event) # Select
        if self.selected_box_index != -1:
            self.delete_selected()
