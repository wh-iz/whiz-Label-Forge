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
        
        # View transforms
        self.fit_scale = 1.0    # Scale to fit window
        self.zoom_level = 1.0   # User zoom
        self.pan_x = 0          # User pan X
        self.pan_y = 0          # User pan Y
        
        self.offset_x = 0       # Calculated center offset X
        self.offset_y = 0       # Calculated center offset Y
        
        # Labels: list of [class_id, x_center, y_center, w, h] (normalized)
        self.labels = []
        
        # Drawing state
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        self.selected_box_index = -1
        
        # Interaction state
        self.interaction_mode = 'idle' # idle, draw, move, resize, pan
        self.active_handle = None
        self.drag_start_pos = None
        self.initial_box_state = None
        self.pan_start_pos = None
        
        # History
        self.history = []
        self.history_index = -1
        
        self.thumbnail_cache = {}
        
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
        
        # Treeview with Scrollbar
        self.sidebar_frame = ttk.Frame(self.sidebar)
        self.sidebar_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.file_tree = ttk.Treeview(self.sidebar_frame, selectmode="extended", show="tree", columns=("filename"))
        self.file_tree.column("#0", width=60, minwidth=60, stretch=False) # Thumbnail column
        self.file_tree.column("filename", anchor="w")
        self.file_tree.heading("#0", text="Prev")
        self.file_tree.heading("filename", text="Filename")
        
        self.scrollbar = ttk.Scrollbar(self.sidebar_frame, orient="vertical", command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=self.scrollbar.set)
        
        self.file_tree.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        self.file_tree.bind("<<TreeviewSelect>>", self.on_file_select)
        
        # Configure row height for thumbnails
        style = ttk.Style()
        style.configure("Treeview", rowheight=60)
        
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
        self.class_entry.bind("<Return>", self.update_selected_class)
        self.class_entry.bind("<FocusOut>", self.update_selected_class)
        
        ttk.Button(self.toolbar, text="Save (Ctrl+S)", command=self.save_labels).pack(side="left", padx=10)
        ttk.Button(self.toolbar, text="Delete Box (Del)", command=self.delete_selected).pack(side="left", padx=5)
        ttk.Button(self.toolbar, text="Delete Image", command=self.delete_selected_images).pack(side="left", padx=5)
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
        
        # Pan & Zoom
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start) # Middle click
        self.canvas.bind("<B2-Motion>", self.on_pan_drag)
        self.canvas.bind("<MouseWheel>", self.on_zoom) # Windows
        self.canvas.bind("<Button-4>", self.on_zoom)   # Linux up
        self.canvas.bind("<Button-5>", self.on_zoom)   # Linux down
        
        # Right click to select/delete
        self.canvas.bind("<ButtonPress-3>", self.on_right_click)
        
        # Key bindings
        self.canvas.bind("<Delete>", lambda e: self.delete_selected())
        self.canvas.bind("<Control-s>", lambda e: self.save_labels())
        self.canvas.bind("<Control-z>", lambda e: self.undo())
        self.canvas.bind("<Control-y>", lambda e: self.redo())
        self.canvas.bind("<Left>", lambda e: self.prev_image())
        self.canvas.bind("<Right>", lambda e: self.next_image())
        
        # Focus canvas on enter
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())

    def get_thumbnail(self, path):
        if path in self.thumbnail_cache:
            return self.thumbnail_cache[path]
            
        try:
            img = Image.open(path)
            img.thumbnail((50, 50))
            photo = ImageTk.PhotoImage(img)
            self.thumbnail_cache[path] = photo
            return photo
        except Exception:
            return None

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
        
        # Clear Treeview
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
            
        # Populate Treeview
        for i, f in enumerate(self.image_list):
            full_path = os.path.join(img_dir, f)
            thumb = self.get_thumbnail(full_path)
            # Use i as iid to easily map back to index
            if thumb:
                self.file_tree.insert("", "end", iid=str(i), text="", image=thumb, values=(f,))
            else:
                self.file_tree.insert("", "end", iid=str(i), text="", values=(f,))
            
        self.lbl_status.config(text=f"Found {len(self.image_list)} images")

    def on_file_select(self, event):
        sel = self.file_tree.selection()
        if not sel: return
        
        # In extended mode, user can select multiple. 
        # For preview, we just load the first one selected.
        try:
            idx = int(sel[0])
            self.load_image_by_index(idx)
        except ValueError:
            pass

    def update_selected_class(self, event=None):
        if self.selected_box_index != -1 and 0 <= self.selected_box_index < len(self.labels):
            new_cls = self.class_var.get()
            # Update the class of the selected box
            self.labels[self.selected_box_index][0] = new_cls
            self.draw_boxes()

    def load_image_by_index(self, index):
        if index < 0 or index >= len(self.image_list): return
        
        # Auto-save previous image
        if self.current_image_path and self.current_label_path:
             self.save_labels(silent=True)

        self.current_index = index
        filename = self.image_list[index]
        output_path = self.app.output_path.get()
        
        self.current_image_path = os.path.join(output_path, "images", filename)
        
        # Load Labels
        label_name = os.path.splitext(filename)[0] + ".txt"
        self.current_label_path = os.path.join(output_path, "labels", label_name)
        
        # Reset View
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
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
        self.reset_history()

    def reset_history(self):
        self.history = []
        self.history_index = -1
        self.save_state()

    def save_state(self):
        # Deep copy labels
        current_state = [list(lbl) for lbl in self.labels]
        
        # If we are not at the end, cut the future
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index+1]
            
        self.history.append(current_state)
        self.history_index += 1
        
        # Limit history
        if len(self.history) > 50:
            self.history.pop(0)
            self.history_index -= 1

    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.labels = [list(lbl) for lbl in self.history[self.history_index]]
            self.draw_boxes()
            self.lbl_status.config(text="Undo")

    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.labels = [list(lbl) for lbl in self.history[self.history_index]]
            self.draw_boxes()
            self.lbl_status.config(text="Redo")


    def save_labels(self, silent=False):
        if not self.current_label_path: return
        
        # Ensure labels dir exists
        os.makedirs(os.path.dirname(self.current_label_path), exist_ok=True)
        
        with open(self.current_label_path, "w") as f:
            for lbl in self.labels:
                f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")
        
        if not silent:
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

    @property
    def total_scale(self):
        return self.fit_scale * self.zoom_level

    def image_to_screen(self, ix, iy):
        sx = ix * self.total_scale + self.offset_x + self.pan_x
        sy = iy * self.total_scale + self.offset_y + self.pan_y
        return sx, sy

    def screen_to_image(self, sx, sy):
        if self.total_scale == 0: return 0, 0
        ix = (sx - self.offset_x - self.pan_x) / self.total_scale
        iy = (sy - self.offset_y - self.pan_y) / self.total_scale
        return ix, iy

    def redraw(self):
        if not self.raw_image: return
        
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        iw, ih = self.raw_image.size
        
        if cw <= 1 or ch <= 1: return
        
        # Calculate fit scale
        scale_w = cw / iw
        scale_h = ch / ih
        self.fit_scale = min(scale_w, scale_h) * 0.95
        
        # Current drawing dimensions
        current_scale = self.total_scale
        nw = int(iw * current_scale)
        nh = int(ih * current_scale)
        
        # Center offset (base)
        self.offset_x = (cw - nw) // 2
        self.offset_y = (ch - nh) // 2
        
        # Final draw position
        final_x = self.offset_x + self.pan_x
        final_y = self.offset_y + self.pan_y
        
        try:
            # Resize image for display
            resized = self.raw_image.resize((max(1, nw), max(1, nh)), Image.Resampling.NEAREST)
            self.tk_image = ImageTk.PhotoImage(resized)
            
            self.canvas.delete("all")
            self.canvas.create_image(final_x, final_y, anchor="nw", image=self.tk_image)
            
            self.draw_boxes()
        except Exception as e:
            print(f"Redraw error: {e}")

    def get_box_pixel_coords(self, index):
        if index < 0 or index >= len(self.labels): return None
        cls, cx, cy, w, h = self.labels[index]
        iw, ih = self.raw_image.size
        
        # Normalized to Image Pixels
        img_x1 = (cx - w/2) * iw
        img_y1 = (cy - h/2) * ih
        img_x2 = (cx + w/2) * iw
        img_y2 = (cy + h/2) * ih
        
        # Image Pixels to Screen Pixels
        x1, y1 = self.image_to_screen(img_x1, img_y1)
        x2, y2 = self.image_to_screen(img_x2, img_y2)
        
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
            
            if i == self.selected_box_index:
                self.draw_handles(x1, y1, x2, y2)

    def draw_handles(self, x1, y1, x2, y2):
        handles = self.get_handle_coords(x1, y1, x2, y2)
        for name, (hx1, hy1, hx2, hy2) in handles.items():
            self.canvas.create_rectangle(hx1, hy1, hx2, hy2, fill="cyan", outline="black", tags=("handle", f"handle_{name}"))

    def get_handle_coords(self, x1, y1, x2, y2):
        s = 8 
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

    def on_zoom(self, event):
        if not self.raw_image: return
        
        factor = 1.1
        if event.num == 5 or event.delta < 0:
            self.zoom_level /= factor
        else:
            self.zoom_level *= factor
            
        self.zoom_level = max(0.1, min(self.zoom_level, 50.0))
        self.redraw()

    def on_pan_start(self, event):
        if not self.raw_image: return
        self.interaction_mode = 'pan'
        self.pan_start_pos = (event.x, event.y)
        self.canvas.config(cursor="fleur")

    def on_pan_drag(self, event):
        if self.interaction_mode != 'pan': return
        dx = event.x - self.pan_start_pos[0]
        dy = event.y - self.pan_start_pos[1]
        
        self.pan_x += dx
        self.pan_y += dy
        
        self.pan_start_pos = (event.x, event.y)
        self.redraw()

    def on_mouse_down(self, event):
        if not self.raw_image: return
        
        x, y = self.get_canvas_coords(event)
        
        # 1. Check handles
        handle = self.get_handle_at(x, y)
        if handle:
            self.save_state() # Save before resize
            self.interaction_mode = 'resize'
            self.active_handle = handle
            self.drag_start_pos = (x, y)
            self.initial_box_state = list(self.labels[self.selected_box_index])
            return

        # 2. Check boxes
        clicked_box = -1
        for i in range(len(self.labels)-1, -1, -1):
            coords = self.get_box_pixel_coords(i)
            if not coords: continue
            x1, y1, x2, y2 = coords
            if x1 <= x <= x2 and y1 <= y <= y2:
                clicked_box = i
                break
        
        if clicked_box != -1:
            self.save_state() # Save before move
            self.selected_box_index = clicked_box
            self.class_var.set(str(self.labels[clicked_box][0]))
            self.draw_boxes()
            
            self.interaction_mode = 'move'
            self.drag_start_pos = (x, y)
            self.initial_box_state = list(self.labels[clicked_box])
            return
        
        # 3. Start Draw
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
            
            iw, ih = self.raw_image.size
            norm_dx = dx / self.total_scale / iw
            norm_dy = dy / self.total_scale / ih
            
            cls, cx, cy, w, h = self.initial_box_state
            self.labels[self.selected_box_index] = [cls, cx + norm_dx, cy + norm_dy, w, h]
            self.draw_boxes()
            
        elif self.interaction_mode == 'resize':
            if not self.drag_start_pos: return
            
            iw, ih = self.raw_image.size
            cls, cx, cy, w, h = self.initial_box_state
            
            # Initial image pixels
            img_x1 = (cx - w/2) * iw
            img_y1 = (cy - h/2) * ih
            img_x2 = (cx + w/2) * iw
            img_y2 = (cy + h/2) * ih
            
            # Screen Delta -> Image Delta
            dx_img = (x - self.drag_start_pos[0]) / self.total_scale
            dy_img = (y - self.drag_start_pos[1]) / self.total_scale
            
            if 'w' in self.active_handle: img_x1 += dx_img
            if 'e' in self.active_handle: img_x2 += dx_img
            if 'n' in self.active_handle: img_y1 += dy_img
            if 's' in self.active_handle: img_y2 += dy_img
            
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
                
                # Convert screen rect to image rect
                ix1, iy1 = self.screen_to_image(min(self.start_x, end_x), min(self.start_y, end_y))
                ix2, iy2 = self.screen_to_image(max(self.start_x, end_x), max(self.start_y, end_y))
                
                self.canvas.delete(self.current_rect)
                
                iw, ih = self.raw_image.size
                
                # Clamp to image
                ix1 = max(0, min(ix1, iw))
                iy1 = max(0, min(iy1, ih))
                ix2 = max(0, min(ix2, iw))
                iy2 = max(0, min(iy2, ih))
                
                if (ix2 - ix1) > 2 and (iy2 - iy1) > 2:
                    w = ix2 - ix1
                    h = iy2 - iy1
                    cx = ix1 + w/2
                    cy = iy1 + h/2
                    
                    cls_id = self.class_var.get() or "0"
                    self.labels.append([cls_id, cx/iw, cy/ih, w/iw, h/ih])
                    self.selected_box_index = len(self.labels) - 1
            
            self.start_x = None
            self.draw_boxes()
        
        if self.interaction_mode == 'pan':
            self.canvas.config(cursor="")
            
        self.interaction_mode = 'idle'
        self.active_handle = None
        self.drag_start_pos = None

    def prev_image(self):
        if self.current_index > 0:
            next_idx = self.current_index - 1
            self.file_tree.selection_set(str(next_idx))
            self.file_tree.see(str(next_idx))
            self.load_image_by_index(next_idx)

    def next_image(self):
        if self.current_index < len(self.image_list) - 1:
            next_idx = self.current_index + 1
            self.file_tree.selection_set(str(next_idx))
            self.file_tree.see(str(next_idx))
            self.load_image_by_index(next_idx)

    def delete_selected(self):
        if self.selected_box_index != -1:
            del self.labels[self.selected_box_index]
            self.selected_box_index = -1
            self.draw_boxes()

    def delete_selected_images(self):
        sel = self.file_tree.selection()
        if not sel: return

        if not messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete {len(sel)} images?"):
            return

        output_path = self.app.output_path.get()
        deleted_count = 0
        
        # Get filenames first because indices change if we pop from list
        # But we will rebuild list anyway
        files_to_delete = []
        for iid in sel:
            try:
                idx = int(iid)
                if 0 <= idx < len(self.image_list):
                    files_to_delete.append(self.image_list[idx])
            except ValueError:
                pass

        for fname in files_to_delete:
            try:
                img_path = os.path.join(output_path, "images", fname)
                lbl_path = os.path.join(output_path, "labels", os.path.splitext(fname)[0] + ".txt")
                
                if os.path.exists(img_path): os.remove(img_path)
                if os.path.exists(lbl_path): os.remove(lbl_path)
                
                # Remove from cache
                if img_path in self.thumbnail_cache:
                    del self.thumbnail_cache[img_path]
                    
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {fname}: {e}")

        # Refresh entire list
        self.refresh_file_list()
        
        # Reset view
        self.current_image_path = None
        self.current_label_path = None
        self.labels = []
        self.raw_image = None
        self.canvas.delete("all")
        self.lbl_status.config(text=f"Deleted {deleted_count} images")

    def on_right_click(self, event):
        self.on_mouse_down(event)
        if self.selected_box_index != -1:
            self.delete_selected()
