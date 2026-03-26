#!/usr/bin/env python3
"""
Image Viewer with OpenCode Chat Integration
A tkinter GUI application for viewing images and chatting with OpenCode AI.
"""

import os
import base64
import subprocess
import threading
from pathlib import Path
from tkinter import (
    Tk, Frame, Label, Button, Entry, Text, Scrollbar, Checkbutton,
    filedialog, messagebox, BooleanVar, Canvas, END
)
from PIL import Image, ImageTk
import requests

SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')


class OpenCodeWorker:
    def __init__(self, master, server_url, session_id, message, image_path, callback):
        self.master = master
        self.server_url = server_url
        self.session_id = session_id
        self.message = message
        self.image_path = image_path
        self.callback = callback
    
    def run(self):
        try:
            parts = [{"type": "text", "text": self.message, "role": "user"}]
            
            if self.image_path and os.path.exists(self.image_path):
                with open(self.image_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                image_ext = os.path.splitext(self.image_path)[1].lower().replace('.', '')
                mime_type = f"image/{image_ext}" if image_ext not in ('jpg', 'jpeg') else 'image/jpeg'
                parts.append({
                    "type": "file",
                    "mime": mime_type,
                    "url": f"data:{mime_type};base64,{img_data}",
                    "filename": os.path.basename(self.image_path)
                })
            
            payload = {"parts": parts}
            
            response = requests.post(
                f"{self.server_url}/session/{self.session_id}/message",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                text_parts = []
                for part in data.get('parts', []):
                    if part.get('type') == 'text':
                        text_parts.append(part.get('text', ''))
                response_text = ' '.join(text_parts) if text_parts else 'No response'
                self.master.after(0, lambda t=response_text: self.callback(t))
            else:
                error_msg = response.text
                if "does not support image input" in error_msg.lower():
                    self.master.after(0, lambda: self.callback("Error: This model does not support image input. Please uncheck 'Include Current Image' and try again, or use a different model."))
                else:
                    self.master.after(0, lambda: self.callback(f"Server error: {response.status_code} - {error_msg}"))
        except requests.exceptions.ConnectionError:
            self.master.after(0, lambda: self.callback("Could not connect to OpenCode server"))
        except Exception as e:
            self.master.after(0, lambda msg=str(e): self.callback(f"Error: {msg}"))


class ImageViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer with OpenCode Chat")
        self.root.geometry("1200x800")
        
        self.images = []
        self.current_index = 0
        self.current_image_path = None
        self.server_process = None
        self.server_url = "http://127.0.0.1:8080"
        self.server_running = False
        self.photo = None
        self.session_id = None
        
        self.select_mode = False
        self.bbox_mode = False
        self.selecting = False
        self.sel_start = (0, 0)
        self.sel_rect = None
        self.selection_bbox = None
        self.modified_image_path = None
        self.original_pil_image = None
        self.scaled_pil_image = None
        
        self.bbox_list = []
        self.bbox_rects = []
        self.current_bbox_rect = None
        self.output_folder = './train'
        
        self.setup_ui()
        self.check_opencode()
    
    def setup_ui(self):
        main_frame = Frame(self.root)
        main_frame.pack(fill='both', expand=True)
        
        left_frame = Frame(main_frame, width=750)
        left_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        right_frame = Frame(main_frame, width=450)
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        self.setup_image_viewer(left_frame)
        self.setup_chat_panel(right_frame)
        self.setup_control_bar(left_frame)
    
    def setup_image_viewer(self, parent):
        viewer_frame = Frame(parent)
        viewer_frame.pack(fill='both', expand=True)
        
        Label(viewer_frame, text="Image Viewer", font=('Arial', 14, 'bold')).pack(pady=5)
        
        self.canvas = Canvas(viewer_frame, bg='#2b2b2b', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, pady=5)
        
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor='center')
        
        self.status_label = Label(viewer_frame, text="No image loaded", 
                                   font=('Arial', 10), anchor='center')
        self.status_label.pack(fill='x', pady=5)
    
    def setup_chat_panel(self, parent):
        chat_frame = Frame(parent)
        chat_frame.pack(fill='both', expand=True)
        
        Label(chat_frame, text="OpenCode Chat", font=('Arial', 14, 'bold')).pack(pady=5)
        
        self.connection_label = Label(chat_frame, text="Status: Checking OpenCode...", 
                                       font=('Arial', 9), fg='gray')
        self.connection_label.pack(fill='x', pady=(0, 5))
        
        self.server_btn = Button(chat_frame, text="Start OpenCode Server", 
                                  command=self.toggle_server, bg='#4CAF50', fg='white')
        self.server_btn.pack(fill='x', pady=(0, 5))
        
        text_frame = Frame(chat_frame)
        text_frame.pack(fill='both', expand=True, pady=5)
        
        self.chat_display = Text(text_frame, wrap='word', font=('Arial', 10),
                                  bg='#1e1e1e', fg='#d4d4d4', insertbackground='white')
        scrollbar = Scrollbar(text_frame, command=self.chat_display.yview)
        self.chat_display.config(yscrollcommand=scrollbar.set)
        
        self.chat_display.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        input_frame = Frame(chat_frame)
        input_frame.pack(fill='x', pady=5)
        
        self.chat_input = Entry(input_frame, font=('Arial', 11))
        self.chat_input.pack(side='left', fill='x', expand=True, padx=(0, 5))
        self.chat_input.bind('<Return>', lambda e: self.send_message())
        
        self.send_btn = Button(input_frame, text="Send", command=self.send_message,
                               bg='#2196F3', fg='white')
        self.send_btn.pack(side='right')
        
        checkbox_frame = Frame(chat_frame)
        checkbox_frame.pack(fill='x', pady=5)
        
        self.include_image_var = BooleanVar(value=True)
        include_check = Checkbutton(checkbox_frame, text="Include Current Image",
                                     variable=self.include_image_var)
        include_check.pack(side='left')
    
    def setup_control_bar(self, parent):
        control_frame = Frame(parent)
        control_frame.pack(fill='x', pady=5)
        
        Button(control_frame, text="Open Folder", command=self.open_folder).pack(side='left', padx=2)
        Button(control_frame, text="<< Prev", command=self.prev_image).pack(side='left', padx=2)
        Button(control_frame, text="Next >>", command=self.next_image).pack(side='left', padx=2)
        Button(control_frame, text="Select", command=self.toggle_select_mode, bg='#FF9800', fg='white').pack(side='left', padx=2)
        Button(control_frame, text="Reset", command=self.reset_selection, bg='#9C27B0', fg='white').pack(side='left', padx=2)
        Button(control_frame, text="Bbox", command=self.toggle_bbox_mode, bg='#2196F3', fg='white').pack(side='left', padx=2)
        
        output_frame = Frame(parent)
        output_frame.pack(fill='x', pady=5)
        
        Label(output_frame, text="Output Folder:").pack(side='left', padx=2)
        self.output_input = Entry(output_frame, font=('Arial', 10), width=30)
        self.output_input.insert(0, self.output_folder)
        self.output_input.pack(side='left', padx=2)
        self.output_input.bind('<Return>', lambda e: self.update_output_folder())
        self.output_input.bind('<FocusOut>', lambda e: self.update_output_folder())
        Button(output_frame, text="Browse", command=self.browse_output_folder).pack(side='left', padx=2)
    
    def browse_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder = folder
            self.output_input.delete(0, 'end')
            self.output_input.insert(0, folder)
            os.makedirs(f'{folder}/images', exist_ok=True)
            os.makedirs(f'{folder}/labels', exist_ok=True)
            self.add_chat_message("system", f"Output folder set to: {folder}/images and {folder}/labels")
    
    def update_output_folder(self):
        folder = self.output_input.get().strip()
        if folder:
            self.output_folder = folder
            os.makedirs(f'{folder}/images', exist_ok=True)
            os.makedirs(f'{folder}/labels', exist_ok=True)
    
    def toggle_select_mode(self):
        self.select_mode = not self.select_mode
        self.bbox_mode = False
        if self.sel_rect:
            self.canvas.delete(self.sel_rect)
            self.sel_rect = None
        if self.current_bbox_rect:
            self.canvas.delete(self.current_bbox_rect)
            self.current_bbox_rect = None
        if self.select_mode:
            self.add_chat_message("system", "Select mode ON - click and drag on image to select region")
        else:
            self.add_chat_message("system", "Select mode OFF")
    
    def toggle_bbox_mode(self):
        self.bbox_mode = not self.bbox_mode
        self.select_mode = False
        if self.sel_rect:
            self.canvas.delete(self.sel_rect)
            self.sel_rect = None
        if self.bbox_mode:
            self.add_chat_message("system", "Bbox mode ON - click and drag to draw multiple boxes, then Save Bbox")
        else:
            self.add_chat_message("system", "Bbox mode OFF")
    
    def on_mouse_press(self, event):
        if not (self.select_mode or self.bbox_mode) or not self.original_pil_image:
            return
        
        self.selecting = True
        self.sel_start = (event.x, event.y)
        
        if self.bbox_mode and self.current_bbox_rect:
            self.canvas.delete(self.current_bbox_rect)
        
        color = 'red' if self.select_mode else 'yellow'
        self.current_bbox_rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline=color, width=2
        )
    
    def on_mouse_drag(self, event):
        if not self.selecting:
            return
        self.canvas.coords(self.current_bbox_rect, self.sel_start[0], self.sel_start[1], event.x, event.y)
    
    def on_mouse_release(self, event):
        if not self.selecting:
            return
        self.selecting = False
        
        x1, y1 = self.sel_start
        x2, y2 = event.x, event.y
        
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        if x2 - x1 < 5 or y2 - y1 < 5:
            if self.current_bbox_rect:
                self.canvas.delete(self.current_bbox_rect)
                self.current_bbox_rect = None
            return
        
        if self.select_mode:
            self.finalize_selection(x1, y1, x2, y2)
        elif self.bbox_mode:
            self.finalize_bbox(x1, y1, x2, y2)
    
    def finalize_selection(self, x1, y1, x2, y2):
        canvas_width = self.canvas.winfo_width() or 600
        canvas_height = self.canvas.winfo_height() or 500
        
        img_display_w = self.scaled_pil_image.width
        img_display_h = self.scaled_pil_image.height
        
        offset_x = (canvas_width - img_display_w) // 2
        offset_y = (canvas_height - img_display_h) // 2
        
        img_x1 = max(0, x1 - offset_x)
        img_y1 = max(0, y1 - offset_y)
        img_x2 = min(img_display_w, x2 - offset_x)
        img_y2 = min(img_display_h, y2 - offset_y)
        
        scale_x = self.original_pil_image.width / img_display_w
        scale_y = self.original_pil_image.height / img_display_h
        
        orig_x1 = int(img_x1 * scale_x)
        orig_y1 = int(img_y1 * scale_y)
        orig_x2 = int(img_x2 * scale_x)
        orig_y2 = int(img_y2 * scale_y)
        
        if orig_x2 > orig_x1 and orig_y2 > orig_y1:
            self.selection_bbox = (orig_x1, orig_y1, orig_x2, orig_y2)
            if self.current_bbox_rect:
                self.canvas.delete(self.current_bbox_rect)
                self.current_bbox_rect = None
            self.apply_mask()
            self.add_chat_message("system", f"Region selected: ({orig_x1}, {orig_y1}) to ({orig_x2}, {orig_y2})")
        else:
            self.add_chat_message("system", "Selection too small")
    
    def finalize_bbox(self, x1, y1, x2, y2):
        canvas_width = self.canvas.winfo_width() or 600
        canvas_height = self.canvas.winfo_height() or 500
        
        img_display_w = self.scaled_pil_image.width
        img_display_h = self.scaled_pil_image.height
        
        offset_x = (canvas_width - img_display_w) // 2
        offset_y = (canvas_height - img_display_h) // 2
        
        img_x1 = max(0, x1 - offset_x)
        img_y1 = max(0, y1 - offset_y)
        img_x2 = min(img_display_w, x2 - offset_x)
        img_y2 = min(img_display_h, y2 - offset_y)
        
        if img_x2 <= img_x1 or img_y2 <= img_y1:
            return
        
        scale_x = self.original_pil_image.width / img_display_w
        scale_y = self.original_pil_image.height / img_display_h
        
        orig_x1 = img_x1 * scale_x
        orig_y1 = img_y1 * scale_y
        orig_x2 = img_x2 * scale_x
        orig_y2 = img_y2 * scale_y
        
        orig_w = orig_x2 - orig_x1
        orig_h = orig_y2 - orig_y1
        
        x_center = (orig_x1 + orig_x2) / 2 / self.original_pil_image.width
        y_center = (orig_y1 + orig_y2) / 2 / self.original_pil_image.height
        w_norm = orig_w / self.original_pil_image.width
        h_norm = orig_h / self.original_pil_image.height
        
        bbox_data = (0, x_center, y_center, w_norm, h_norm)
        self.bbox_list.append(bbox_data)
        
        self.bbox_rects.append(self.current_bbox_rect)
        self.current_bbox_rect = None
        
        self._save_bbox_to_file()
        
        box_num = len(self.bbox_list)
        self.add_chat_message("system", f"Bbox #{box_num} added: class=0, xc={x_center:.4f}, yc={y_center:.4f}, w={w_norm:.4f}, h={h_norm:.4f}")
    
    def _save_bbox_to_file(self):
        if not self.current_image_path:
            return
        os.makedirs(f'{self.output_folder}/labels', exist_ok=True)
        img_name = Path(self.current_image_path).stem
        label_path = f'{self.output_folder}/labels/{img_name}.txt'
        with open(label_path, 'w') as f:
            for bbox in self.bbox_list:
                f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
        self.add_chat_message("system", f"Saved label to: {label_path}")
    
    def save_bbox(self):
        if not self.bbox_list or not self.current_image_path:
            messagebox.showwarning("No Bboxes", "Draw some bounding boxes first")
            return
        
        os.makedirs(f'{self.output_folder}/images', exist_ok=True)
        os.makedirs(f'{self.output_folder}/labels', exist_ok=True)
        
        img_name = Path(self.current_image_path).stem
        img_ext = Path(self.current_image_path).suffix
        
        img_path = f'{self.output_folder}/images/{img_name}{img_ext}'
        if self.selection_bbox and self.modified_image_path and os.path.exists(self.modified_image_path):
            from shutil import copy2
            copy2(self.modified_image_path, img_path)
        else:
            self.original_pil_image.save(img_path)
        
        label_path = f'{self.output_folder}/labels/{img_name}.txt'
        
        with open(label_path, 'w') as f:
            for bbox in self.bbox_list:
                f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
        
        self.add_chat_message("system", f"Saved: {img_path} and {label_path}")
    
    def clear_bboxes(self):
        for rect in self.bbox_rects:
            self.canvas.delete(rect)
        if self.current_bbox_rect:
            self.canvas.delete(self.current_bbox_rect)
        self.bbox_rects = []
        self.bbox_list = []
        self.current_bbox_rect = None
        self.bbox_mode = False
    
    def apply_mask(self):
        if not self.original_pil_image or not self.selection_bbox:
            return
        
        os.makedirs('./tmp_image', exist_ok=True)
        os.makedirs('./tmp', exist_ok=True)
        os.makedirs(f'{self.output_folder}/images', exist_ok=True)
        
        img = self.original_pil_image.copy()
        x1, y1, x2, y2 = self.selection_bbox
        
        masked = Image.new('RGB', (img.width, img.height), (0, 0, 0))
        masked.paste(img.crop((x1, y1, x2, y2)), (x1, y1))
        
        img_name = Path(self.current_image_path).stem
        img_ext = Path(self.current_image_path).suffix
        self.modified_image_path = f'./tmp_image/{img_name}_masked.png'
        masked.save(self.modified_image_path)
        
        output_img_path = f'{self.output_folder}/images/{img_name}{img_ext}'
        masked.save(output_img_path)
        self.add_chat_message("system", f"Saved image to: {output_img_path}")
        
        sel_path = f'./tmp/{img_name}_selection.txt'
        with open(sel_path, 'w') as f:
            f.write(f"{x1},{y1},{x2},{y2}\n")
        
        self.display_modified_image()
    
    def display_modified_image(self):
        if not self.modified_image_path:
            return
        
        img = Image.open(self.modified_image_path)
        canvas_width = self.canvas.winfo_width() or 600
        canvas_height = self.canvas.winfo_height() or 500
        
        img.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.itemconfig(self.image_on_canvas, image=self.photo)
    
    def reset_selection(self):
        if self.current_image_path:
            img_name = Path(self.current_image_path).stem
            try:
                os.remove(f'./tmp/{img_name}_selection.txt')
            except: pass
            try:
                os.remove(f'./tmp_image/{img_name}_masked.png')
            except: pass
            try:
                os.remove(f'{self.output_folder}/labels/{img_name}.txt')
            except: pass
        
        self.selection_bbox = None
        self.modified_image_path = None
        self.select_mode = False
        if self.sel_rect:
            self.canvas.delete(self.sel_rect)
            self.sel_rect = None
        if self.current_bbox_rect:
            self.canvas.delete(self.current_bbox_rect)
            self.current_bbox_rect = None
        self.clear_bboxes()
        self.display_current_image()
        self.add_chat_message("system", "Reset - showing full image")
    
    def check_opencode(self):
        try:
            result = subprocess.run(['opencode', '--version'], 
                                   capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.connection_label.config(text=f"OpenCode CLI Found: {result.stdout.strip()}", fg='green')
                self.add_chat_message("system", "OpenCode CLI detected. Click 'Start OpenCode Server' to begin.")
            else:
                self.connection_label.config(text="OpenCode CLI not properly installed", fg='red')
                self.add_chat_message("system", "OpenCode CLI not found. Please install it first.")
        except FileNotFoundError:
            self.connection_label.config(text="OpenCode CLI not found", fg='red')
            self.add_chat_message("system", "OpenCode CLI not found. Please install it from https://opencode.ai")
        except Exception as e:
            self.connection_label.config(text=f"Error: {e}", fg='red')
            self.add_chat_message("system", f"Error checking OpenCode: {e}")
    
    def toggle_server(self):
        if not self.server_running:
            self.start_server()
        else:
            self.stop_server()
    
    def start_server(self):
        self.add_chat_message("system", "Starting OpenCode server...")
        self.server_btn.config(text="Starting...", state='disabled')
        
        threading.Thread(target=self._run_server, daemon=True).start()
        self.root.after(2000, self.check_server_status)
    
    def _run_server(self):
        try:
            self.server_process = subprocess.Popen(
                ['opencode', 'serve', '--port', '8080', '--hostname', '127.0.0.1'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except Exception as e:
            self.root.after(0, lambda: self.add_chat_message("system", f"Failed to start server: {e}"))
    
    def check_server_status(self):
        try:
            response = requests.get(f"{self.server_url}/global/health", timeout=2)
            if response.status_code == 200:
                self.server_running = True
                self.connection_label.config(text="Status: Connected to OpenCode Server", fg='green')
                self.server_btn.config(text="Stop OpenCode Server", state='normal', bg='#f44336')
                self.add_chat_message("system", "Connected to OpenCode server!")
                if not self.session_id:
                    self.create_session()
                return
        except:
            pass
        
        if self.server_process is None or self.server_process.poll() is not None:
            self.connection_label.config(text="Status: Server not running", fg='red')
            self.server_btn.config(text="Start OpenCode Server", state='normal', bg='#4CAF50')
        else:
            self.root.after(1000, self.check_server_status)
    
    def create_session(self):
        try:
            response = requests.post(f"{self.server_url}/session", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.session_id = data.get('id')
                self.add_chat_message("system", f"Session created: {self.session_id}")
            else:
                self.add_chat_message("system", f"Failed to create session: {response.status_code}")
        except Exception as e:
            self.add_chat_message("system", f"Error creating session: {e}")
    
    def stop_server(self):
        if self.server_process:
            self.server_process.terminate()
            self.server_process = None
            self.server_running = False
            self.session_id = None
            self.connection_label.config(text="Status: Server stopped", fg='orange')
            self.server_btn.config(text="Start OpenCode Server", state='normal', bg='#4CAF50')
            self.add_chat_message("system", "Server stopped.")
    
    def open_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.load_images_from_folder(folder)
    
    def load_images_from_folder(self, folder):
        self.images = []
        folder_path = Path(folder)
        
        for ext in SUPPORTED_FORMATS:
            self.images.extend(folder_path.glob(f'*{ext}'))
            self.images.extend(folder_path.glob(f'*{ext.upper()}'))
        
        self.images = sorted(set(self.images))
        
        if not self.images:
            messagebox.showinfo("No Images", "No supported images found in the selected folder.")
            return
        
        self.current_index = 0
        self.reset_selection()
        self.display_current_image()
        self.add_chat_message("system", f"Loaded {len(self.images)} images from {folder}")
    
    def display_current_image(self):
        if not self.images or self.current_index < 0:
            return
        
        image_path = self.images[self.current_index]
        self.current_image_path = str(image_path)
        self.selection_bbox = None
        self.modified_image_path = None
        
        if self.sel_rect:
            self.canvas.delete(self.sel_rect)
            self.sel_rect = None
        if self.current_bbox_rect:
            self.canvas.delete(self.current_bbox_rect)
            self.current_bbox_rect = None
        
        self.clear_bboxes()
        
        try:
            self.original_pil_image = Image.open(image_path)
            
            canvas_width = self.canvas.winfo_width() or 600
            canvas_height = self.canvas.winfo_height() or 500
            
            self.scaled_pil_image = self.original_pil_image.copy()
            self.scaled_pil_image.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
            
            self.photo = ImageTk.PhotoImage(self.scaled_pil_image)
            
            self.canvas.itemconfig(self.image_on_canvas, image=self.photo)
            
            x = canvas_width // 2
            y = canvas_height // 2
            self.canvas.coords(self.image_on_canvas, x, y)
            
            img_display_w = self.scaled_pil_image.width
            img_display_h = self.scaled_pil_image.height
            offset_x = (canvas_width - img_display_w) // 2
            offset_y = (canvas_height - img_display_h) // 2
            
            self.status_label.config(
                text=f"{self.current_index + 1}/{len(self.images)} - {image_path.name}"
            )
            
            img_name = Path(image_path).stem
            
            sel_path = f'./tmp/{img_name}_selection.txt'
            if os.path.exists(sel_path):
                with open(sel_path, 'r') as f:
                    line = f.readline().strip()
                    x1, y1, x2, y2 = map(int, line.split(','))
                    self.selection_bbox = (x1, y1, x2, y2)
                
                masked_path = f'./tmp_image/{img_name}_masked.png'
                if os.path.exists(masked_path):
                    self.modified_image_path = masked_path
                    mod_img = Image.open(masked_path)
                    mod_img.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
                    self.photo = ImageTk.PhotoImage(mod_img)
                    self.canvas.itemconfig(self.image_on_canvas, image=self.photo)
                    self.canvas.coords(self.image_on_canvas, x, y)
            
            label_path = f'{self.output_folder}/labels/{img_name}.txt'
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id, xc, yc, w, h = map(float, parts)
                            bbox = (int(class_id), xc, yc, w, h)
                            self.bbox_list.append(bbox)
                            
                            px1 = int((xc - w/2) * img_display_w) + offset_x
                            py1 = int((yc - h/2) * img_display_h) + offset_y
                            px2 = int((xc + w/2) * img_display_w) + offset_x
                            py2 = int((yc + h/2) * img_display_h) + offset_y
                            rect = self.canvas.create_rectangle(px1, py1, px2, py2, outline='yellow', width=2)
                            self.bbox_rects.append(rect)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def prev_image(self):
        if not self.images:
            return
        self.current_index = (self.current_index - 1) % len(self.images)
        self.display_current_image()
    
    def next_image(self):
        if not self.images:
            return
        self.current_index = (self.current_index + 1) % len(self.images)
        self.display_current_image()
    
    def add_chat_message(self, role, content):
        self.chat_display.config(state='normal')
        
        prefix = f"[{role.upper()}] "
        color = {'user': '#4fc3f7', 'assistant': '#81c784', 'system': '#9e9e9e'}.get(role, 'white')
        
        self.chat_display.insert(END, f"\n{prefix}{content}\n")
        self.chat_display.tag_add(role, f"end-{len(content)+len(prefix)+2}c", "end-1c")
        self.chat_display.tag_config(role, foreground=color)
        
        self.chat_display.see(END)
        self.chat_display.config(state='disabled')
    
    def send_message(self):
        message = self.chat_input.get().strip()
        if not message:
            return
        
        if not self.server_running or not self.session_id:
            messagebox.showwarning("Server Not Running", 
                                   "Please start the OpenCode server first.")
            return
        
        self.chat_input.delete(0, 'end')
        self.add_chat_message("user", message)
        
        image_path = None
        if self.include_image_var.get():
            image_path = self.modified_image_path or self.current_image_path
        
        self.send_btn.config(state='disabled')
        
        worker = OpenCodeWorker(self.root, self.server_url, self.session_id, message, image_path, self.on_response)
        thread = threading.Thread(target=worker.run, daemon=True)
        thread.start()
    
    def on_response(self, response):
        self.add_chat_message("assistant", response)
        self.send_btn.config(state='normal')
    
    def on_closing(self):
        self.stop_server()
        self.root.destroy()


def main():
    root = Tk()
    app = ImageViewerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
