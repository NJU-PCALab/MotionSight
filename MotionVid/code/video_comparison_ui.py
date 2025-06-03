import os
import json
import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import glob
import io
import tempfile
from moviepy.editor import VideoFileClip

class VideoComparisonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Motion Comparison")
        self.root.geometry("1200x800")
        
        # Configure the root window to expand properly
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Font configurations
        self.title_font = ("Arial", 14, "bold")
        self.text_font = ("Arial", 13)
        self.button_font = ("Arial", 12)
        self.label_font = ("Arial", 12)
        
        # Data storage
        self.data = []
        self.results = {}
        self.evaluated_ids = set()  # Track which item IDs have been evaluated
        self.current_index = 0
        self.left_is_original = None
        self.history = []  # Track navigation history
        
        # Cache for frames
        self.frame_cache = {}  # Cache for loaded video frames
        self.next_item_index = None  # For preloading
        self.frames_dir = "frames"  # Directory to store preprocessed frames
        
        # UI elements
        self.setup_ui()
        
        # Keyboard bindings
        self.root.bind("<Left>", lambda e: self.record_preference("left"))
        self.root.bind("<Right>", lambda e: self.record_preference("right"))
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("<F11>", lambda e: self.toggle_fullscreen())
        
        # Additional key bindings for navigation
        self.root.bind("<Prior>", lambda e: self.go_to_previous_item())  # Page Up
        self.root.bind("<Next>", lambda e: self.next_item())  # Page Down
        
        # Initial state
        self.jsonl_file = None
        self.results_file = "results.jsonl"
        self.summary_file = "evaluation_summary.txt"
        self.temp_dir = None
        self.is_fullscreen = False
        
        # Create frames directory if it doesn't exist
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)
        
        # Register cleanup on closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes("-fullscreen", self.is_fullscreen)
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=2)  # Frames area - increased weight
        main_frame.rowconfigure(2, weight=2)  # Text comparison area
        
        # Top controls
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, sticky="ew", pady=10)
        control_frame.columnconfigure(4, weight=1)  # Make the frame expandable
        
        ttk.Label(control_frame, text="JSONL File:", font=self.label_font).grid(row=0, column=0, padx=5)
        self.file_label = ttk.Label(control_frame, text="No file selected", font=self.label_font)
        self.file_label.grid(row=0, column=1, padx=5)
        
        ttk.Button(control_frame, text="Open JSONL", command=self.load_jsonl_file, 
                  style="TButton").grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Start/Resume", command=self.start_evaluation, 
                  style="TButton").grid(row=0, column=3, padx=5)
        
        # Navigation button for previous items only
        ttk.Button(control_frame, text="← Previous", command=self.go_to_previous_item, 
                 style="TButton").grid(row=0, column=4, padx=5)
        
        # Progress info
        self.progress_label = ttk.Label(control_frame, text="0/0 evaluated", font=self.label_font)
        self.progress_label.grid(row=0, column=5, padx=5)
        
        # Create custom styles
        style = ttk.Style()
        style.configure("TButton", font=self.button_font)
        style.configure("TLabelframe.Label", font=self.title_font)
        
        # Video frames display area
        self.frames_frame = ttk.LabelFrame(main_frame, text="Video Frames")
        self.frames_frame.grid(row=1, column=0, sticky="nsew", pady=10)
        self.frames_frame.columnconfigure(0, weight=1)
        self.frames_frame.rowconfigure(0, weight=1)
        
        self.frames_grid = ttk.Frame(self.frames_frame)
        self.frames_grid.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Reduced to 2x4 grid for larger images (2 rows, 4 columns)
        self.image_labels = []
        for row in range(2):
            self.frames_grid.rowconfigure(row, weight=1)
            for col in range(4):  # Changed from 8 to 4 columns
                self.frames_grid.columnconfigure(col, weight=1)
                
                frame = ttk.Frame(self.frames_grid, borderwidth=1, relief="solid")
                frame.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")  # Increased padding
                
                label = ttk.Label(frame)
                label.pack(fill=tk.BOTH, expand=True)
                self.image_labels.append(label)
        
        # Text comparison area - INCREASED HEIGHT
        text_frame = ttk.Frame(main_frame)
        text_frame.grid(row=2, column=0, sticky="nsew", pady=10)
        text_frame.columnconfigure(0, weight=1)
        text_frame.columnconfigure(1, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        # Left text box
        left_frame = ttk.LabelFrame(text_frame, text="Option A (Left Arrow)")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5)
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)
        
        # Increased font size and height
        self.left_text = tk.Text(left_frame, wrap=tk.WORD, font=self.text_font)
        self.left_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Right text box
        right_frame = ttk.LabelFrame(text_frame, text="Option B (Right Arrow)")
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        
        # Increased font size
        self.right_text = tk.Text(right_frame, wrap=tk.WORD, font=self.text_font)
        self.right_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Instructions
        instruction_frame = ttk.Frame(main_frame)
        instruction_frame.grid(row=3, column=0, sticky="ew", pady=5)
        instruction_frame.columnconfigure(0, weight=1)
        
        instruction_text = "Use LEFT ARROW to select Option A or RIGHT ARROW to select Option B. Press F11 to toggle fullscreen. ESC to exit."
        ttk.Label(instruction_frame, text=instruction_text, font=self.label_font).grid(row=0, column=0)
        
        # Video info
        self.video_info = ttk.Label(main_frame, text="", font=self.label_font)
        self.video_info.grid(row=4, column=0, pady=5)
    
    def load_jsonl_file(self):
        file_path = filedialog.askopenfilename(
            title="Select JSONL file",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        self.jsonl_file = file_path
        self.file_label.config(text=os.path.basename(file_path))
        
        # Create results and summary filenames based on input file
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        self.results_file = f"{base_name}_results.jsonl"
        self.summary_file = f"{base_name}_summary.txt"
        
        try:
            self.data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.data.append(json.loads(line.strip()))
            
            self.load_existing_results()
            self.update_progress_label()
            
            # Show processing dialog
            process_videos = messagebox.askyesno(
                "Preprocess Videos", 
                "Would you like to preprocess all videos now?\n\n"
                "This will extract frames from all videos and may take some time,\n"
                "but will make evaluation faster."
            )
            
            if process_videos:
                self.preprocess_all_videos()
            
            messagebox.showinfo("Success", f"Loaded {len(self.data)} items from {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def load_existing_results(self):
        self.results = {}
        self.evaluated_ids = set()  # Reset evaluated IDs
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        result = json.loads(line.strip())
                        self.results[result['item_id']] = result
                        self.evaluated_ids.add(result['item_id'])
                
                messagebox.showinfo("Results", f"Loaded {len(self.results)} existing results")
                
                # Print evaluated IDs for debugging
                print(f"Loaded evaluated IDs: {sorted(self.evaluated_ids)}")
                print(f"Item IDs in data: {sorted(item.get('idx') for item in self.data if 'idx' in item)}")
                
            except Exception as e:
                messagebox.showwarning("Warning", f"Could not load existing results: {str(e)}")
    
    def all_items_evaluated(self):
        """Check if all items in the JSONL file have been evaluated"""
        # Get all item IDs from the data
        all_ids = {item.get('idx') for item in self.data if 'idx' in item}
        print(f"Evaluated IDs: {self.evaluated_ids}, Total unique IDs: {len(all_ids)}")
        print(f"All IDs: {all_ids}")
        # Check if all item IDs have been evaluated
        return all_ids.issubset(self.evaluated_ids) and len(all_ids) > 0
    
    def start_evaluation(self):
        if not self.jsonl_file:
            messagebox.showwarning("Warning", "Please select a JSONL file first")
            return
        
        # Check if all items have already been evaluated
        if self.all_items_evaluated():
            messagebox.showinfo("Already Complete", "All items have already been evaluated!")
            return
        
        # Clear history when starting a new evaluation
        self.history = []
        
        # Find the first item that hasn't been evaluated yet
        self.current_index = 0
        while self.current_index < len(self.data):
            item_id = self.data[self.current_index].get('idx')
            if item_id is not None and item_id not in self.evaluated_ids:
                break
            self.current_index += 1
        
        self.display_current_item()
        self.update_progress_label()
    
    def display_current_item(self):
        if self.current_index >= len(self.data):
            if self.all_items_evaluated():
                messagebox.showinfo("No More Items", "All items have been evaluated!")
            else:
                # This means we've reached the end but not all items are evaluated
                # Reset to the beginning and find next unevaluated item
                self.current_index = 0
                while self.current_index < len(self.data):
                    item_id = self.data[self.current_index].get('idx')
                    if item_id is not None and item_id not in self.evaluated_ids:
                        break
                    self.current_index += 1
                
                if self.current_index < len(self.data):
                    self.display_item_at_index(self.current_index)
                else:
                    print(f"Unexpected state: evaluated IDs {sorted(self.evaluated_ids)}")
                    for i, item in enumerate(self.data):
                        print(f"Item {i}: ID = {item.get('idx')}, video = {item.get('video_file')}")
                    messagebox.showinfo("Unexpected State", "Could not find unevaluated items, but not all marked as complete.")
            return
        
        # Display the current item and add to history
        self.display_item_at_index(self.current_index)
    
    def display_item_at_index(self, index, add_to_history=True):
        """Display item at specific index with option to add to navigation history"""
        if index < 0 or index >= len(self.data):
            messagebox.showinfo("Navigation", "Invalid item index")
            return
            
        # Add to history if requested
        if add_to_history:
            self.history.append(index)
            # Limit history size to prevent memory issues
            if len(self.history) > 100:
                self.history = self.history[-100:]
                
        self.current_index = index
        current_item = self.data[self.current_index]
        video_file = current_item.get('video_file', 'Unknown')
        item_id = current_item.get('idx', 'Unknown')
        
        # Show edit mode indicator if this item has already been evaluated
        already_evaluated = False
        if item_id in self.evaluated_ids:
            already_evaluated = True
        
        # Display video info with edit status
        status = " [EDIT MODE]" if already_evaluated else ""
        self.video_info.config(text=f"Video: {video_file} (ID: {item_id}){status}")
        
        # Load and display video frames
        self.load_video_frames(video_file)
        
        # Set up text comparison
        self.setup_text_comparison(current_item)
        
        # Start preloading the next item
        self.preload_next_item()
    
    def load_video_frames(self, video_file):
        """Load video frames, first checking preprocessed frames"""
        # Clear existing images
        for label in self.image_labels:
            label.config(image='')
        
        # Clean up any previous temporary directory
        self.cleanup_temp_dir()
        
        # First check if preprocessed frames exist for this video
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        video_frames_dir = os.path.join(self.frames_dir, video_name)
        
        if os.path.exists(video_frames_dir) and os.listdir(video_frames_dir):
            # Use preprocessed frames
            frame_files = sorted(glob.glob(os.path.join(video_frames_dir, "*.jpg")))
            if frame_files:
                self.display_frames(frame_files)
                return
        
        # If not found in preprocessed frames, try to extract from video file
        if os.path.isfile(video_file) and video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            try:
                # Create temporary directory to store extracted frames
                self.temp_dir = tempfile.mkdtemp(prefix="video_frames_")
                
                # Load the video and extract frames
                with VideoFileClip(video_file) as clip:
                    # Calculate how many frames to extract (16 frames evenly distributed)
                    duration = clip.duration
                    frame_times = [duration * i / 16 for i in range(16)]
                    
                    # Extract frames at the calculated times
                    frame_files = []
                    for i, t in enumerate(frame_times):
                        frame_path = os.path.join(self.temp_dir, f"frame_{i:03d}.jpg")
                        clip.save_frame(frame_path, t=t)
                        frame_files.append(frame_path)
                    
                    # Save to preprocessed directory for future use
                    if not os.path.exists(video_frames_dir):
                        os.makedirs(video_frames_dir)
                    
                    for i, frame_file in enumerate(frame_files):
                        dest_path = os.path.join(video_frames_dir, f"frame_{i:03d}.jpg")
                        # Copy instead of move to keep temp files for current display
                        import shutil
                        shutil.copy2(frame_file, dest_path)
                    
                    # Display the extracted frames
                    self.display_frames(frame_files)
                    return
            except Exception as e:
                messagebox.showwarning("Warning", f"Could not process video file: {str(e)}")
        
        # If still not found, try original method of searching for frames
        frames_pattern = os.path.join(os.path.dirname(video_file), os.path.basename(video_file).split('.')[0] + '_*.jpg')
        try:
            frame_files = sorted(glob.glob(frames_pattern))
            
            # If no frames found with the pattern, try looking in a frames directory
            if not frame_files:
                frames_pattern = os.path.join("frames", video_name, "*.jpg")
                frame_files = sorted(glob.glob(frames_pattern))
            
            # Display the frames if found
            if frame_files:
                self.display_frames(frame_files)
            else:
                # No frames found, display a message
                self.video_info.config(text=f"Video: {video_file} (No frames found)")
                
        except Exception as e:
            messagebox.showwarning("Warning", f"Could not load frames: {str(e)}")
    
    def display_frames(self, frame_files):
        # Sample frames evenly from all available frames
        # Adjust to match our 2x4 grid (8 frames total)
        if len(frame_files) < 8:
            selected_frames = frame_files
        else:
            step = len(frame_files) // 8
            selected_frames = [frame_files[i] for i in range(0, len(frame_files), step)[:8]]
            
            # Fill remaining slots if we didn't get 8 frames
            while len(selected_frames) < 8:
                selected_frames.append(frame_files[-1])
        
        # Load and display the selected frames
        for i, frame_file in enumerate(selected_frames[:8]):
            if i < len(self.image_labels):
                try:
                    # Open the image file
                    img = Image.open(frame_file)
                    
                    # Resize to a larger size (doubled from before)
                    img = img.resize((240, 180), Image.LANCZOS)
                    
                    # Convert to RGB if in RGBA mode to ensure compatibility
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    
                    # Compress the image to reduce memory usage
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=85)  # Higher quality
                    buffer.seek(0)
                    compressed_img = Image.open(buffer)
                    
                    # Create Tkinter-compatible photo image
                    photo = ImageTk.PhotoImage(compressed_img)
                    self.image_labels[i].config(image=photo)
                    self.image_labels[i].image = photo  # Keep a reference
                except Exception as e:
                    print(f"Error loading frame {frame_file}: {e}")
    
    def record_preference(self, choice):
        if self.current_index >= len(self.data) or self.left_is_original is None:
            return
        
        current_item = self.data[self.current_index]
        video_file = current_item.get('video_file', 'Unknown')
        item_id = current_item.get('idx')
        
        # Skip if no ID (shouldn't happen with proper data)
        if item_id is None:
            print(f"Warning: Item at index {self.current_index} has no ID, skipping")
            self.next_item()
            return
        
        # Determine which option was chosen (original or translated)
        chosen_original = (choice == "left" and self.left_is_original) or (choice == "right" and not self.left_is_original)
        
        # Record the result
        result = {
            'item_id': item_id,
            'video_file': video_file,
            'chosen': 'original' if chosen_original else 'translated',
            'reject': 'translated' if chosen_original else 'original'
        }
        
        # Check if we're editing an existing evaluation
        editing = item_id in self.evaluated_ids
        action = "Updated" if editing else "Recorded"
        
        self.results[item_id] = result
        
        # Mark this ID as evaluated
        self.evaluated_ids.add(item_id)
        
        # Save the result immediately (in a non-blocking way)
        self.root.after(10, lambda: self.save_result(result))
        
        # Update the progress counter
        self.update_progress_label()
        
        # If we were editing, show a confirmation
        if editing:
            self.video_info.config(text=f"{action} evaluation for Video: {video_file} (ID: {item_id})")
            # Don't automatically move to next item when editing
            return
        
        # Check if we've evaluated all items
        if self.all_items_evaluated():
            print("All items evaluated! current index:", self.current_index, "data length:", len(self.data))
            print(f"Evaluated IDs: {sorted(self.evaluated_ids)}")
            messagebox.showinfo("Evaluation Finished", "All items have been evaluated!")
            self.display_summary()
            return
        
        # Move to next item quickly without waiting for file operations
        self.next_item()
    
    def next_item(self):
        """Find and display the next unevaluated item"""
        # Save current index
        prev_index = self.current_index
        
        # Move to the next item
        self.current_index += 1
        
        # Find the next item that hasn't been evaluated yet
        while self.current_index < len(self.data):
            item_id = self.data[self.current_index].get('idx')
            if item_id is not None and item_id not in self.evaluated_ids:
                break
            self.current_index += 1
        
        # If we found an unevaluated item, display it
        if self.current_index < len(self.data):
            # If we preloaded this item, we can display it immediately
            if self.next_item_index == self.current_index:
                self.display_preloaded_item()
            else:
                self.display_current_item()
        else:
            # If we've gone through all items but there are still some unevaluated ones
            # (can happen with duplicate video files), go back to the beginning
            self.current_index = 0
            while self.current_index < len(self.data):
                item_id = self.data[self.current_index].get('idx')
                if item_id is not None and item_id not in self.evaluated_ids:
                    break
                self.current_index += 1
            
            if self.current_index < len(self.data):
                self.display_current_item()
            else:
                # This should not happen if all_items_evaluated() is working correctly
                print("Inconsistency detected! current index:", self.current_index, "data length:", len(self.data))
                print(f"Evaluated IDs: {sorted(self.evaluated_ids)}")
                messagebox.showinfo("Last Item Completed", "All items have been evaluated!")
                self.display_summary()
        
        # Start preloading the next item if not everything is evaluated
        if not self.all_items_evaluated():
            self.preload_next_item()
    
    def preload_next_item(self):
        """Preload the next unevaluated item's frames"""
        self.next_item_index = None
        
        # Find the next item to preload
        next_index = self.current_index + 1
        while next_index < len(self.data):
            item_id = self.data[next_index].get('idx')
            if item_id is not None and item_id not in self.evaluated_ids:
                self.next_item_index = next_index
                break
            next_index += 1
        
        # If we wrapped around, look from the beginning
        if self.next_item_index is None and next_index >= len(self.data):
            next_index = 0
            while next_index < self.current_index:
                item_id = self.data[next_index].get('idx')
                if item_id is not None and item_id not in self.evaluated_ids:
                    self.next_item_index = next_index
                    break
                next_index += 1
        
        # Preload frames if we found a next item
        if self.next_item_index is not None:
            next_video_file = self.data[self.next_item_index].get('video_file', '')
            self.root.after(100, lambda: self.preload_frames(next_video_file))
    
    def preload_frames(self, video_file):
        """Preload frames in background - just cache the path since we have preprocessed frames"""
        if video_file and video_file not in self.frame_cache:
            try:
                # Check if preprocessed frames exist
                video_name = os.path.splitext(os.path.basename(video_file))[0]
                video_frames_dir = os.path.join(self.frames_dir, video_name)
                
                if os.path.exists(video_frames_dir):
                    frame_files = sorted(glob.glob(os.path.join(video_frames_dir, "*.jpg")))
                    if frame_files:
                        self.frame_cache[video_file] = frame_files
                        return
                
                # If not in preprocessed frames, check standard locations
                frames_pattern = os.path.join(os.path.dirname(video_file), os.path.basename(video_file).split('.')[0] + '_*.jpg')
                frame_files = sorted(glob.glob(frames_pattern))
                
                # If no frames found with the pattern, try looking in a frames directory
                if not frame_files:
                    frames_pattern = os.path.join("frames", video_name, "*.jpg")
                    frame_files = sorted(glob.glob(frames_pattern))
                
                # If frames found, cache them
                if frame_files:
                    # We'll cache the file paths rather than loading all images to save memory
                    self.frame_cache[video_file] = frame_files
                elif os.path.isfile(video_file) and video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    # We'll handle video extraction during actual display
                    self.frame_cache[video_file] = "VIDEO_FILE"
            except Exception as e:
                print(f"Error preloading frames: {e}")
    
    def display_preloaded_item(self):
        """Display the preloaded item quickly"""
        if self.current_index >= len(self.data):
            return
        
        # Use the display_item_at_index method for consistency
        self.display_item_at_index(self.current_index)
    
    def setup_text_comparison(self, current_item):
        """Set up the text comparison boxes"""
        # Randomize which side gets the original vs translated text
        self.left_is_original = random.choice([True, False])
        
        # Get the motion texts
        object_motion = current_item.get('object_motion', '')
        camera_motion = current_item.get('camera_motion', '')
        original_object_motion = current_item.get('original_object_motion', '')
        original_camera_motion = current_item.get('original_camera_motion', '')
        
        # Combine object and camera motion for display
        original_text = f"Object Motion:\n{original_object_motion}\n\nCamera Motion:\n{original_camera_motion}"
        translated_text = f"Object Motion:\n{object_motion}\n\nCamera Motion:\n{camera_motion}"
        
        # Update text boxes
        self.left_text.delete(1.0, tk.END)
        self.right_text.delete(1.0, tk.END)
        
        if self.left_is_original:
            self.left_text.insert(tk.END, original_text)
            self.right_text.insert(tk.END, translated_text)
        else:
            self.left_text.insert(tk.END, translated_text)
            self.right_text.insert(tk.END, original_text)
    
    def save_result(self, result):
        try:
            with open(self.results_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save result: {str(e)}")
    
    def update_progress_label(self):
        if self.data:
            # Count how many unique IDs are in the data
            all_ids = {item.get('idx') for item in self.data if 'idx' in item}
            completed = len(self.evaluated_ids)
            total = len(all_ids)
            self.progress_label.config(text=f"{completed}/{total} evaluated")
    
    def display_summary(self):
        if not self.results:
            return
            
        original_count = sum(1 for r in self.results.values() if r['chosen'] == 'original')
        translated_count = sum(1 for r in self.results.values() if r['chosen'] == 'translated')
        total = len(self.results)
        
        original_percent = original_count/total*100 if total > 0 else 0
        translated_percent = translated_count/total*100 if total > 0 else 0
        
        # Generate timestamp for the summary
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create summary message
        message = f"Evaluation Complete!\n\nResults:\n"
        message += f"Original preferred: {original_count} ({original_percent:.1f}%)\n"
        message += f"Translated preferred: {translated_count} ({translated_percent:.1f}%)\n"
        message += f"\nResults saved to {self.results_file}"
        message += f"\nSummary saved to {self.summary_file}"
        
        # Save summary to text file
        try:
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Evaluation Summary - {timestamp}\n")
                f.write(f"================================\n\n")
                f.write(f"Input file: {self.jsonl_file}\n")
                f.write(f"Total items evaluated: {total}\n\n")
                f.write(f"RESULTS:\n")
                f.write(f"Original preferred: {original_count} ({original_percent:.1f}%)\n")
                f.write(f"Translated preferred: {translated_count} ({translated_percent:.1f}%)\n\n")
                
                # Add detailed breakdown by motion type if available
                object_motion_results = self.get_results_by_type("object")
                camera_motion_results = self.get_results_by_type("camera")
                
                if object_motion_results['total'] > 0:
                    f.write(f"\nOBJECT MOTION RESULTS:\n")
                    f.write(f"  Total object motion items: {object_motion_results['total']}\n")
                    f.write(f"  Original preferred: {object_motion_results['original']} ({object_motion_results['original_percent']:.1f}%)\n")
                    f.write(f"  Translated preferred: {object_motion_results['translated']} ({object_motion_results['translated_percent']:.1f}%)\n")
                
                if camera_motion_results['total'] > 0:
                    f.write(f"\nCAMERA MOTION RESULTS:\n")
                    f.write(f"  Total camera motion items: {camera_motion_results['total']}\n")
                    f.write(f"  Original preferred: {camera_motion_results['original']} ({camera_motion_results['original_percent']:.1f}%)\n")
                    f.write(f"  Translated preferred: {camera_motion_results['translated']} ({camera_motion_results['translated_percent']:.1f}%)\n")
                
        except Exception as e:
            print(f"Error saving summary: {e}")
            message += f"\nError saving summary file: {str(e)}"
        
        messagebox.showinfo("Summary", message)
    
    def get_results_by_type(self, motion_type):
        """Calculate results specifically for object or camera motion types"""
        # Identify items by type
        type_items = []
        for idx, item in enumerate(self.data):
            # Check if this is the right type of motion
            item_id = item.get('idx')
            if item_id in self.evaluated_ids:
                if motion_type == "object" and "object_motion" in item and "original_object_motion" in item:
                    type_items.append(item_id)
                elif motion_type == "camera" and "camera_motion" in item and "original_camera_motion" in item:
                    type_items.append(item_id)
        
        # Count preferences for this type
        total = len(type_items)
        original_count = sum(1 for item_id in type_items if item_id in self.results and self.results[item_id]['chosen'] == 'original')
        translated_count = sum(1 for item_id in type_items if item_id in self.results and self.results[item_id]['chosen'] == 'translated')
        
        # Calculate percentages
        original_percent = (original_count / total * 100) if total > 0 else 0
        translated_percent = (translated_count / total * 100) if total > 0 else 0
        
        return {
            'total': total,
            'original': original_count,
            'translated': translated_count,
            'original_percent': original_percent,
            'translated_percent': translated_percent
        }
    
    def on_closing(self):
        # Clean up temporary directory if it exists
        self.cleanup_temp_dir()
        
        # Clear frame cache to free memory
        self.frame_cache.clear()
        
        self.root.destroy()
        
    def cleanup_temp_dir(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                for file in os.listdir(self.temp_dir):
                    os.remove(os.path.join(self.temp_dir, file))
                os.rmdir(self.temp_dir)
                self.temp_dir = None
            except Exception as e:
                print(f"Error cleaning up temp directory: {e}")
    
    def go_to_previous_item(self):
        """Navigate to the previous item in the evaluation history"""
        # Check if we have history to go back to
        if len(self.history) <= 1:
            messagebox.showinfo("Navigation", "No previous items available")
            return
            
        # Remove current item from history
        if self.history:
            self.history.pop()
            
        # Get the previous index
        if self.history:
            previous_index = self.history[-1]
            self.current_index = previous_index
            # Display the item without adding to history
            self.display_item_at_index(self.current_index, add_to_history=False)
        else:
            messagebox.showinfo("Navigation", "Beginning of history reached")
    
    def preprocess_all_videos(self):
        """Extract frames from all videos in advance"""
        # Get unique video files
        video_files = set()
        for item in self.data:
            video_file = item.get('video_file')
            if video_file:
                video_files.add(video_file)
        
        if not video_files:
            messagebox.showinfo("Info", "No video files found in the data.")
            return
            
        # Create progress dialog
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Processing Videos")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        progress_label = ttk.Label(progress_window, text="Processing videos...", font=self.label_font)
        progress_label.pack(pady=10)
        
        progress_bar = ttk.Progressbar(progress_window, length=350, mode="determinate")
        progress_bar.pack(pady=10)
        
        status_label = ttk.Label(progress_window, text="", font=self.label_font)
        status_label.pack(pady=10)
        
        # Set up progress bar
        total_videos = len(video_files)
        progress_bar["maximum"] = total_videos
        progress_bar["value"] = 0
        
        # Force window update
        progress_window.update()
        
        # Process each video
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        for video_file in video_files:
            try:
                # Update status
                status_label.config(text=f"Processing: {os.path.basename(video_file)}")
                progress_window.update()
                
                # Create directory for this video if it doesn't exist
                video_name = os.path.splitext(os.path.basename(video_file))[0]
                video_frames_dir = os.path.join(self.frames_dir, video_name)
                
                # Skip if frames already exist
                if os.path.exists(video_frames_dir) and len(os.listdir(video_frames_dir)) >= 8:
                    print(f"Frames for {video_name} already exist, skipping.")
                    skipped_count += 1
                else:
                    if not os.path.exists(video_frames_dir):
                        os.makedirs(video_frames_dir)
                    
                    # Extract frames if the video file exists
                    if os.path.isfile(video_file) and video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        with VideoFileClip(video_file) as clip:
                            # Calculate how many frames to extract (16 frames evenly distributed)
                            duration = clip.duration
                            frame_times = [duration * i / 16 for i in range(16)]
                            
                            # Extract frames at the calculated times
                            for i, t in enumerate(frame_times):
                                frame_path = os.path.join(video_frames_dir, f"frame_{i:03d}.jpg")
                                clip.save_frame(frame_path, t=t)
                    else:
                        print(f"Video file not found or not supported: {video_file}")
                        error_count += 1
                        continue
                    
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing video {video_file}: {e}")
                error_count += 1
            
            # Update progress
            progress_bar["value"] = processed_count + skipped_count + error_count
            progress_window.update()
        
        # Final status
        status_text = f"Complete! Processed: {processed_count}, Skipped: {skipped_count}, Errors: {error_count}"
        status_label.config(text=status_text)
        
        # Add close button
        ttk.Button(
            progress_window, 
            text="Close", 
            command=progress_window.destroy
        ).pack(pady=10)

def main():
    root = tk.Tk()
    app = VideoComparisonApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 