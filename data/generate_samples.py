import sys
import os
import torch
import numpy as np
import textwrap
from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path to handle imports
curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
sys.path.append(parent_dir)

from data.dataset_interpert import StellarQuestionsDataset

def draw_box(text, title=None, width=80):
    """Draw a text box around the content for terminal output"""
    lines = text.split('\n')
    
    # Text wrapping for terminal
    wrapped_lines = []
    for line in lines:
        if not line:
            wrapped_lines.append("")
            continue
            
        while len(line) > width - 4:
            wrapped_lines.append(line[:width-4])
            line = line[width-4:]
        wrapped_lines.append(line)
        
    print("+" + "-" * (width - 2) + "+")
    if title:
        # Centered title
        t_len = len(title)
        if t_len > width - 4:
             title = title[:width-7] + "..."
             t_len = len(title)
        padding = (width - 2 - t_len) // 2
        print("|" + " " * padding + title + " " * (width - 2 - padding - t_len) + "|")
        print("+" + "-" * (width - 2) + "+")
        
    for line in wrapped_lines:
        print(f"| {line:<{width-4}} |")
    print("+" + "-" * (width - 2) + "+")

def create_image_from_text(text, output_path, title=None):
    # Image settings
    padding = 60
    line_spacing = 15
    font_size = 24
    title_font_size = 32
    background_color = (255, 255, 255)  # White
    text_color = (0, 0, 0)  # Black
    box_color = (240, 240, 240) # Light grey for boxes
    border_color = (100, 100, 100)
    
    # Try to load a known sans-serif font
    # Common linux fonts
    font_candidates = [
        "/usr/share/fonts/google-droid-sans-fonts/DroidSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf"
    ]
    
    font_path = None
    for f in font_candidates:
        if os.path.exists(f):
            font_path = f
            break
            
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
            # Try to find bold version or reuse normal
            bold_path = font_path.replace("Regular", "Bold").replace("-R", "-B").replace("Sans.ttf", "Sans-Bold.ttf")
            if os.path.exists(bold_path):
                title_font = ImageFont.truetype(bold_path, title_font_size)
                bold_font = ImageFont.truetype(bold_path, font_size)
            else:
                title_font = ImageFont.truetype(font_path, title_font_size)
                bold_font = ImageFont.truetype(font_path, font_size)
        else:
             # Fallback to default load if no path found (this often fails to look good or fails on size)
             raise OSError("No font found")
             
    except Exception as e:
        print(f"Warning: Could not load custom font ({e}), using default PIL font.")
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
        bold_font = ImageFont.load_default()

    # Calculate image size
    # Width is fixed, height is dynamic
    img_width = 1200
    max_text_width = img_width - 2 * padding
    
    # Prepare lines
    draw_ops = [] # List of (text, font, is_title, is_header)
    
    if title:
        draw_ops.append((title, title_font, True, False))
        draw_ops.append(("", font, False, False)) # Spacer
        
    lines = text.split('\n')
    
    try:
        avg_char_width = font.getbbox("A")[2] 
    except AttributeError:
        # Older PIL versions
        avg_char_width = font.getsize("A")[0]
        
    if avg_char_width == 0: avg_char_width = 10 # Safety
    
    wrap_width = int(max_text_width / (avg_char_width * 0.95)) # approximation

    for line in lines:
        if line.startswith("Main Question:") or line.startswith("Main Answer:") or \
           line.startswith("FQ") or line.startswith("FA") or "Follow-ups" in line or "Follow-up Conversation" in line:
            # Headers
            draw_ops.append((line, bold_font, False, True))
        else:
            # Wrap content lines
            if not line.strip():
                draw_ops.append(("", font, False, False))
                continue
                
            wrapped = textwrap.wrap(line, width=wrap_width)
            for w in wrapped:
                draw_ops.append((w, font, False, False))
    
    # Calculate Height
    dummy_img = Image.new('RGB', (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    
    current_h = padding
    for text_line, fnt, is_title, is_header in draw_ops:
        try:
            bbox = dummy_draw.textbbox((0, 0), text_line, font=fnt)
            h = bbox[3] - bbox[1]
        except AttributeError:
             # Older PIL
             w, h = dummy_draw.textsize(text_line, font=fnt)
             
        current_h += h + line_spacing
        if is_title or is_header:
            current_h += 10 # Extra space after headers
            
    img_height = current_h + padding
    
    # Draw proper image
    img = Image.new('RGB', (img_width, img_height), background_color)
    draw = ImageDraw.Draw(img)
    
    current_h = padding
    
    for text_line, fnt, is_title, is_header in draw_ops:
        if is_title:
             # Center title
             try:
                 bbox = draw.textbbox((0, 0), text_line, font=fnt)
                 w = bbox[2] - bbox[0]
                 h_line = bbox[3] - bbox[1]
             except AttributeError:
                 w, h_line = draw.textsize(text_line, font=fnt)
                 bbox = (0, 0, w, h_line) # Dummy
                 
             x = (img_width - w) // 2
             draw.text((x, current_h), text_line, font=fnt, fill=text_color)
             
             # Underline title
             draw.line([(x, current_h + h_line + 5), (x + w, current_h + h_line + 5)], fill=text_color, width=2)
             
        else:
             draw.text((padding, current_h), text_line, font=fnt, fill=text_color)
        
        try:
            bbox = draw.textbbox((0, 0), text_line, font=fnt)
            h = bbox[3] - bbox[1]
        except AttributeError:
            w, h = draw.textsize(text_line, font=fnt)
            
        current_h += h + line_spacing
        if is_header:
            current_h += 10

    # Draw border
    draw.rectangle([(10, 10), (img_width-10, img_height-10)], outline=border_color, width=2)

    print(f"Saving image to {output_path}")
    img.save(output_path)


def generate_samples():
    json_file = "/home/ilay.kamai/work/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json"
    followup_json_file = "/home/ilay.kamai/work/TalkingLatents/data/dataset/stellar_descriptions_questions.json"
    output_dir = "/home/ilay.kamai/work/TalkingLatents/figs"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if files exist
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found.")
        # Fallback
        json_file = "/home/ilay.kamai/work/TalkingLatents/data/dataset/stellar_descriptions.json"
        
    if not os.path.exists(json_file):
         print(f"Error: Could not find dataset json file.")
         return

    print(f"Using JSON file: {json_file}")

    # Initialize dataset
    dataset = StellarQuestionsDataset(
        json_file=json_file,
        followup_json_file=followup_json_file,
        split='train',
        enable_followup=True,
        followup_prob=1.0,  # Force followups if possible
        tokenizer_path="/home/ilay.kamai/work/.llama/Llama3.2-1B/tokenizer.model", 
        random_state=42
    )
    
    print(f"Dataset initialized with {len(dataset)} samples.")
    
    samples_to_generate = 5
    count = 0
    
    for i in range(len(dataset)):
        if count >= samples_to_generate:
            break
            
        try:
            item = dataset[i]
            
            # Extract content
            q_main = item['input_text']
            a_main = item['target_text']
            followups = item['followup_turns']
            
            # Skip if no followups generated (unless we want to see empty ones too)
            if not followups:
                continue

            content = f"Main Question:\n{q_main}\n\nMain Answer:\n{a_main}"
            
            content += "\n\n" + "Follow-up Conversation" + "\n"
            for idx, turn in enumerate(followups):
                # turn is tuple (Q, A) or dict
                if isinstance(turn, dict):
                    fq = turn.get('question', '')
                    fa = turn.get('answer', '')
                elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
                    fq = turn[0]
                    fa = turn[1]
                else:
                    fq = str(turn)
                    fa = "?"
                    
                content += f"\nFQ{idx+1}: {fq}\nFA{idx+1}: {fa}"
            
            obsid = item.get('obsid', 'NA')
            
            # Print to stdout
            print("\n")
            draw_box(content, title=f"Sample {count+1} (ObsID: {obsid})")
            
            # Save to PNG
            filename = os.path.join(output_dir, f"sample_{obsid}_idx{i}.png")
            create_image_from_text(content, filename, title=f"Sample ObsID: {obsid}")
            
            count += 1
            
        except Exception as e:
            print(f"Error generating sample {i}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    generate_samples()
