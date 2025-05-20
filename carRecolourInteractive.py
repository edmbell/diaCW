# python carRecolourInteractive.py

import imageio
import numpy as np
import os
from tkinter import filedialog, Tk, colorchooser

def get_color_mask(frame, rgb, tolerance=15):
    """
    Creates a mask for pixels within a specified tolerance of the given RGB color.
    """
    r_min, r_max = rgb[0] - tolerance, rgb[0] + tolerance
    g_min, g_max = rgb[1] - tolerance, rgb[1] + tolerance
    b_min, b_max = rgb[2] - tolerance, rgb[2] + tolerance

    return (
        (frame[:, :, 0] >= r_min) & (frame[:, :, 0] <= r_max) &
        (frame[:, :, 1] >= g_min) & (frame[:, :, 1] <= g_max) &
        (frame[:, :, 2] >= b_min) & (frame[:, :, 2] <= b_max)
    )

def recolor_car_interactive():
    """
    Recolors the car in a racing GIF by detecting a set of reddish shades used in the sprite and
    replacing them with a new user-selected color via GUI.
    """

    # GUI dialogs
    root = Tk()
    root.withdraw()

    input_path = filedialog.askopenfilename(
        title="Select a CarRacing GIF",
        filetypes=[("GIF files", "*.gif")]
    )
    if not input_path:
        print("❌ No file selected.")
        return

    color_code, hex_code = colorchooser.askcolor(title="Choose new car color")
    if color_code is None:
        print("❌ No color selected.")
        return
    new_color = tuple(int(c) for c in color_code)
    print(f"🎨 Selected new color: {new_color}")

    # Red hues used in the car outline and fill (shaded variants)
    car_red_shades = [
        (203, 0, 1),    # #cb0001 (main fill)
        (171, 38, 36),  # #ab2624
        (148, 55, 55),  # #943737
        (157, 47, 46),  # #9d2f2e
        (193, 19, 62)  # c1133e
    ]

    reader = imageio.get_reader(input_path)
    frames = []

    print("🔄 Recoloring frames...")
    for frame in reader:
        # Build a combined mask for all red tones
        full_mask = np.zeros(frame.shape[:2], dtype=bool)
        for red_rgb in car_red_shades:
            full_mask |= get_color_mask(frame, red_rgb, tolerance=15)

        recolored = frame.copy()
        recolored[full_mask] = new_color
        frames.append(recolored)

    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_recolored.gif"
    imageio.mimsave(output_path, frames, duration=1/30)

    print(f"✅ Saved recolored GIF to: {output_path}")

# Optional: Uncomment to run directly
if __name__ == "__main__":
    recolor_car_interactive()
