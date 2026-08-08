#!/usr/bin/env python3
"""Test patterns for BUSE 1bpp framebuffer debugging
   Auto-detects display dimensions from the framebuffer device.
"""

import sys
import fcntl
import struct

def detect_fb(dev='/dev/fb0'):
    """Read framebuffer dimensions from the kernel via ioctl."""
    FBIOGET_VSCREENINFO = 0x4600
    with open(dev, 'rb') as f:
        data = fcntl.ioctl(f, FBIOGET_VSCREENINFO, b'\x00' * 160)
    xres, yres = struct.unpack('II', data[:8])
    return xres, yres

FB_DEV = '/dev/fb0'
WIDTH, HEIGHT = detect_fb(FB_DEV)
FB_SIZE = WIDTH * HEIGHT // 8
print(f"Detected framebuffer: {WIDTH}x{HEIGHT} ({FB_SIZE} bytes)")

def pixel_to_byte_bit(x, y):
    """Convert x,y to byte index and bit position (no mirroring)"""
    idx = y * WIDTH + x
    return idx >> 3, idx & 7

def create_fb():
    return bytearray(FB_SIZE)

def set_pixel(fb, x, y):
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        byte_idx, bit = pixel_to_byte_bit(x, y)
        fb[byte_idx] |= (1 << bit)

def pattern_left_half():
    """Light up left half of the display"""
    fb = create_fb()
    for y in range(HEIGHT):
        for x in range(WIDTH // 2):
            set_pixel(fb, x, y)
    return fb

def pattern_right_half():
    """Light up right half of the display"""
    fb = create_fb()
    for y in range(HEIGHT):
        for x in range(WIDTH // 2, WIDTH):
            set_pixel(fb, x, y)
    return fb

def pattern_all_on():
    """All pixels on"""
    fb = create_fb()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            set_pixel(fb, x, y)
    return fb

def pattern_vertical_stripes():
    """Vertical stripes every 8 columns"""
    fb = create_fb()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if (x // 8) % 2 == 0:
                set_pixel(fb, x, y)
    return fb

def pattern_horizontal_stripes():
    """Horizontal stripes every 2 rows"""
    fb = create_fb()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if y % 2 == 0:
                set_pixel(fb, x, y)
    return fb

def pattern_panel_id():
    """Different pattern for each panel to identify order"""
    fb = create_fb()
    panel_width = 32
    for y in range(HEIGHT):
        for x in range(WIDTH):
            panel = x // panel_width
            if panel == 0:
                set_pixel(fb, x, y)
            elif panel == 1 and y < HEIGHT // 2:
                set_pixel(fb, x, y)
            elif panel == 2 and y >= HEIGHT // 2:
                set_pixel(fb, x, y)
            elif panel == 3 and (x % panel_width) < panel_width // 2:
                set_pixel(fb, x, y)
            elif panel >= 4:
                set_pixel(fb, x, y)
    return fb

def pattern_single_column(col):
    """Light up a single column"""
    fb = create_fb()
    for y in range(HEIGHT):
        set_pixel(fb, col, y)
    return fb

def pattern_column_march():
    """Single column - specify column as arg"""
    col = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    return pattern_single_column(col)

def pattern_group_test():
    """Show which group each column belongs to (group 0 only)"""
    fb = create_fb()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if x % 4 == 0:
                set_pixel(fb, x, y)
    return fb

def pattern_group_n(grp_num):
    """Show all columns belonging to a specific group"""
    fb = create_fb()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if x % 4 == grp_num:
                set_pixel(fb, x, y)
    return fb

def pattern_diagonal():
    """Diagonal lines across the display"""
    fb = create_fb()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if (x + y) % 8 == 0:
                set_pixel(fb, x, y)
    return fb

def pattern_diagonal_thick():
    """Thicker diagonal lines"""
    fb = create_fb()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if (x + y) % 8 < 2:
                set_pixel(fb, x, y)
    return fb

def pattern_diagonal_reverse():
    """Diagonal lines going the other way"""
    fb = create_fb()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if (x - y) % 8 == 0:
                set_pixel(fb, x, y)
    return fb

def pattern_crosshatch():
    """Crosshatch pattern (both diagonals)"""
    fb = create_fb()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if (x + y) % 8 == 0 or (x - y) % 8 == 0:
                set_pixel(fb, x, y)
    return fb

def pattern_single_diagonal():
    """Single diagonal line - specify offset as arg"""
    offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    fb = create_fb()
    for y in range(HEIGHT):
        x = (y + offset) % WIDTH
        set_pixel(fb, x, y)
    return fb

def pattern_checkerboard():
    """Checkerboard pattern"""
    fb = create_fb()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if (x + y) % 2 == 0:
                set_pixel(fb, x, y)
    return fb

def pattern_grid():
    """Grid pattern (every 8 pixels)"""
    fb = create_fb()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if x % 8 == 0 or y % 8 == 0:
                set_pixel(fb, x, y)
    return fb

patterns = {
    'left': pattern_left_half,
    'right': pattern_right_half,
    'all': pattern_all_on,
    'vstripe': pattern_vertical_stripes,
    'hstripe': pattern_horizontal_stripes,
    'panels': pattern_panel_id,
    'col': pattern_column_march,
    'group': pattern_group_test,
    'g0': lambda: pattern_group_n(0),
    'g1': lambda: pattern_group_n(1),
    'g2': lambda: pattern_group_n(2),
    'g3': lambda: pattern_group_n(3),
    'diag': pattern_diagonal,
    'diag2': pattern_diagonal_thick,
    'diagr': pattern_diagonal_reverse,
    'cross': pattern_crosshatch,
    'line': pattern_single_diagonal,
    'check': pattern_checkerboard,
    'grid': pattern_grid,
}

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in patterns:
        print(f"Usage: {sys.argv[0]} <pattern> [arg]")
        print(f"Patterns: {', '.join(patterns.keys())}")
        print(f"\n  Display: {WIDTH}x{HEIGHT} ({FB_SIZE} bytes)")
        print("\n  left    - left half of display")
        print("  right   - right half of display")
        print("  all     - all pixels on")
        print("  vstripe - vertical stripes")
        print("  hstripe - horizontal stripes")
        print("  panels  - identify each panel")
        print("  col N   - single column N")
        print("  group   - show group 0 columns")
        print("  g0-g3   - show group 0/1/2/3 columns")
        print("  diag    - diagonal lines")
        print("  diag2   - thick diagonal lines")
        print("  diagr   - reverse diagonal")
        print("  cross   - crosshatch")
        print("  line N  - single diagonal at offset N")
        print("  check   - checkerboard")
        print("  grid    - grid pattern")
        sys.exit(1)

    fb = patterns[sys.argv[1]]()

    with open(FB_DEV, 'wb') as f:
        f.write(fb)

    print(f"Wrote {len(fb)} bytes to {FB_DEV}")
