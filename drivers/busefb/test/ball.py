#! /usr/bin/python3

import numpy as np
import time
import random

WIDTH = 32
HEIGHT = 16

def clear_frame():
    return np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

def draw_pixel(frame, x, y):
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        frame[y, x] = 1

def draw_ball(frame, cx, cy, radius=1):
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                draw_pixel(frame, cx + dx, cy + dy)

def render_frame(frame):
    packed = np.packbits(frame, axis=1, bitorder='little')
    with open("/dev/fb0", "wb") as fb:
        fb.write(packed.tobytes())

# Ball state
x = random.randint(2, WIDTH - 3)
y = random.randint(2, HEIGHT - 3)
vx = random.choice([-1, 1])
vy = random.choice([-1, 1])
radius =1

try:
    while True:
        frame = clear_frame()
        draw_ball(frame, x, y, radius)
        render_frame(frame)

        # Update ball position
        x += vx
        y += vy

        # Bounce off edges
        if x - radius <= 0 or x + radius >= WIDTH - 1:
            vx = -vx
        if y - radius <= 0 or y + radius >= HEIGHT - 1:
            vy = -vy

        time.sleep(0.05)
except KeyboardInterrupt:
    frame=clear_frame()
