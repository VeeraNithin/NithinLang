# src/nithinlang/game_engine.py
"""
NithinLang Built-in 2D Game Engine
=====================================
A lightweight 2D game framework built on pygame (with a turtle fallback).

Injected functions
------------------
game_start(width, height, title, fps)  — Initialise the game window
game_stop()                            — Close the window and quit
game_draw(shape, x, y, **kwargs)       — Draw a primitive or sprite
game_clear(color)                      — Clear the screen
game_loop(update_fn, draw_fn)          — Start the main game loop
game_color(r, g, b)                    → (r, g, b) colour tuple helper
game_text(text, x, y, size, color)     — Render text on screen
game_sprite(image_path, x, y, scale)  — Draw an image sprite
game_key(key_name)                     → bool  (is key currently held?)
game_collide(rect1, rect2)             → bool  (do rectangles overlap?)
game_sound(path, volume, loop)         — Play a sound
game_fps(fps)                          — Change target FPS at runtime
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Try pygame first, fall back to turtle
# ---------------------------------------------------------------------------

try:
    import pygame
    import pygame.mixer
    _PYGAME = True
    _TURTLE = False
except ImportError:
    _PYGAME = False
    try:
        import turtle as _turtle_mod
        _TURTLE = True
    except ImportError:
        _TURTLE = False

# Colour type alias
Color = Union[Tuple[int, int, int], str]


# ---------------------------------------------------------------------------
# GameEngine
# ---------------------------------------------------------------------------

class GameEngine:
    """
    Manages a single game window and all 2D rendering / input / audio.

    The engine is stateful — one instance per NithinLang program.
    """

    def __init__(self) -> None:
        self._screen          : Optional[Any]   = None
        self._clock           : Optional[Any]   = None
        self._running         : bool            = False
        self._fps             : int             = 60
        self._bg_color        : Tuple[int,int,int] = (30, 30, 46)  # dark navy
        self._width           : int             = 800
        self._height          : int             = 600
        self._title           : str             = "NithinLang Game"
        self._font_cache      : Dict[int, Any]  = {}
        self._image_cache     : Dict[str, Any]  = {}
        self._sound_cache     : Dict[str, Any]  = {}
        self._key_state       : Dict[str, bool] = {}

        # Turtle fallback state
        self._turtle_screen   : Optional[Any]  = None
        self._turtle_pen      : Optional[Any]  = None

    # =========================================================================
    # Initialisation
    # =========================================================================

    def game_start(
        self,
        width  : int  = 800,
        height : int  = 600,
        title  : str  = "NithinLang Game",
        fps    : int  = 60,
        bg     : Color = (30, 30, 46),
    ) -> None:
        """
        Initialise the game window.

        Args:
            width  : Window width in pixels.
            height : Window height in pixels.
            title  : Window title / caption.
            fps    : Target frames per second.
            bg     : Background colour (R, G, B) or colour name string.

        Example (telugu):
            game_start(800, 600, "Nithin Aata", 60)
        """
        self._width    = width
        self._height   = height
        self._title    = title
        self._fps      = fps
        self._bg_color = self._parse_color(bg)

        if _PYGAME:
            pygame.init()
            pygame.display.set_caption(title)
            self._screen  = pygame.display.set_mode((width, height))
            self._clock   = pygame.time.Clock()
            self._running = True
            pygame.mixer.init()
            self._screen.fill(self._bg_color)
            pygame.display.flip()

        elif _TURTLE:
            self._turtle_screen = _turtle_mod.Screen()
            self._turtle_screen.setup(width, height)
            self._turtle_screen.title(title)
            bg_hex = (
                "#{:02x}{:02x}{:02x}".format(*self._bg_color)
                if isinstance(self._bg_color, tuple)
                else str(self._bg_color)
            )
            self._turtle_screen.bgcolor(bg_hex)
            self._turtle_pen = _turtle_mod.Turtle()
            self._turtle_pen.speed(0)
            self._turtle_pen.hideturtle()
            self._running = True

        else:
            raise RuntimeError(
                "game_start: Neither pygame nor turtle is available. "
                "Install pygame: pip install pygame"
            )

    def game_stop(self) -> None:
        """Close the game window and clean up resources."""
        self._running = False
        if _PYGAME and pygame.get_init():
            pygame.quit()
        elif _TURTLE and self._turtle_screen:
            try:
                self._turtle_screen.bye()
            except Exception:
                pass

    # =========================================================================
    # Rendering
    # =========================================================================

    def game_clear(self, color: Optional[Color] = None) -> None:
        """
        Clear the screen to `color` (default: background colour set at start).
        """
        c = self._parse_color(color) if color is not None else self._bg_color

        if _PYGAME and self._screen:
            self._screen.fill(c)
        elif _TURTLE and self._turtle_pen:
            self._turtle_pen.clear()

    def game_draw(
        self,
        shape  : str,
        x      : int,
        y      : int,
        *,
        width  : int   = 50,
        height : int   = 50,
        radius : int   = 25,
        color  : Color = (255, 255, 255),
        fill   : bool  = True,
        border : int   = 0,
    ) -> None:
        """
        Draw a primitive shape on the screen.

        Args:
            shape  : "rect", "circle", "ellipse", "line", "point",
                     "triangle".
            x, y   : Top-left corner (rect/ellipse) or centre (circle).
            width  : Width for rects / ellipses.
            height : Height for rects / ellipses.
            radius : Radius for circles.
            color  : Fill colour.
            fill   : Filled (True) or outline (False).
            border : Border thickness when fill=False.

        Example (english):
            game_draw("circle", 400, 300, radius=40, color=(255, 0, 0))
            game_draw("rect", 100, 100, width=80, height=40, color=(0,200,0))
        """
        c = self._parse_color(color)

        if _PYGAME and self._screen:
            shape_l = shape.lower()
            if shape_l == "rect":
                rect = pygame.Rect(x, y, width, height)
                if fill:
                    pygame.draw.rect(self._screen, c, rect)
                else:
                    pygame.draw.rect(self._screen, c, rect, max(1, border))
            elif shape_l == "circle":
                if fill:
                    pygame.draw.circle(self._screen, c, (x, y), radius)
                else:
                    pygame.draw.circle(self._screen, c, (x, y), radius, max(1, border))
            elif shape_l == "ellipse":
                rect = pygame.Rect(x, y, width, height)
                if fill:
                    pygame.draw.ellipse(self._screen, c, rect)
                else:
                    pygame.draw.ellipse(self._screen, c, rect, max(1, border))
            elif shape_l == "line":
                pygame.draw.line(
                    self._screen, c,
                    (x, y),
                    (x + width, y + height),
                    max(1, border or 2)
                )
            elif shape_l == "point":
                pygame.draw.circle(self._screen, c, (x, y), max(1, border or 2))
            elif shape_l == "triangle":
                pts = [
                    (x, y + height),
                    (x + width // 2, y),
                    (x + width, y + height),
                ]
                if fill:
                    pygame.draw.polygon(self._screen, c, pts)
                else:
                    pygame.draw.polygon(self._screen, c, pts, max(1, border))
            else:
                raise ValueError(
                    f"game_draw: Unknown shape '{shape}'. "
                    "Choose: rect, circle, ellipse, line, point, triangle"
                )

        elif _TURTLE and self._turtle_pen:
            pen = self._turtle_pen
            hex_c = "#{:02x}{:02x}{:02x}".format(*c) if isinstance(c, tuple) else str(c)
            pen.penup()
            pen.goto(x - self._width // 2, self._height // 2 - y)  # pygame→turtle coords
            pen.pendown()
            pen.color(hex_c)
            if fill:
                pen.begin_fill()

            shape_l = shape.lower()
            if shape_l == "rect":
                for _ in range(2):
                    pen.forward(width)
                    pen.left(90)
                    pen.forward(height)
                    pen.left(90)
            elif shape_l == "circle":
                pen.circle(radius)
            else:
                pen.forward(width)

            if fill:
                pen.end_fill()

    def game_text(
        self,
        text  : str,
        x     : int,
        y     : int,
        size  : int   = 24,
        color : Color = (255, 255, 255),
        bold  : bool  = False,
    ) -> None:
        """
        Render text on the game screen.

        Example:
            game_text("Score: 100", 10, 10, size=32, color=(255,215,0))
        """
        c = self._parse_color(color)

        if _PYGAME and self._screen:
            if not pygame.font.get_init():
                pygame.font.init()
            cache_key = (size, bold)
            if cache_key not in self._font_cache:
                self._font_cache[cache_key] = pygame.font.SysFont(
                    "Arial", size, bold=bold
                )
            font    = self._font_cache[cache_key]
            surface = font.render(text, True, c)
            self._screen.blit(surface, (x, y))

        elif _TURTLE and self._turtle_pen:
            pen     = self._turtle_pen
            hex_c   = "#{:02x}{:02x}{:02x}".format(*c) if isinstance(c, tuple) else str(c)
            pen.penup()
            pen.goto(x - self._width // 2, self._height // 2 - y)
            pen.color(hex_c)
            style = ("Arial", size, "bold" if bold else "normal")
            pen.write(text, font=style)

    def game_sprite(
        self,
        image_path : str,
        x          : int,
        y          : int,
        scale      : float = 1.0,
    ) -> None:
        """
        Draw an image sprite at position (x, y).

        Args:
            image_path : Path to PNG/JPG/BMP image file.
            x, y       : Top-left pixel position.
            scale      : Scale factor (1.0 = original size).

        Example:
            game_sprite("player.png", 100, 200, scale=0.5)
        """
        if not _PYGAME:
            print(f"[NithinLang Game] game_sprite requires pygame.")
            return

        if image_path not in self._image_cache:
            if not os.path.isfile(image_path):
                raise FileNotFoundError(
                    f"game_sprite: Image not found: '{image_path}'"
                )
            img = pygame.image.load(image_path).convert_alpha()
            self._image_cache[image_path] = img

        img = self._image_cache[image_path]
        if scale != 1.0:
            new_w = int(img.get_width()  * scale)
            new_h = int(img.get_height() * scale)
            img   = pygame.transform.scale(img, (new_w, new_h))

        self._screen.blit(img, (x, y))

    def game_present(self) -> None:
        """Flip the display buffer (present rendered frame)."""
        if _PYGAME and self._screen:
            pygame.display.flip()

    # =========================================================================
    # Game loop
    # =========================================================================

    def game_loop(
        self,
        update_fn : Optional[Callable[[float], None]] = None,
        draw_fn   : Optional[Callable[[], None]]       = None,
    ) -> None:
        """
        Start the main game loop.

        The loop runs at self._fps until the window is closed or
        game_stop() is called.

        Args:
            update_fn : Called every frame with `dt` (seconds since last frame).
                        Use this for physics / game logic.
            draw_fn   : Called every frame after update_fn.
                        Use this for all rendering calls.

        Example:
            score = [0]

            def update(dt):
                score[0] += 1

            def draw():
                game_clear()
                game_text(f"Score: {score[0]}", 10, 10)

            game_loop(update, draw)
        """
        if not _PYGAME and not _TURTLE:
            raise RuntimeError("game_loop: No display backend available.")

        if _PYGAME:
            self._pygame_loop(update_fn, draw_fn)
        elif _TURTLE:
            self._turtle_loop(update_fn, draw_fn)

    def _pygame_loop(
        self,
        update_fn : Optional[Callable],
        draw_fn   : Optional[Callable],
    ) -> None:
        """Internal pygame event / render loop."""
        if self._screen is None:
            raise RuntimeError("game_loop: Call game_start() first.")

        last_time = time.perf_counter()
        self._running = True

        while self._running:
            # ── Event handling ────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    break
                if event.type == pygame.KEYDOWN:
                    key_name = pygame.key.name(event.key)
                    self._key_state[key_name] = True
                if event.type == pygame.KEYUP:
                    key_name = pygame.key.name(event.key)
                    self._key_state[key_name] = False

            if not self._running:
                break

            # ── Update ────────────────────────────────────────────────────
            now = time.perf_counter()
            dt  = now - last_time
            last_time = now

            if update_fn is not None:
                update_fn(dt)

            # ── Draw ──────────────────────────────────────────────────────
            if draw_fn is not None:
                draw_fn()

            pygame.display.flip()

            if self._clock:
                self._clock.tick(self._fps)

        pygame.quit()

    def _turtle_loop(
        self,
        update_fn : Optional[Callable],
        draw_fn   : Optional[Callable],
    ) -> None:
        """Simple turtle main loop (non-threaded, limited)."""
        if self._turtle_screen is None:
            raise RuntimeError("game_loop: Call game_start() first.")

        last_time = time.perf_counter()
        frame_dt  = 1.0 / self._fps

        def _tick():
            if not self._running:
                return
            now = time.perf_counter()
            dt  = now - last_time
            if update_fn:
                update_fn(dt)
            if draw_fn:
                draw_fn()
            self._turtle_screen.update()
            delay_ms = max(1, int(frame_dt * 1000))
            self._turtle_screen.ontimer(_tick, delay_ms)

        self._turtle_screen.tracer(0)
        _tick()
        _turtle_mod.mainloop()

    # =========================================================================
    # Input
    # =========================================================================

    def game_key(self, key_name: str) -> bool:
        """
        Return True if the given key is currently held down.

        Args:
            key_name : Key name string, e.g., "left", "right", "up",
                       "down", "space", "return", "a" … "z".

        Example:
            if game_key("space"):
                bullet_fire()
        """
        if _PYGAME and pygame.get_init():
            keys = pygame.key.get_pressed()
            try:
                key_id = getattr(pygame, f"K_{key_name.upper()}", None)
                if key_id is None:
                    key_id = getattr(pygame, f"K_{key_name.lower()}", None)
                if key_id is not None:
                    return bool(keys[key_id])
            except Exception:
                pass
        return self._key_state.get(key_name.lower(), False)

    # =========================================================================
    # Collision detection
    # =========================================================================

    def game_collide(
        self,
        rect1: Tuple[int, int, int, int],
        rect2: Tuple[int, int, int, int],
    ) -> bool:
        """
        Axis-Aligned Bounding Box (AABB) collision test.

        Args:
            rect1, rect2 : (x, y, width, height) tuples.

        Returns:
            True if the rectangles overlap.

        Example:
            if game_collide(player_rect, enemy_rect):
                game_text("Hit!", 300, 300)
        """
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        return not (
            x1 + w1 <= x2 or
            x2 + w2 <= x1 or
            y1 + h1 <= y2 or
            y2 + h2 <= y1
        )

    # =========================================================================
    # Audio
    # =========================================================================

    def game_sound(
        self,
        path   : str,
        volume : float = 1.0,
        loop   : bool  = False,
    ) -> None:
        """
        Play a sound file (WAV / OGG).

        Args:
            path   : Path to sound file.
            volume : Volume 0.0 – 1.0.
            loop   : If True, loop indefinitely.

        Example:
            game_sound("jump.wav", volume=0.8)
        """
        if not _PYGAME:
            print(f"[NithinLang Game] game_sound requires pygame.")
            return

        if path not in self._sound_cache:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"game_sound: Sound file not found: '{path}'"
                )
            snd = pygame.mixer.Sound(path)
            self._sound_cache[path] = snd

        snd: pygame.mixer.Sound = self._sound_cache[path]
        snd.set_volume(max(0.0, min(1.0, volume)))
        loops = -1 if loop else 0
        snd.play(loops=loops)

    # =========================================================================
    # FPS control
    # =========================================================================

    def game_fps(self, fps: int) -> None:
        """Change the target frame rate."""
        self._fps = max(1, fps)

    # =========================================================================
    # Colour helper
    # =========================================================================

    def game_color(self, r: int, g: int, b: int) -> Tuple[int, int, int]:
        """
        Create an (R, G, B) colour tuple.

        Example:
            red   = game_color(255, 0, 0)
            green = game_color(0, 255, 0)
        """
        return (
            max(0, min(255, r)),
            max(0, min(255, g)),
            max(0, min(255, b)),
        )

    # =========================================================================
    # Internal helpers
    # =========================================================================

    @staticmethod
    def _parse_color(color: Color) -> Tuple[int, int, int]:
        """Normalise a colour argument to an (R, G, B) tuple."""
        if isinstance(color, (tuple, list)) and len(color) >= 3:
            return (int(color[0]), int(color[1]), int(color[2]))
        if isinstance(color, str):
            _named: Dict[str, Tuple[int,int,int]] = {
                "red"    : (255,   0,   0),
                "green"  : (  0, 200,   0),
                "blue"   : (  0,   0, 255),
                "white"  : (255, 255, 255),
                "black"  : (  0,   0,   0),
                "yellow" : (255, 255,   0),
                "orange" : (255, 165,   0),
                "purple" : (128,   0, 128),
                "cyan"   : (  0, 255, 255),
                "pink"   : (255, 192, 203),
                "grey"   : (128, 128, 128),
                "gray"   : (128, 128, 128),
            }
            name_lower = color.lower().strip()
            if name_lower in _named:
                return _named[name_lower]
            # Try hex string #RRGGBB
            if name_lower.startswith("#") and len(name_lower) == 7:
                r = int(name_lower[1:3], 16)
                g = int(name_lower[3:5], 16)
                b = int(name_lower[5:7], 16)
                return (r, g, b)
        return (255, 255, 255)  # fallback: white