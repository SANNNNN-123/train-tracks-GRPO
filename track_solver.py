"""Choose a puzzle image, extract its clues and givens, then run the solver."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import tkinter as tk
from tkinter import messagebox, ttk

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageTk
except ImportError as exc:
    raise SystemExit(
        "Image parsing needs OpenCV, NumPy, and Pillow. Install them with: "
        "py -m pip install opencv-python numpy pillow"
    ) from exc


ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"
SOLVER_DIR = ROOT / "tracks-solver"

import sys

sys.path.insert(0, str(SOLVER_DIR))
import tracks


def _grid_lines(projection: np.ndarray, minimum: float) -> list[int]:
    """Return centers of strong, continuous grid lines in a pixel projection."""
    active = projection >= minimum
    groups: list[list[int]] = []
    for index in np.flatnonzero(active):
        if not groups or index > groups[-1][-1] + 1:
            groups.append([int(index)])
        else:
            groups[-1].append(int(index))
    return [int(round(sum(group) / len(group))) for group in groups if len(group) <= 4]


def _normalize_glyph(mask: np.ndarray) -> np.ndarray | None:
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    glyph = mask[y : y + height, x : x + width]
    scale = min(26 / max(width, 1), 30 / max(height, 1))
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    glyph = cv2.resize(glyph, new_size, interpolation=cv2.INTER_AREA)
    canvas = np.zeros((32, 32), dtype=np.uint8)
    left = (32 - new_size[0]) // 2
    top = (32 - new_size[1]) // 2
    canvas[top : top + new_size[1], left : left + new_size[0]] = glyph
    return (canvas > 90).astype(np.uint8)


@lru_cache(maxsize=1)
def _digit_templates() -> list[tuple[str, np.ndarray]]:
    font_paths = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
        Path(r"C:\Windows\Fonts\tahoma.ttf"),
    ]
    templates: list[tuple[str, np.ndarray]] = []
    for font_path in font_paths:
        if not font_path.exists():
            continue
        for point_size in (34, 38, 42, 46):
            font = ImageFont.truetype(str(font_path), point_size)
            for digit in "0123456789":
                image = Image.new("L", (90, 90), 0)
                draw = ImageDraw.Draw(image)
                draw.text((16, 8), digit, fill=255, font=font)
                template = _normalize_glyph(np.array(image))
                if template is not None:
                    templates.append((digit, template))
    return templates


def _read_digit(crop: np.ndarray, templates: list[tuple[str, np.ndarray]]) -> int:
    binary = (crop < 135).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    candidates = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if height >= crop.shape[0] * 0.28 and width >= 2 and area >= 12:
            candidates.append((area, x, y, width, height, label))
    if not candidates:
        raise ValueError("Could not find a clue digit in the image.")
    _, x, y, width, height, label = max(candidates)
    component = (labels[y : y + height, x : x + width] == label).astype(np.uint8) * 255
    normalized = _normalize_glyph(component)
    if normalized is None or not templates:
        raise ValueError("Could not recognize a clue digit in the image.")
    best_digit = None
    best_score = -1.0
    for digit, template in templates:
        union = np.logical_or(normalized, template).sum()
        score = np.logical_and(normalized, template).sum() / max(union, 1)
        if score > best_score:
            best_digit, best_score = digit, score
    if best_score < 0.30:
        raise ValueError("A clue digit was unclear. Use a sharper puzzle image.")
    return int(best_digit)


def _read_number(crop, templates):
    binary = (crop < 135).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    boxes = sorted((x, y, w, h) for x, y, w, h, area in stats[1:]
                   if h >= crop.shape[0] * .28 and w >= 2 and area >= 12)
    if not boxes:
        raise ValueError("Could not read a clue number.")
    return int("".join(str(_read_digit(crop[y:y+h, x:x+w], templates))
                       for x, y, w, h in boxes))


def _read_clues(gray: np.ndarray, xs: list[int], ys: list[int]) -> tuple[list[int], list[int]]:
    templates = _digit_templates()
    columns, rows = [], []
    top = max(0, ys[0] - int((ys[1] - ys[0]) * 0.96))
    right = min(gray.shape[1], xs[-1] + int((xs[1] - xs[0]) * 0.98))
    for col in range(len(xs) - 1):
        columns.append(_read_number(gray[top : ys[0] - 2, xs[col] + 2 : xs[col + 1] - 2], templates))
    for row in range(len(ys) - 1):
        rows.append(_read_number(gray[ys[row] + 2 : ys[row + 1] - 2, xs[-1] + 2 : right], templates))
    return columns, rows


def _is_track_cell(image: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> bool:
    tile = image[y0 + 3 : y1 - 3, x0 + 3 : x1 - 3]
    if tile.size == 0:
        return False
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
    neutral_gray = (gray >= 165) & (gray <= 238) & (hsv[:, :, 1] <= 45)
    return float(neutral_gray.mean()) > 0.12


def _track_directions(image: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> str:
    """Estimate the two rail ports from the colored sleepers near cell edges."""
    tile = image[y0:y1, x0:x1]
    height, width = tile.shape[:2]
    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    yellow = (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 42) & (hsv[:, :, 1] >= 55)
    dark = gray < 75
    side_regions = {
        "N": (slice(int(height * 0.03), int(height * 0.10)), slice(int(width * 0.15), int(width * 0.85))),
        "S": (slice(int(height * 0.90), int(height * 0.97)), slice(int(width * 0.15), int(width * 0.85))),
        "W": (slice(int(height * 0.15), int(height * 0.85)), slice(int(width * 0.03), int(width * 0.10))),
        "E": (slice(int(height * 0.15), int(height * 0.85)), slice(int(width * 0.90), int(width * 0.97))),
    }
    scores = {}
    for direction, (ys, xs) in side_regions.items():
        region = (yellow | dark)[ys, xs]
        scores[direction] = int(region.sum())
    pairs = (("N", "S"), ("E", "W"), ("N", "E"), ("N", "W"), ("S", "E"), ("S", "W"))
    pair = max(pairs, key=lambda candidate: scores[candidate[0]] + scores[candidate[1]])
    if min(scores[pair[0]], scores[pair[1]]) < 3:
        raise ValueError("Could not identify a fixed track piece in one of the shaded cells.")
    return "".join(pair)


def parse_puzzle_image(path: Path) -> tuple[dict, dict]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not open image: {path.name}")
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = (gray < 85).astype(np.uint8)
    x_lines = _grid_lines(dark.sum(axis=0), height * 0.60)
    y_lines = _grid_lines(dark.sum(axis=1), width * 0.60)
    if len(x_lines) < 3 or len(y_lines) < 3:
        raise ValueError("Could not detect a complete grid with at least 2 rows and columns.")
    for lines in (x_lines, y_lines):
        gaps = np.diff(lines)
        if np.max(np.abs(gaps - np.median(gaps))) > max(2, np.median(gaps) * .12):
            raise ValueError("Grid lines are not evenly spaced. Use a complete, clear PNG.")
    row_count, column_count = len(y_lines) - 1, len(x_lines) - 1
    columns, rows = _read_clues(gray, x_lines, y_lines)
    if sum(columns) != sum(rows):
        raise ValueError(
            f"Clue totals do not match (columns {sum(columns)}, rows {sum(rows)}); "
            "the image clues may have been misread."
        )

    tracks_found = []
    outward_ports = []
    for image_row in range(row_count):
        for col in range(column_count):
            x0, x1 = x_lines[col], x_lines[col + 1]
            y0, y1 = y_lines[image_row], y_lines[image_row + 1]
            if not _is_track_cell(image, x0, x1, y0, y1):
                continue
            directions = _track_directions(image, x0, x1, y0, y1)
            solver_row = row_count - 1 - image_row
            tracks_found.append([directions, solver_row, col])
            for direction in directions:
                if (direction == "N" and image_row == 0) or (direction == "S" and image_row == row_count - 1) or (direction == "W" and col == 0) or (direction == "E" and col == column_count - 1):
                    outward_ports.append((solver_row, col, direction))

    if len(outward_ports) != 2:
        raise ValueError(
            f"Detected {len(outward_ports)} track ends on the grid edge; expected 2. "
            "Check the image or add support for this puzzle layout."
        )
    start = outward_ports[0][:2]
    end = outward_ports[1][:2]
    for piece in tracks_found:
        if tuple(piece[1:]) == start:
            piece.append("s")
        elif tuple(piece[1:]) == end:
            piece.append("e")

    if any(v < 0 or v > row_count for v in columns) or any(v < 0 or v > column_count for v in rows):
        raise ValueError("A clue exceeds the detected grid dimensions.")
    data = {"columns": columns, "rows": rows, "pieces": tracks_found}
    return data, data


class PuzzlePicker:
    def __init__(self, root: tk.Tk):
        from queue import Queue
        self.root = root
        self.results = Queue()
        self.busy = False
        self.board = None
        self.data = None
        root.title("Train Tracks Solver")
        self.files = []
        panel = ttk.Frame(root, padding=12)
        panel.pack(fill="both", expand=True)
        controls = ttk.Frame(panel)
        controls.grid(row=0, column=0, sticky="w", pady=8)
        ttk.Label(controls, text="Month").grid(row=0, column=0, sticky="w")
        ttk.Label(controls, text="Date / image").grid(row=0, column=1, sticky="w", padx=(12, 0))
        self.month = tk.StringVar()
        self.month_dropdown = ttk.Combobox(controls, textvariable=self.month,
                                          state="readonly", width=18,
                                          postcommand=self._refresh_months)
        self.month_dropdown.grid(row=1, column=0)
        self.month_dropdown.bind("<<ComboboxSelected>>", self._load_month)
        self.selected = tk.StringVar()
        self.dropdown = ttk.Combobox(controls, textvariable=self.selected,
                                    state="readonly", width=28,
                                    postcommand=self._refresh_dates)
        self.dropdown.grid(row=1, column=1, padx=(12, 0))
        self.dropdown.bind("<<ComboboxSelected>>", self._show_preview)
        self.random_button = ttk.Button(controls, text="Random", command=self._random_puzzle)
        self.random_button.grid(row=1, column=2, padx=(8, 0))
        self.run_button = ttk.Button(panel, text="Run solver", command=self.run)
        self.run_button.grid(row=0, column=1, sticky="e")
        ttk.Label(panel, text="Original PNG").grid(row=1, column=0, pady=6)
        ttk.Label(panel, text="Parsed grid / solution (blue = given tracks)").grid(row=1, column=1, pady=6)
        self.preview_label = ttk.Label(panel, anchor="center")
        self.preview_label.grid(row=2, column=0, padx=(0, 12))
        self.canvas = tk.Canvas(panel, width=500, height=500, background="white", highlightthickness=1)
        self.canvas.grid(row=2, column=1)
        self.status = tk.StringVar()
        ttk.Label(panel, textvariable=self.status, wraplength=980).grid(row=3, column=0, columnspan=2, sticky="w", pady=10)
        self._refresh_months()

    def _refresh_months(self):
        folders = sorted(
            (p.name for p in ASSETS_DIR.iterdir()
             if p.is_dir() and p.name.casefold() != "traintracks-rules"),
            key=str.casefold,
        ) if ASSETS_DIR.exists() else []
        self.month_dropdown.configure(values=folders)
        if self.month.get() not in folders:
            self.month.set(folders[0] if folders else "")
            self._load_month()

    def _load_month(self, _event=None):
        self.selected.set("")
        self._refresh_dates()

    def _refresh_dates(self):
        folder = ASSETS_DIR / self.month.get()
        self.files = sorted(
            (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"),
            key=lambda p: p.name.casefold(),
        ) if self.month.get() and folder.is_dir() else []
        names = [p.name for p in self.files]
        self.dropdown.configure(values=names)
        self.random_button.configure(state="normal" if names and not self.busy else "disabled")
        if not names:
            self.selected.set("")
            self.board = self.data = None
            self.preview_label.configure(image="")
            self.canvas.delete("all")
            self.run_button.configure(state="disabled")
            self.status.set("No PNG images in this month." if self.month.get() else "No month folders found in assets.")
        elif self.selected.get() not in names:
            self.selected.set(names[0])
            self._show_preview()

    def _random_puzzle(self):
        from random import choice
        if self.busy:
            return
        self._refresh_dates()
        if not self.files:
            return
        candidates = [p.name for p in self.files if p.name != self.selected.get()]
        self.selected.set(choice(candidates or [p.name for p in self.files]))
        self._show_preview()

    def _show_preview(self, _event=None):
        path = ASSETS_DIR / self.month.get() / self.selected.get()
        self.board = None
        self.data = None
        self.canvas.delete("all")
        try:
            with Image.open(path) as original:
                image = original.copy()
            image.thumbnail((500, 500))
            self.preview = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self.preview)
            encoded, self.data = parse_puzzle_image(path)
            self.encoded = encoded
            self.board = tracks.parse(encoded, graphical=False)
            self.draw_board()
            self.run_button.configure(state="normal")
            self.status.set(f"{path.name}: mapped to grid. Press Run solver.")
        except Exception as exc:
            self.run_button.configure(state="disabled")
            self.status.set(f"Cannot map {path.name}: {exc}")
            self.canvas.create_text(250, 250, text="Image could not be mapped.\nSee the message below.", justify="center")

    def draw_board(self):
        board = self.board
        self.canvas.delete("all")
        nr, nc = board.rows, board.columns
        step = min(420 / nc, 420 / nr)
        left, top = 20, 50
        for i in range(nc + 1):
            self.canvas.create_line(left + i * step, top, left + i * step, top + nr * step, fill="#bbb")
        for i in range(nr + 1):
            self.canvas.create_line(left, top + i * step, left + nc * step, top + i * step, fill="#bbb")
        for col, value in enumerate(board.col_constraints):
            self.canvas.create_text(left + (col + .5) * step, 25, text=str(value), font=("Arial", 16))
        for row in range(nr):
            y = top + (nr - 1 - row + .5) * step
            self.canvas.create_text(left + nc * step + 25, y, text=str(board.row_constraints[row]), font=("Arial", 16))
            for col in range(nc):
                cell = board.layout[row][col]
                if not cell.track or cell.track == tracks.Track.TEMP:
                    continue
                x = left + (col + .5) * step
                color = "#2259b7" if cell.permanent else "#222222"
                offsets = {"N": (0, -step / 2), "S": (0, step / 2), "W": (-step / 2, 0), "E": (step / 2, 0)}
                for direction in cell.track.name:
                    dx, dy = offsets[direction]
                    self.canvas.create_line(x, y, x + dx, y + dy, fill=color, width=4, capstyle="round")
                if cell.is_start or cell.is_end:
                    self.canvas.create_text(x - step * .28, y - step * .28, text="A" if cell.is_start else "B", fill=color)

    def run(self):
        from threading import Thread
        if self.busy or self.data is None:
            return
        self.board = tracks.parse(self.encoded, graphical=False)
        self.draw_board()
        self.busy = True
        self.month_dropdown.configure(state="disabled")
        self.dropdown.configure(state="disabled")
        self.random_button.configure(state="disabled")
        self.run_button.configure(state="disabled")
        self.status.set("Solving... The solution will appear in the right-hand grid.")
        board = self.board

        def solve():
            try:
                board.solve()
                self.results.put((False, "No solution returned."))
            except ValueError as exc:
                self.results.put((str(exc) == "Solved", str(exc)))
            except Exception as exc:
                self.results.put((False, str(exc)))

        Thread(target=solve, daemon=True).start()
        self.root.after(100, self._poll)

    def _poll(self):
        from queue import Empty
        try:
            solved, message = self.results.get_nowait()
        except Empty:
            self.root.after(100, self._poll)
            return
        self.busy = False
        self.month_dropdown.configure(state="readonly")
        self.dropdown.configure(state="readonly")
        self.random_button.configure(state="normal" if self.files else "disabled")
        self.run_button.configure(state="normal")
        if solved:
            self.draw_board()
            self.status.set("Puzzle solved. Select another image or close the window when finished.")
            messagebox.showinfo("Done", "Puzzle solved", parent=self.root)
        else:
            self.board = tracks.parse(self.encoded, graphical=False)
            self.draw_board()
            self.status.set(f"Solver stopped: {message}")
            messagebox.showerror("Solver", message, parent=self.root)


def main():
    root = tk.Tk()
    try:
        PuzzlePicker(root)
    except Exception as exc:
        messagebox.showerror("Train Tracks Solver", str(exc), parent=root)
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
