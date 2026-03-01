"""
Grid-Based Pathfinding Navigation System
Implements A* and Greedy Best-First Search algorithms
with Manhattan and Euclidean distance heuristics
Features real-time visualization and dynamic re-planning
"""

import tkinter as tk
from tkinter import messagebox
import heapq
import random
import time
import math
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════
#  THEME CONFIGURATION
# ══════════════════════════════════════════════════════════════════
THEME_DARK_BG = "#0d0d14"
THEME_CARD_BG = "#13131f"
THEME_CARD_ALT = "#1a1a2e"
THEME_BORDER_COL = "#2a2a45"
THEME_PRIMARY = "#7c6af7"
THEME_SECONDARY = "#5eead4"
THEME_ACCENT_PINK = "#f472b6"
THEME_TEXT_REGULAR = "#e2e8f0"
THEME_TEXT_MUTED = "#64748b"
THEME_TEXT_HIGHLIGHT = "#f8fafc"

COLOR_CELL_EMPTY = "#111120"
COLOR_OBSTACLE = "#2d2d4e"
COLOR_OBSTACLE_BORDER = "#3d3d6e"
COLOR_START_NODE = "#10b981"
COLOR_GOAL_NODE = "#f43f5e"
COLOR_EXPLORATION_FRONTIER = "#fbbf24"
COLOR_EXPLORED_NODE = "#3b82f6"
COLOR_SOLUTION_PATH = "#a78bfa"
COLOR_AGENT_POS = "#ffffff"
COLOR_GRID_LINE = "#1e1e35"

CELL_DIMENSION = 26
CELL_SPACING = 2

TYPOGRAPHY_HEADING = ("Courier", 15, "bold")
TYPOGRAPHY_LABEL = ("Courier", 9)
TYPOGRAPHY_SMALL = ("Courier", 8)
TYPOGRAPHY_METRIC = ("Courier", 18, "bold")
TYPOGRAPHY_CAPTION = ("Courier", 8)
TYPOGRAPHY_BUTTON = ("Courier", 9, "bold")
TYPOGRAPHY_SECTION = ("Courier", 10, "bold")

# ══════════════════════════════════════════════════════════════════
#  PATHFINDING ALGORITHMS
# ══════════════════════════════════════════════════════════════════

class SearchAlgorithm:
    """Base class for search algorithms"""
    def __init__(self, grid, rows, cols, start, goal, heuristic_fn):
        self.grid = grid
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.heuristic = heuristic_fn
        self.explored_nodes = []
    
    def compute(self):
        raise NotImplementedError
    
    def _expand_neighbors(self, node):
        """Generate valid neighboring cells"""
        row, col = node
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                if self.grid[new_row][new_col] != 1:
                    yield (new_row, new_col)
    
    def _backtrack_path(self, parent_map, end_node):
        """Reconstruct path from start to goal"""
        p = []
        current = end_node
        while current is not None:
            p.append(current)
            current = parent_map[current]
        p.reverse()
        return p


class AStarSearcher(SearchAlgorithm):
    """A* search implementation"""
    def compute(self):
        open_set = [(self.heuristic(self.start, self.goal), 0, self.start)]
        g_scores = defaultdict(lambda: float('inf'))
        g_scores[self.start] = 0
        parent_map = {self.start: None}
        
        while open_set:
            f_val, g_val, current = heapq.heappop(open_set)
            
            # Skip if we've already found a shorter path
            if g_val > g_scores[current]:
                continue
            
            if current == self.goal:
                return self._backtrack_path(parent_map, self.goal), self.explored_nodes
            
            self.explored_nodes.append(current)
            
            for neighbor in self._expand_neighbors(current):
                tentative_g = g_scores[current] + 1
                if tentative_g < g_scores[neighbor]:
                    parent_map[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, self.goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
        
        return None, self.explored_nodes


class GreedyBestFirstSearcher(SearchAlgorithm):
    """Greedy Best-First search implementation"""
    def compute(self):
        open_set = [(self.heuristic(self.start, self.goal), self.start)]
        parent_map = {self.start: None}
        
        while open_set:
            h_val, current = heapq.heappop(open_set)
            
            if current == self.goal:
                return self._backtrack_path(parent_map, self.goal), self.explored_nodes
            
            self.explored_nodes.append(current)
            
            for neighbor in self._expand_neighbors(current):
                if neighbor not in parent_map:
                    parent_map[neighbor] = current
                    h_score = self.heuristic(neighbor, self.goal)
                    heapq.heappush(open_set, (h_score, neighbor))
        
        return None, self.explored_nodes


def calculate_manhattan_distance(pos1, pos2):
    """Manhattan distance heuristic"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def calculate_euclidean_distance(pos1, pos2):
    """Euclidean distance heuristic"""
    return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])

# ══════════════════════════════════════════════════════════════════
#  UI COMPONENTS
# ══════════════════════════════════════════════════════════════════

class ModernButton(tk.Label):
    """Modern styled button component"""
    def __init__(self, parent, label_text, on_press, bg_color=THEME_PRIMARY,
                 text_color=THEME_TEXT_HIGHLIGHT, btn_width=140, btn_height=34,
                 font=TYPOGRAPHY_BUTTON, **kw):
        self._base_bg = bg_color
        self._hover_bg = self._brighten_hex(bg_color)
        self._on_press = on_press
        self._is_enabled = True
        self._text_color = text_color

        super().__init__(
            parent, text=label_text, font=font,
            bg=bg_color, fg=text_color,
            padx=10, pady=6,
            cursor="hand2", relief=tk.FLAT,
            width=max(1, btn_width // 8),
            **kw
        )
        self.bind("<Enter>", lambda e: self._hover_in())
        self.bind("<Leave>", lambda e: self._hover_out())
        self.bind("<Button-1>", lambda e: self._click_action())

    def _brighten_hex(self, hex_color):
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f"#{min(255, r+35):02x}{min(255, g+35):02x}{min(255, b+35):02x}"

    def _hover_in(self):
        if self._is_enabled:
            self.config(bg=self._hover_bg)

    def _hover_out(self):
        self.config(bg=self._base_bg if self._is_enabled else "#333344")

    def _click_action(self):
        if self._is_enabled:
            self._on_press()

    def set_button_state(self, is_active):
        self._is_enabled = is_active
        if is_active:
            self.config(bg=self._base_bg, fg=self._text_color, cursor="hand2")
        else:
            self.config(bg="#333344", fg="#666677", cursor="")


class ToggleButtonGroup(tk.Frame):
    def __init__(self, parent, options_list, selected_var, color_mapping=None, **kw):
        super().__init__(parent, bg=THEME_CARD_ALT, padx=2, pady=2, **kw)
        self._selected_var = selected_var
        self._buttons_dict = {}
        self._color_mapping = color_mapping or {}
        
        for option_val, option_label in options_list:
            btn = tk.Label(self, text=option_label, font=TYPOGRAPHY_LABEL,
                          cursor="hand2", padx=10, pady=4,
                          bg=THEME_CARD_ALT, fg=THEME_TEXT_MUTED)
            btn.pack(side=tk.LEFT, padx=1)
            btn.bind("<Button-1>", lambda e, v=option_val: self._selected_var.set(v))
            btn.bind("<Enter>", lambda e, b=btn, v=option_val: self._on_hover(b, v))
            btn.bind("<Leave>", lambda e: self._update_display())
            self._buttons_dict[option_val] = btn
        
        selected_var.trace_add("write", lambda *a: self._update_display())
        self._update_display()

    def _on_hover(self, button_widget, option_val):
        if self._selected_var.get() != option_val:
            button_widget.config(fg=THEME_TEXT_REGULAR, bg=THEME_BORDER_COL)

    def _update_display(self):
        current_val = self._selected_var.get()
        for val, btn in self._buttons_dict.items():
            if val == current_val:
                btn.config(bg=self._color_mapping.get(val, THEME_PRIMARY), 
                          fg=THEME_TEXT_HIGHLIGHT)
            else:
                btn.config(bg=THEME_CARD_ALT, fg=THEME_TEXT_MUTED)


class ControlSlider(tk.Frame):
    def __init__(self, parent, title_text, control_var, min_val, max_val,
                 step_size, display_fmt="{:.2f}", slider_color=THEME_PRIMARY, **kw):
        super().__init__(parent, bg=THEME_CARD_BG, **kw)
        self._display_fmt = display_fmt
        
        header = tk.Frame(self, bg=THEME_CARD_BG)
        header.pack(fill=tk.X, padx=8, pady=(6, 0))
        tk.Label(header, text=title_text, font=TYPOGRAPHY_CAPTION, 
                bg=THEME_CARD_BG, fg=THEME_TEXT_MUTED).pack(side=tk.LEFT)
        self._value_label = tk.Label(header, text=display_fmt.format(control_var.get()),
                                     font=TYPOGRAPHY_LABEL, bg=THEME_CARD_BG, 
                                     fg=slider_color)
        self._value_label.pack(side=tk.RIGHT)
        
        tk.Scale(self, variable=control_var, from_=min_val, to=max_val,
                resolution=step_size, orient=tk.HORIZONTAL, length=180,
                showvalue=False, bg=THEME_CARD_BG, fg=slider_color,
                highlightthickness=0, troughcolor=THEME_CARD_ALT,
                activebackground=slider_color, sliderrelief=tk.FLAT, bd=0,
                command=lambda v: self._value_label.config(
                    text=display_fmt.format(float(v)))
                ).pack(padx=8, pady=(0, 6))


class StatsDisplay(tk.Frame):
    def __init__(self, parent, metric_label, metric_var, color=THEME_SECONDARY, **kw):
        super().__init__(parent, bg=THEME_CARD_BG, **kw)
        tk.Frame(self, bg=color, height=3).pack(fill=tk.X)
        tk.Label(self, textvariable=metric_var, font=TYPOGRAPHY_METRIC,
                bg=THEME_CARD_BG, fg=color).pack(pady=(6, 0))
        tk.Label(self, text=metric_label, font=TYPOGRAPHY_CAPTION,
                bg=THEME_CARD_BG, fg=THEME_TEXT_MUTED).pack(pady=(0, 8))


class SectionTitle(tk.Frame):
    def __init__(self, parent, section_text, **kw):
        bg_color = kw.pop("bg", THEME_CARD_BG)
        super().__init__(parent, bg=bg_color, **kw)
        tk.Label(self, text=section_text.upper(), font=TYPOGRAPHY_CAPTION,
                bg=bg_color, fg=THEME_TEXT_MUTED).pack(side=tk.LEFT)
        tk.Frame(self, bg=THEME_BORDER_COL, height=1).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0), pady=6)


class ColorIndicator(tk.Frame):
    def __init__(self, parent, indicator_color, indicator_label, **kw):
        bg_color = kw.pop("bg", THEME_CARD_BG)
        super().__init__(parent, bg=bg_color, **kw)
        circle = tk.Canvas(self, width=12, height=12, bg=bg_color, 
                          highlightthickness=0)
        circle.create_oval(1, 1, 11, 11, fill=indicator_color, outline="")
        circle.pack(side=tk.LEFT)
        tk.Label(self, text=indicator_label, font=TYPOGRAPHY_SMALL, 
                bg=bg_color, fg=THEME_TEXT_MUTED).pack(side=tk.LEFT, padx=4)


# ══════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════

class GridPathfinderApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Grid-Based Pathfinding Navigation System")
        self.root.configure(bg=THEME_DARK_BG)
        self.root.resizable(True, True)

        # Grid dimensions
        self.grid_rows = 22
        self.grid_cols = 32
        self.grid_matrix = [[0] * self.grid_cols for _ in range(self.grid_rows)]
        self.start_node = (0, 0)
        self.goal_node = (self.grid_rows - 1, self.grid_cols - 1)
        
        # Path and visualization tracking
        self.solution_path = []
        self.explored_nodes = []
        self.agent_current_pos = None
        self.agent_current_step = 0
        self.is_running = False
        self._scheduled_draw = None
        self._scheduled_agent = None
        self._vis_frame_index = 0
        self._replanning_count = 0

        # Configuration variables
        self.selected_algorithm = tk.StringVar(value="astar")
        self.selected_heuristic = tk.StringVar(value="manhattan")
        self.edit_brush_mode = tk.StringVar(value="wall")
        self.obstacle_spawn_density = tk.DoubleVar(value=0.28)
        self.dynamic_spawn_chance = tk.DoubleVar(value=0.015)
        self.frame_delay_ms = tk.IntVar(value=30)
        self.is_dynamic_enabled = tk.BooleanVar(value=False)

        # Metrics display
        self.explored_count_display = tk.StringVar(value="—")
        self.path_cost_display = tk.StringVar(value="—")
        self.compute_time_display = tk.StringVar(value="—")
        self.replan_count_display = tk.StringVar(value="0")
        self.status_display = tk.StringVar(value="READY")

        # Visualization sets
        self._clear_visualization_sets()
        self._construct_ui()
        self._render_grid()

    # ─────────────────────────────────────────────────────────────
    #  UI LAYOUT
    # ─────────────────────────────────────────────────────────────

    def _construct_ui(self):
        main_container = tk.Frame(self.root, bg=THEME_DARK_BG)
        main_container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        self._render_header_section(main_container)
        
        content_area = tk.Frame(main_container, bg=THEME_DARK_BG)
        content_area.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self._render_control_panel(content_area)
        self._render_visualization_area(content_area)

    def _render_header_section(self, parent_container):
        header = tk.Frame(parent_container, bg=THEME_CARD_BG, pady=10, padx=16)
        header.pack(fill=tk.X, pady=(0, 8))

        indicator = tk.Canvas(header, width=10, height=10, bg=THEME_CARD_BG, highlightthickness=0)
        indicator.create_oval(0, 0, 10, 10, fill=THEME_PRIMARY, outline="")
        indicator.pack(side=tk.LEFT, padx=(0, 10))

        tk.Label(header, text="GRID PATHFINDING SYSTEM", font=TYPOGRAPHY_HEADING,
                bg=THEME_CARD_BG, fg=THEME_TEXT_HIGHLIGHT).pack(side=tk.LEFT)

        tk.Label(header, 
                text="A*  •  GBFS  •  Manhattan  •  Euclidean  •  Dynamic Re-planning",
                font=TYPOGRAPHY_SMALL, bg=THEME_CARD_BG, fg=THEME_TEXT_MUTED).pack(side=tk.LEFT, padx=16)

        status_container = tk.Frame(header, bg=THEME_CARD_BG)
        status_container.pack(side=tk.RIGHT)
        self._status_display_pill = tk.Label(status_container, 
                                             textvariable=self.status_display,
                                             font=TYPOGRAPHY_BUTTON, bg=THEME_PRIMARY, 
                                             fg=THEME_TEXT_HIGHLIGHT,
                                             padx=14, pady=4)
        self._status_display_pill.pack()

    def _wrap_card(self, parent_container, card_title=None):
        outer_wrapper = tk.Frame(parent_container, bg=THEME_DARK_BG, pady=3)
        outer_wrapper.pack(fill=tk.X)
        card_content = tk.Frame(outer_wrapper, bg=THEME_CARD_BG, padx=12, pady=10)
        card_content.pack(fill=tk.X)
        if card_title:
            SectionTitle(card_content, card_title, bg=THEME_CARD_BG).pack(fill=tk.X, pady=(0, 8))
        return card_content

    def _render_control_panel(self, parent_container):
        self._control_panel = tk.Frame(parent_container, bg=THEME_DARK_BG, width=226)
        self._control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self._control_panel.pack_propagate(False)

        # Algorithm selection
        card = self._wrap_card(self._control_panel, "Algorithm")
        ToggleButtonGroup(card, [("astar", "A*  Search"), ("gbfs", "GBFS")],
                         self.selected_algorithm,
                         color_mapping={"astar": THEME_PRIMARY, "gbfs": "#0ea5e9"}
                         ).pack(fill=tk.X, pady=(0, 8))
        SectionTitle(card, "Heuristic", bg=THEME_CARD_BG).pack(fill=tk.X, pady=(2, 6))
        ToggleButtonGroup(card, [("manhattan", "Manhattan"), ("euclidean", "Euclidean")],
                         self.selected_heuristic,
                         color_mapping={"manhattan": THEME_ACCENT_PINK, "euclidean": THEME_SECONDARY}
                         ).pack(fill=tk.X)

        # Edit mode
        card = self._wrap_card(self._control_panel, "Edit Mode")
        ToggleButtonGroup(card,
                         [("wall", "Wall"), ("start", "Start"), ("goal", "Goal")],
                         self.edit_brush_mode,
                         color_mapping={"wall": "#475569", "start": COLOR_START_NODE, "goal": COLOR_GOAL_NODE}
                         ).pack(fill=tk.X, pady=(0, 6))
        tk.Label(card, text="L-click: place   R-click: erase   Drag: paint",
                font=TYPOGRAPHY_SMALL, bg=THEME_CARD_BG, fg=THEME_TEXT_MUTED).pack()

        # Grid sizing
        card = self._wrap_card(self._control_panel, "Grid Size")
        size_inputs = tk.Frame(card, bg=THEME_CARD_BG)
        size_inputs.pack(fill=tk.X, pady=(0, 6))
        for label_txt, attr_name, default_val in [("Rows", "row_input", "22"), ("Cols", "col_input", "32")]:
            input_box = tk.Frame(size_inputs, bg=THEME_CARD_BG)
            input_box.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
            tk.Label(input_box, text=label_txt, font=TYPOGRAPHY_CAPTION, 
                    bg=THEME_CARD_BG, fg=THEME_TEXT_MUTED).pack(anchor="w")
            spinner = tk.Spinbox(input_box, from_=5, to=60, width=5, 
                                bg=THEME_CARD_ALT, fg=THEME_TEXT_REGULAR,
                                relief=tk.FLAT, buttonbackground=THEME_BORDER_COL,
                                insertbackground=THEME_TEXT_REGULAR, highlightthickness=1,
                                highlightcolor=THEME_PRIMARY, highlightbackground=THEME_BORDER_COL,
                                font=TYPOGRAPHY_LABEL)
            spinner.delete(0, tk.END)
            spinner.insert(0, default_val)
            spinner.pack(fill=tk.X)
            setattr(self, attr_name, spinner)
        
        apply_size_btn = ModernButton(card, "⟳  Apply Size", self.apply_grid_resize, 
                                     bg_color=THEME_BORDER_COL, btn_width=198, btn_height=30)
        apply_size_btn.pack(pady=(0, 4))
        
        ControlSlider(card, "Obstacle Density", self.obstacle_spawn_density,
                     0.1, 0.6, 0.05, display_fmt="{:.0%}", slider_color=THEME_ACCENT_PINK).pack(fill=tk.X, pady=2)
        
        button_row = tk.Frame(card, bg=THEME_CARD_BG)
        button_row.pack(fill=tk.X, pady=2)
        generate_maze_btn = ModernButton(button_row, "⚡ Generate Maze", 
                                        self.generate_maze_map, bg_color=THEME_PRIMARY, 
                                        btn_width=98, btn_height=30)
        generate_maze_btn.pack(side=tk.LEFT, padx=(0, 2))
        clear_grid_btn = ModernButton(button_row, "✕ Clear", self.clear_all_obstacles, 
                                     bg_color="#475569", btn_width=96, btn_height=30)
        clear_grid_btn.pack(side=tk.LEFT)

        # Dynamic mode
        card = self._wrap_card(self._control_panel, "Dynamic Obstacles")
        toggle_area = tk.Frame(card, bg=THEME_CARD_BG)
        toggle_area.pack(fill=tk.X, pady=(0, 8))
        tk.Label(toggle_area, text="Enable Spawning",
                font=TYPOGRAPHY_LABEL, bg=THEME_CARD_BG, fg=THEME_TEXT_REGULAR).pack(side=tk.LEFT)
        self._dynamic_toggle_widget = tk.Canvas(toggle_area, width=42, height=22,
                                                bg=THEME_CARD_BG, highlightthickness=0, cursor="hand2")
        self._dynamic_toggle_widget.pack(side=tk.RIGHT)
        self._dynamic_toggle_widget.bind("<Button-1>", self._handle_dynamic_toggle)
        self._paint_toggle_widget()
        ControlSlider(card, "Spawn Probability", self.dynamic_spawn_chance,
                     0.005, 0.05, 0.005, display_fmt="{:.3f}", slider_color=THEME_ACCENT_PINK).pack(fill=tk.X)

        # Speed control
        card = self._wrap_card(self._control_panel, "Animation Speed")
        ControlSlider(card, "Delay (ms/step)", self.frame_delay_ms,
                     5, 200, 5, display_fmt="{:.0f} ms", slider_color=THEME_SECONDARY).pack(fill=tk.X)

        # Legend
        card = self._wrap_card(self._control_panel, "Legend")
        for color_val, label_text in [(COLOR_START_NODE, "Start Node"), (COLOR_GOAL_NODE, "Goal Node"),
                                      (COLOR_AGENT_POS, "Agent"), (COLOR_EXPLORED_NODE, "Visited"),
                                      (COLOR_EXPLORATION_FRONTIER, "Frontier"), 
                                      (COLOR_SOLUTION_PATH, "Final Path"), (COLOR_OBSTACLE, "Wall")]:
            ColorIndicator(card, color_val, label_text, bg=THEME_CARD_BG).pack(anchor="w", pady=2)

    def _paint_toggle_widget(self):
        self._dynamic_toggle_widget.delete("all")
        is_on = self.is_dynamic_enabled.get()
        circle_color = THEME_PRIMARY if is_on else THEME_BORDER_COL
        self._dynamic_toggle_widget.create_oval(0, 2, 40, 20, fill=circle_color, outline="")
        toggle_x = 24 if is_on else 10
        self._dynamic_toggle_widget.create_oval(toggle_x - 7, 4, toggle_x + 7, 18, 
                                               fill=THEME_TEXT_HIGHLIGHT, outline="")

    def _handle_dynamic_toggle(self, event=None):
        self.is_dynamic_enabled.set(not self.is_dynamic_enabled.get())
        self._paint_toggle_widget()

    def _render_visualization_area(self, parent_container):
        visualization_panel = tk.Frame(parent_container, bg=THEME_DARK_BG)
        visualization_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._render_action_controls(visualization_panel)
        self._render_metrics_display(visualization_panel)
        
        canvas_border = tk.Frame(visualization_panel, bg=THEME_BORDER_COL, padx=1, pady=1)
        canvas_border.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        canvas_inner = tk.Frame(canvas_border, bg=COLOR_CELL_EMPTY)
        canvas_inner.pack(fill=tk.BOTH, expand=True)
        
        self.canvas_widget = tk.Canvas(canvas_inner, bg=COLOR_CELL_EMPTY, highlightthickness=0)
        self.canvas_widget.pack(padx=4, pady=4)
        self._calculate_canvas_size()
        self.canvas_widget.bind("<Button-1>", self.handle_canvas_click)
        self.canvas_widget.bind("<B1-Motion>", self.handle_canvas_drag)
        self.canvas_widget.bind("<Button-3>", self.handle_right_click)

    def _render_action_controls(self, parent_container):
        controls = tk.Frame(parent_container, bg=THEME_DARK_BG)
        controls.pack(fill=tk.X, pady=(0, 6))

        self._start_search_btn = ModernButton(
            controls, "▶   START SEARCH", self.start_pathfinding_search,
            bg_color=COLOR_START_NODE, text_color="#000", btn_width=180, btn_height=40,
            font=("Courier", 10, "bold"))
        self._start_search_btn.pack(side=tk.LEFT, padx=(0, 6))

        self._stop_search_btn = ModernButton(
            controls, "⏹  STOP", self.stop_pathfinding_search,
            bg_color=COLOR_GOAL_NODE, text_color="#fff", btn_width=110, btn_height=40)
        self._stop_search_btn.pack(side=tk.LEFT, padx=(0, 6))
        self._stop_search_btn.set_button_state(False)

        reset_btn = ModernButton(controls, "↺  Reset", self.reset_all_state, 
                                bg_color=THEME_BORDER_COL, btn_width=110, btn_height=40)
        reset_btn.pack(side=tk.LEFT)

        status_display_card = tk.Frame(controls, bg=THEME_CARD_BG, padx=14, pady=10)
        status_display_card.pack(side=tk.RIGHT)
        tk.Label(status_display_card, text="STATUS", font=TYPOGRAPHY_CAPTION,
                bg=THEME_CARD_BG, fg=THEME_TEXT_MUTED).pack(side=tk.LEFT, padx=(0, 10))
        self._status_label = tk.Label(status_display_card, textvariable=self.status_display,
                                      font=("Courier", 10, "bold"),
                                      bg=THEME_CARD_BG, fg=THEME_SECONDARY, width=14, anchor="w")
        self._status_label.pack(side=tk.LEFT)

    def _render_metrics_display(self, parent_container):
        metrics_panel = tk.Frame(parent_container, bg=THEME_DARK_BG)
        metrics_panel.pack(fill=tk.X)
        for metric_name, metric_var, metric_color in [
            ("NODES VISITED", self.explored_count_display, THEME_PRIMARY),
            ("PATH COST", self.path_cost_display, THEME_SECONDARY),
            ("TIME  (ms)", self.compute_time_display, THEME_ACCENT_PINK),
            ("RE-PLANS", self.replan_count_display, "#f59e0b"),
        ]:
            StatsDisplay(metrics_panel, metric_name, metric_var, metric_color).pack(
                side=tk.LEFT, fill=tk.X, expand=True, padx=3)

    # ─────────────────────────────────────────────────────────────
    #  CANVAS OPERATIONS
    # ─────────────────────────────────────────────────────────────

    def _calculate_canvas_size(self):
        canvas_width = self.grid_cols * (CELL_DIMENSION + CELL_SPACING) + CELL_SPACING
        canvas_height = self.grid_rows * (CELL_DIMENSION + CELL_SPACING) + CELL_SPACING
        self.canvas_widget.config(width=canvas_width, height=canvas_height)

    def _render_grid(self):
        self.canvas_widget.delete("all")
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                self._render_single_cell(row, col)

    def _render_single_cell(self, row, col):
        x1 = col * (CELL_DIMENSION + CELL_SPACING) + CELL_SPACING
        y1 = row * (CELL_DIMENSION + CELL_SPACING) + CELL_SPACING
        x2, y2 = x1 + CELL_DIMENSION, y1 + CELL_DIMENSION
        cell_tag = f"cell_{row}_{col}"
        self.canvas_widget.delete(cell_tag)
        cell_pos = (row, col)

        if cell_pos == self.agent_current_pos:
            self.canvas_widget.create_rectangle(x1, y1, x2, y2, fill=COLOR_SOLUTION_PATH, outline="", tags=cell_tag)
            padding = 5
            self.canvas_widget.create_oval(x1+padding, y1+padding, x2-padding, y2-padding,
                                          fill=COLOR_AGENT_POS, outline="", tags=cell_tag)
        elif cell_pos == self.start_node:
            self.canvas_widget.create_rectangle(x1, y1, x2, y2, fill=COLOR_START_NODE, outline="", tags=cell_tag)
            self.canvas_widget.create_text((x1+x2)//2, (y1+y2)//2, text="S",
                                          fill="#000", font=("Courier", 8, "bold"), tags=cell_tag)
        elif cell_pos == self.goal_node:
            self.canvas_widget.create_rectangle(x1, y1, x2, y2, fill=COLOR_GOAL_NODE, outline="", tags=cell_tag)
            self.canvas_widget.create_text((x1+x2)//2, (y1+y2)//2, text="G",
                                          fill="#fff", font=("Courier", 8, "bold"), tags=cell_tag)
        elif self.grid_matrix[row][col] == 1:
            self.canvas_widget.create_rectangle(x1, y1, x2, y2,
                                               fill=COLOR_OBSTACLE, outline=COLOR_OBSTACLE_BORDER, tags=cell_tag)
        else:
            cell_color = COLOR_CELL_EMPTY
            if cell_pos in self._solution_path_set:
                cell_color = COLOR_SOLUTION_PATH
            elif cell_pos in self._explored_nodes_set:
                cell_color = COLOR_EXPLORED_NODE
            elif cell_pos in self._frontier_nodes_set:
                cell_color = COLOR_EXPLORATION_FRONTIER
            
            outline_color = COLOR_GRID_LINE if cell_color == COLOR_CELL_EMPTY else ""
            self.canvas_widget.create_rectangle(x1, y1, x2, y2,
                                               fill=cell_color, outline=outline_color, tags=cell_tag)

    def _refresh_single_cell(self, row, col):
        self._render_single_cell(row, col)

    # ─────────────────────────────────────────────────────────────
    #  USER INTERACTION
    # ─────────────────────────────────────────────────────────────

    def _get_cell_from_click(self, event):
        col = event.x // (CELL_DIMENSION + CELL_SPACING)
        row = event.y // (CELL_DIMENSION + CELL_SPACING)
        if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
            return row, col
        return None

    def handle_canvas_click(self, event):
        if self.is_running:
            return
        cell = self._get_cell_from_click(event)
        if cell:
            self._apply_brush_edit(cell)

    def handle_canvas_drag(self, event):
        if self.is_running:
            return
        cell = self._get_cell_from_click(event)
        if cell:
            self._apply_brush_edit(cell)

    def handle_right_click(self, event):
        if self.is_running:
            return
        cell = self._get_cell_from_click(event)
        if cell:
            row, col = cell
            if (row, col) not in (self.start_node, self.goal_node):
                self.grid_matrix[row][col] = 0
                self._refresh_single_cell(row, col)

    def _apply_brush_edit(self, cell):
        row, col = cell
        brush_mode = self.edit_brush_mode.get()
        
        if brush_mode == "wall":
            if (row, col) not in (self.start_node, self.goal_node):
                self.grid_matrix[row][col] = 1 - self.grid_matrix[row][col]
                self._refresh_single_cell(row, col)
        elif brush_mode == "start":
            old_start = self.start_node
            self.start_node = (row, col)
            self.grid_matrix[row][col] = 0
            self._refresh_single_cell(*old_start)
            self._refresh_single_cell(row, col)
        elif brush_mode == "goal":
            old_goal = self.goal_node
            self.goal_node = (row, col)
            self.grid_matrix[row][col] = 0
            self._refresh_single_cell(*old_goal)
            self._refresh_single_cell(row, col)

    # ─────────────────────────────────────────────────────────────
    #  GRID MANAGEMENT
    # ─────────────────────────────────────────────────────────────

    def apply_grid_resize(self):
        self.stop_pathfinding_search()
        try:
            rows = max(5, min(50, int(self.row_input.get())))
            cols = max(5, min(60, int(self.col_input.get())))
        except ValueError:
            return
        
        self.grid_rows, self.grid_cols = rows, cols
        self.grid_matrix = [[0] * self.grid_cols for _ in range(self.grid_rows)]
        self.start_node = (0, 0)
        self.goal_node = (self.grid_rows - 1, self.grid_cols - 1)
        self._reset_all_state()
        self._calculate_canvas_size()
        self._render_grid()

    def generate_maze_map(self):
        self.stop_pathfinding_search()
        density = self.obstacle_spawn_density.get()
        self.grid_matrix = [[0] * self.grid_cols for _ in range(self.grid_rows)]
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                if (row, col) not in (self.start_node, self.goal_node):
                    self.grid_matrix[row][col] = 1 if random.random() < density else 0
        
        self._reset_all_state()
        self._render_grid()

    def clear_all_obstacles(self):
        self.stop_pathfinding_search()
        self.grid_matrix = [[0] * self.grid_cols for _ in range(self.grid_rows)]
        self._reset_all_state()
        self._render_grid()

    def _clear_visualization_sets(self):
        self._solution_path_set = set()
        self._explored_nodes_set = set()
        self._frontier_nodes_set = set()

    def _reset_all_state(self):
        self._clear_visualization_sets()
        self.solution_path = []
        self.explored_nodes = []
        self.agent_current_pos = None
        self._replanning_count = 0

    # ─────────────────────────────────────────────────────────────
    #  SEARCH & ANIMATION ENGINE
    # ─────────────────────────────────────────────────────────────

    def start_pathfinding_search(self):
        if self.is_running:
            return
        
        self._clear_visualization_sets()
        self.solution_path = []
        self.explored_nodes = []
        self.agent_current_pos = None
        self.agent_current_step = 0
        self._replanning_count = 0
        
        self.explored_count_display.set("—")
        self.path_cost_display.set("—")
        self.compute_time_display.set("—")
        self.replan_count_display.set("0")
        self._update_status("SEARCHING…", THEME_PRIMARY)
        self._render_grid()

        # Select heuristic function
        heuristic_f = (calculate_manhattan_distance if self.selected_heuristic.get() == "manhattan" 
                      else calculate_euclidean_distance)
        
        # Select algorithm
        if self.selected_algorithm.get() == "astar":
            searcher = AStarSearcher(self.grid_matrix, self.grid_rows, self.grid_cols, 
                                    self.start_node, self.goal_node, heuristic_f)
        else:
            searcher = GreedyBestFirstSearcher(self.grid_matrix, self.grid_rows, self.grid_cols, 
                                              self.start_node, self.goal_node, heuristic_f)

        # Execute search
        start_time = time.perf_counter()
        solution_path, visited_nodes = searcher.compute()
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        self.explored_count_display.set(str(len(visited_nodes)))
        self.compute_time_display.set(f"{elapsed_ms:.1f}")

        if solution_path is None:
            self._update_status("NO PATH!", COLOR_GOAL_NODE)
            messagebox.showwarning("No Path Found",
                "No path exists between Start and Goal.\nTry removing some walls.")
            return

        self.path_cost_display.set(str(len(solution_path) - 1))
        self._update_status("ANIMATING", THEME_SECONDARY)
        self.explored_nodes = visited_nodes
        self.solution_path = solution_path
        self.is_running = True
        self._start_search_btn.set_button_state(False)
        self._stop_search_btn.set_button_state(True)
        self._vis_frame_index = 0
        self._animate_exploration_phase()

    def _animate_exploration_phase(self):
        if not self.is_running:
            return
        
        speed = self.frame_delay_ms.get()
        batch_size = max(1, len(self.explored_nodes) // 80)
        end_idx = min(self._vis_frame_index + batch_size, len(self.explored_nodes))
        
        for i in range(self._vis_frame_index, end_idx):
            node = self.explored_nodes[i]
            if node not in (self.start_node, self.goal_node):
                self._explored_nodes_set.add(node)
                self._refresh_single_cell(*node)
        
        self._vis_frame_index = end_idx
        
        if self._vis_frame_index < len(self.explored_nodes):
            self._scheduled_draw = self.root.after(speed // 5, self._animate_exploration_phase)
        else:
            self._display_solution_and_start_movement()

    def _display_solution_and_start_movement(self):
        for node in self.solution_path:
            if node not in (self.start_node, self.goal_node):
                self._solution_path_set.add(node)
                self._refresh_single_cell(*node)
        
        self.agent_current_step = 0
        self.agent_current_pos = self.start_node
        self._move_agent_step()

    def _move_agent_step(self):
        if not self.is_running:
            return
        
        speed = self.frame_delay_ms.get()

        # Dynamic obstacle spawning
        if self.is_dynamic_enabled.get():
            spawn_prob = self.dynamic_spawn_chance.get()
            for row in range(self.grid_rows):
                for col in range(self.grid_cols):
                    if random.random() < spawn_prob:
                        if ((row, col) not in (self.start_node, self.goal_node, self.agent_current_pos) 
                            and self.grid_matrix[row][col] == 0):
                            self.grid_matrix[row][col] = 1
                            self._solution_path_set.discard((row, col))
                            self._refresh_single_cell(row, col)
            
            if self._check_path_collision():
                self._trigger_replanning()
                return

        if self.agent_current_step >= len(self.solution_path):
            self._finish_search()
            return

        prev_pos = self.agent_current_pos
        self.agent_current_pos = self.solution_path[self.agent_current_step]
        self.agent_current_step += 1

        if prev_pos and prev_pos not in (self.start_node, self.goal_node):
            self._solution_path_set.discard(prev_pos)
            self._refresh_single_cell(*prev_pos)
        
        self._refresh_single_cell(*self.agent_current_pos)

        if self.agent_current_pos == self.goal_node:
            self._finish_search()
            return

        self._scheduled_agent = self.root.after(speed, self._move_agent_step)

    def _check_path_collision(self):
        for node in self.solution_path[self.agent_current_step:]:
            if self.grid_matrix[node[0]][node[1]] == 1:
                return True
        return False

    def _trigger_replanning(self):
        heuristic_f = (calculate_manhattan_distance if self.selected_heuristic.get() == "manhattan" 
                      else calculate_euclidean_distance)
        
        if self.selected_algorithm.get() == "astar":
            searcher = AStarSearcher(self.grid_matrix, self.grid_rows, self.grid_cols, 
                                    self.agent_current_pos or self.start_node, 
                                    self.goal_node, heuristic_f)
        else:
            searcher = GreedyBestFirstSearcher(self.grid_matrix, self.grid_rows, self.grid_cols, 
                                              self.agent_current_pos or self.start_node, 
                                              self.goal_node, heuristic_f)

        start_time = time.perf_counter()
        new_path, new_visited = searcher.compute()
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Clear old path visualization
        for node in list(self._solution_path_set):
            if node not in (self.start_node, self.goal_node):
                self._refresh_single_cell(*node)
        self._solution_path_set.clear()

        if new_path is None:
            self._update_status("PATH BLOCKED!", COLOR_GOAL_NODE)
            self._finish_search(failed=True)
            return

        self._replanning_count += 1
        self.solution_path = new_path
        self.agent_current_step = 1
        self.explored_count_display.set(str(len(new_visited)))
        self.compute_time_display.set(f"{elapsed_ms:.1f}")
        self.path_cost_display.set(str(len(new_path) - 1))
        self.replan_count_display.set(str(self._replanning_count))
        self._update_status(f"RE-PLAN #{self._replanning_count}", "#f59e0b")

        # Visualize new path
        for node in self.solution_path:
            if node not in (self.start_node, self.goal_node):
                self._solution_path_set.add(node)
                self._refresh_single_cell(*node)

        self._scheduled_agent = self.root.after(self.frame_delay_ms.get(), self._move_agent_step)

    def _finish_search(self, failed=False):
        self.is_running = False
        self._start_search_btn.set_button_state(True)
        self._stop_search_btn.set_button_state(False)
        
        if not failed:
            self._update_status("✓  DONE!", COLOR_START_NODE)
            if self.agent_current_pos:
                self._refresh_single_cell(*self.agent_current_pos)
        else:
            self._update_status("✕  FAILED", COLOR_GOAL_NODE)

    def stop_pathfinding_search(self):
        self.is_running = False
        if self._scheduled_draw:
            self.root.after_cancel(self._scheduled_draw)
        if self._scheduled_agent:
            self.root.after_cancel(self._scheduled_agent)
        
        if hasattr(self, '_start_search_btn'):
            self._start_search_btn.set_button_state(True)
        if hasattr(self, '_stop_search_btn'):
            self._stop_search_btn.set_button_state(False)
        if hasattr(self, 'status_display'):
            self._update_status("STOPPED", THEME_TEXT_MUTED)

    def reset_all_state(self):
        self.stop_pathfinding_search()
        self._clear_visualization_sets()
        self.solution_path = []
        self.explored_nodes = []
        self.agent_current_pos = None
        self._replanning_count = 0
        
        self.explored_count_display.set("—")
        self.path_cost_display.set("—")
        self.compute_time_display.set("—")
        self.replan_count_display.set("0")
        self._update_status("READY", THEME_PRIMARY)
        self._render_grid()

    def _update_status(self, status_text, status_color):
        self.status_display.set(status_text)
        if hasattr(self, '_status_label'):
            self._status_label.config(fg=status_color)
        if hasattr(self, '_status_display_pill'):
            self._status_display_pill.config(bg=status_color if status_color not in (THEME_TEXT_MUTED,) else THEME_BORDER_COL)


# ══════════════════════════════════════════════════════════════════
#  APPLICATION ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root_window = tk.Tk()
    root_window.minsize(980, 640)
    application = GridPathfinderApp(root_window)
    root_window.mainloop()