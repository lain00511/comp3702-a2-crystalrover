import random
import tkinter as tk

from game_env import GameEnv

"""
Graphical Visualiser for Crystal Rover. You may modify this file if desired.

COMP3702 Assignment 2 "CrystalRover" Support Code

Last updated by vp 09/09/2026
"""

arrows = {
    "UP": "↑",
    "DOWN": "↓",
    "LEFT": "←",
    "RIGHT": "→",
}

class Viewer:

    TILE_W = 32
    TILE_H = 32
    TILE_W_SMALL = 16
    TILE_H_SMALL = 16

    TWEEN_STEPS = 16
    TWEEN_DELAY_MS = 30
    STORM_ANIMATION_INTERVAL_MS = 250

    def __init__(self, game_env, enable_storm_particles=True):
        self.game_env = game_env
        self.enable_storm_particles = enable_storm_particles
        init_state = game_env.get_init_state()
        self.last_state = init_state

        # choose small or large mode
        self.window = tk.Tk()
        screen_width, screen_height = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        if (screen_width < self.game_env.n_cols * self.TILE_W) or (screen_height < self.game_env.n_rows * self.TILE_H):
            small_mode = True
            self.tile_w = self.TILE_W_SMALL
            self.tile_h = self.TILE_H_SMALL
        else:
            small_mode = False
            self.tile_w = self.TILE_W
            self.tile_h = self.TILE_H

        self.window.title("Crystal Rover Visualiser")
        self.window.geometry(f'{self.game_env.n_cols * self.tile_w}x{self.game_env.n_rows * self.tile_h}')

        self.canvas = tk.Canvas(self.window)
        self.canvas.configure(bg="white")
        self.canvas.pack(fill="both", expand=True)

        # Add info label
        self.info_label = tk.StringVar()
        crystals_remaining = self.game_env.min_samples - sum(game_env.get_init_state().crystal_status)
        self.info_label.set("crystals left = " + str(crystals_remaining) + ", storm " + str(self.game_env.storm_directional_cost) + " " + str(arrows.get(self.game_env.storm_direction)))
        self.label = tk.Label(self.window, textvariable=self.info_label, bg="white")
        self.label.place(x=0, y=0)

        # load images
        if small_mode:
            self.tile_ground = tk.PhotoImage(file='gui_assets/game_tile_ground_small.png')
            self.tile_rock = tk.PhotoImage(file='gui_assets/game_tile_rock_small.png')
            self.tile_crater= tk.PhotoImage(file='gui_assets/game_tile_crater_small.png')
            self.tile_crystal = tk.PhotoImage(file='gui_assets/game_tile_crystal_small.png')
            self.tile_launch = tk.PhotoImage(file='gui_assets/game_tile_launch_small.png')
            self.tile_rover = tk.PhotoImage(file='gui_assets/game_tile_rover_small.png')
            self.tile_lava = tk.PhotoImage(file='gui_assets/game_tile_lava_small.png')
        else:
            self.tile_ground = tk.PhotoImage(file='gui_assets/game_tile_ground.png')
            self.tile_rock = tk.PhotoImage(file='gui_assets/game_tile_rock.png')
            self.tile_crater= tk.PhotoImage(file='gui_assets/game_tile_crater.png')
            self.tile_crystal = tk.PhotoImage(file='gui_assets/game_tile_crystal.png')
            self.tile_launch = tk.PhotoImage(file='gui_assets/game_tile_launch.png')
            self.tile_rover = tk.PhotoImage(file='gui_assets/game_tile_rover.png')
            self.tile_lava = tk.PhotoImage(file='gui_assets/game_tile_lava.png')

        # draw background (all permanent features, i.e. everything except rover and crystals)
        for r in range(self.game_env.n_rows):
            for c in range(self.game_env.n_cols):
                if self.game_env.grid_data[r][c] == GameEnv.GROUND_TILE:
                    self.canvas.create_image((c * self.tile_w), (r * self.tile_h), image=self.tile_ground, anchor=tk.NW)
                elif self.game_env.grid_data[r][c] == GameEnv.ROCK_TILE:
                    self.canvas.create_image((c * self.tile_w), (r * self.tile_h), image=self.tile_rock, anchor=tk.NW)
                elif self.game_env.grid_data[r][c] == GameEnv.CRATER_TILE:
                    self.canvas.create_image((c * self.tile_w), (r * self.tile_h), image=self.tile_crater, anchor=tk.NW)
                elif self.game_env.grid_data[r][c] == GameEnv.LAVA_TILE:
                    self.canvas.create_image((c * self.tile_w), (r * self.tile_h), image=self.tile_lava, anchor=tk.NW)
             

        # draw all the launch positions
        for (r, c) in self.game_env.launch_positions:
            self.canvas.create_image((c * self.tile_w), (r * self.tile_h), image=self.tile_launch, anchor=tk.NW)

        # draw crystals for initial state
        self.crystal_images = None
        self.draw_crystals(init_state)

        # draw rover position for initial state
        self.rover_image = None
        self.draw_rover(init_state.row, init_state.col)

        self.max_storm_particles = 10 if self.game_env.n_cols * self.game_env.n_rows > 100 else 5

        self.storm_particles = []

        # Added to stop ugly error messages when closing the window while the storm animation is running
        self.storm_animation_running = False
        self.storm_loop_scheduled = False
        self.storm_after_id = None
        self.tween_after_id = None
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        self.window.update()
        self.tween_finished = tk.BooleanVar(value=True)

        # cache canvas size instead of calling winfo_width/height every frame
        self._canvas_width = self.canvas.winfo_width()
        self._canvas_height = self.canvas.winfo_height()
        self.canvas.bind('<Configure>', self._on_canvas_resize)

        if self.enable_storm_particles:
            self._init_storm_particles()
            self._start_storm_animation()


    def update_state(self, state):
        crystals_remaining = self.game_env.min_samples - sum(state.crystal_status)
        self.info_label.set("crystals remaining = " + str(crystals_remaining) + ", storm " + str(self.game_env.storm_directional_cost) + " " + str(arrows.get(self.game_env.storm_direction)))

        prev_state = self.last_state
        self.last_state = state

        for c in self.crystal_images:
            self.canvas.delete(c)
        self.draw_crystals(state)

        self.tween_finished.set(False)
        self._tween_rover(prev_state, state)
        self.window.wait_variable(self.tween_finished)


    def draw_crystals(self, state):
        self.crystal_images = []
        for (gr, gc), status in zip(self.game_env.crystal_positions, state.crystal_status):
            if status == 0:
                img = self.canvas.create_image((gc * self.tile_w), (gr * self.tile_h), image=self.tile_crystal, anchor=tk.NW)
                self.crystal_images.append(img)

    def draw_rover(self, row, col):
        x, y = col * self.tile_w, row * self.tile_h
        if self.rover_image is None:
            self.rover_image = self.canvas.create_image(x, y, image=self.tile_rover, anchor=tk.NW)
        else:
            self.canvas.coords(self.rover_image, x, y)
        self.current_rover_position = (row, col)

    def _init_storm_particles(self):
        self.storm_particles = []
        for _ in range(self.max_storm_particles):
            self._spawn_storm_particle()

    def _spawn_storm_particle(self):
        width = max(self._canvas_width, 1)
        height = max(self._canvas_height, 1)

        x = random.uniform(0, width)
        y = random.uniform(0, height)
        if self.game_env.storm_direction == 'LEFT':
            vx = -random.uniform(5.0, 9.0)
            vy = random.uniform(-2.2, 2.2)
        elif self.game_env.storm_direction == 'RIGHT':
            vx = random.uniform(5.0, 9.0)
            vy = random.uniform(-2.2, 2.2)
        elif self.game_env.storm_direction == 'UP':
            vx = random.uniform(-2.2, 2.2)
            vy = -random.uniform(5.0, 9.0)
        elif self.game_env.storm_direction == 'DOWN':
            vx = random.uniform(-2.2, 2.2)
            vy = random.uniform(5.0, 9.0)
        else:
            vx = vy = 0.0

        size = random.uniform(3.2, 5.8)
        particle_id = self.canvas.create_oval(x, y, x + size, y + size, fill="#c28f74", outline='')
        self.storm_particles.append({'id': particle_id, 'x': x, 'y': y, 'vx': vx, 'vy': vy, 'size': size})

    def _start_storm_animation(self):
        if getattr(self, 'storm_animation_running', False):
            return
        self.storm_animation_running = True
        self.storm_loop_scheduled = False
        self._animate_storm_loop()

    def _animate_storm_loop(self):
        if not self.window.winfo_exists():
            return

        if not getattr(self, 'storm_animation_running', False):
            return

        self.animate_storm_particles()
        self.storm_loop_scheduled = True
        self.storm_after_id = self.window.after(
                self.STORM_ANIMATION_INTERVAL_MS,
                self._animate_storm_loop
            )

    def animate_storm_particles(self):
        width = max(self._canvas_width, 1)
        height = max(self._canvas_height, 1)

        for particle in list(self.storm_particles):
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            self.canvas.move(particle['id'], particle['vx'], particle['vy'])

            if (particle['x'] < -10 or particle['x'] > width + 10 or
                    particle['y'] < -10 or particle['y'] > height + 10):
                self.canvas.delete(particle['id'])
                self.storm_particles.remove(particle)
                if len(self.storm_particles) < self.max_storm_particles:
                    self._spawn_storm_particle()

    def _tween_rover(self, start_state, target_state, step=0):
        if not self.window.winfo_exists():
            return

        if step > self.TWEEN_STEPS:
            self.tween_finished.set(True)
            return

        r = start_state.row + (step / self.TWEEN_STEPS) * (
            target_state.row - start_state.row
        )
        c = start_state.col + (step / self.TWEEN_STEPS) * (
            target_state.col - start_state.col
        )

        self.draw_rover(r, c)

        self.tween_after_id = self.window.after(
            int(self.TWEEN_DELAY_MS),
            lambda: self._tween_rover(start_state, target_state, step + 1)
        )

    # try to exit gracefully when the window is closed while the storm animation is running
    def _on_close(self):
        self.storm_animation_running = False

        if self.storm_after_id is not None:
            try:
                self.window.after_cancel(self.storm_after_id)
            except tk.TclError:
                pass

        if self.tween_after_id is not None:
            try:
                self.window.after_cancel(self.tween_after_id)
            except tk.TclError:
                pass

        self.window.destroy()

    def _on_canvas_resize(self, event):
        self._canvas_width = event.width
        self._canvas_height = event.height