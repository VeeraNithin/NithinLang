nithin
lang+ english

game_start(800, 600, "My First Game", 60)
player_x = 400

def update(dt):
    global player_x
    yadi game_key("right"): player_x = player_x + 5
    yadi game_key("left"): player_x = player_x - 5

def draw():
    game_clear("black")
    game_draw("circle", player_x, 300, radius=30, color="cyan")

game_loop(update, draw)
end nithin
