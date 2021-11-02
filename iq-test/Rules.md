# Rules
- (Line) Addition 1
  - color1 + color1 = Color 2
  - Color2 + Color2 = Color 1
  - Color1 + Color2 = Color 2
- (Dots 2x.) Flip 1
  - Flip First picture on both axis's
  - Ignore Picture 2
- (Figure in square) Movement 1
  - Move object same direction for X pixels
    - Calculate difference in Pixels, use to predict next step
- (Dots 3x.) Rotation 2
  - Rotate 45 degrees counterclockwise
- (Dots 4x4.) Pop-up 1
  - Insert colorful circle around in surounding fields
  - Opposite colors cancel each other out 
- (Cross Dot) Addition 2 (Color dots)
  - white + white = White
  - Black + black = black
  - white + Black = Black
- (Star) Color rotation
  - Triangle 1 moves to xy position in star formation
    - 2 steps clockwise
  - Triangle 2 moves to xy position in star formation
    - 1 step clockwise
  - If meet, color turns white
- (Figure) Bordersoftener
  - Black outline moves toward center dot
- (dots 5x5.) Subtraction
  - Dots in one line
    - Subtract one at each step either vertically or horisontaly


# Model Setup
- which picture does the test set look like
- Then apply Rules (e.g. Star Rules)