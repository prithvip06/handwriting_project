import numpy as np
import pygame
import sys
import cv2
import torch
import random
from tkinter import Tk, filedialog
from cnn_model import LetterCNN

# ── Colors ────────────────────────────────────────────────
BG            = (253, 246, 227)
CANVAS_COL    = (255, 255, 255)
CHARCOAL      = (45,  45,  45)
BLUE          = (74,  144, 217)
GREEN         = (92,  184, 92)
RED           = (224, 92,  92)
YELLOW        = (245, 166, 35)
NAVY          = (26,  42,  74)
MUTED         = (138, 127, 114)
SHADOW_CANVAS = (192, 184, 154)
SHADOW_BTN    = (180, 160, 120)
LINE_PAPER    = (232, 224, 204)

# ── Load CNN weights ──────────────────────────────────────
def load_model():
    try:
        model = LetterCNN()
        model.load_state_dict(torch.load('cnn_weights.pth', map_location='cpu'))
        model.eval()
        print("CNN weights loaded!")
        return model
    except FileNotFoundError:
        print("No CNN weights found — run cnn_train.py first!")
        sys.exit()

# ── Preprocessing ─────────────────────────────────────────
def preprocess_drawn(surface):
    raw  = pygame.surfarray.array3d(surface)
    raw  = np.transpose(raw, (1, 0, 2))
    gray = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
    coords = cv2.findNonZero(gray)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    if w < 5 or h < 5:
        return None
    cropped = gray[y:y+h, x:x+w]
    resized = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)
    resized = cv2.flip(resized, 0)
    resized = cv2.rotate(resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
    tensor  = torch.tensor(resized / 255.0, dtype=torch.float32)
    tensor  = tensor.unsqueeze(0).unsqueeze(0)
    return tensor

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    cropped = thresh[y:y+h, x:x+w]
    resized = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)
    resized = cv2.flip(resized, 0)
    resized = cv2.rotate(resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
    tensor  = torch.tensor(resized / 255.0, dtype=torch.float32)
    tensor  = tensor.unsqueeze(0).unsqueeze(0)
    return tensor

def predict(tensor, model):
    with torch.no_grad():
        outputs    = model(tensor)
        probs      = torch.softmax(outputs, dim=1)
        pred       = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item() * 100
        letter     = chr(pred + ord('a')).upper()
    return letter, confidence

# ── Drawing helpers ───────────────────────────────────────
def gen_sketchy_pts(x, y, w, h, wobble=2):
    return [
        (x + random.randint(-wobble, wobble), y + random.randint(-wobble, wobble)),
        (x + w + random.randint(-wobble, wobble), y + random.randint(-wobble, wobble)),
        (x + w + random.randint(-wobble, wobble), y + h + random.randint(-wobble, wobble)),
        (x + random.randint(-wobble, wobble), y + h + random.randint(-wobble, wobble)),
    ]

def draw_sketchy_rect(surface, color, pts, width=3):
    pygame.draw.polygon(surface, color, pts, width)

def draw_doodle_button(surface, text, rect, color, font, hover=False):
    x, y, w, h = rect
    offset = 2 if hover else 0
    shadow = pygame.Rect(x + 4, y + 4, w, h)
    pygame.draw.rect(surface, SHADOW_BTN, shadow, border_radius=10)
    face = pygame.Rect(x, y + offset, w, h)
    pygame.draw.rect(surface, color, face, border_radius=10)
    pygame.draw.rect(surface, CHARCOAL, face, width=2, border_radius=10)
    label = font.render(text, True, (255, 255, 255))
    surface.blit(label, (
        face.centerx - label.get_width() // 2,
        face.centery - label.get_height() // 2
    ))

def draw_squiggle(surface, color, x, y, width, amplitude=3, segments=20):
    pts = []
    for i in range(segments + 1):
        px = x + int(i * width / segments)
        py = y + (amplitude if i % 2 == 0 else -amplitude)
        pts.append((px, py))
    pygame.draw.lines(surface, color, False, pts, 2)

def draw_notebook_lines(surface, x, y, w, h):
    for ly in range(y, y + h, 30):
        pygame.draw.line(surface, LINE_PAPER, (x, ly), (x + w, ly), 1)

def open_file():
    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
    )
    root.destroy()
    return path

# ── Main ──────────────────────────────────────────────────
def main():
    model = load_model()
    pygame.init()

    WIDTH, HEIGHT = 620, 650
    CANVAS_SIZE   = 360
    CANVAS_X      = 120
    CANVAS_Y      = 80

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Handwriting Reader — CNN")
    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

    # Fonts
    try:
        font_title   = pygame.font.Font("Caveat-Bold.ttf", 36)
        font_btn     = pygame.font.Font("Caveat-Bold.ttf", 26)
        font_hint    = pygame.font.Font("Caveat-Regular.ttf", 20)
        font_predict = pygame.font.Font("Caveat-Bold.ttf", 100)
        font_conf    = pygame.font.Font("Caveat-Regular.ttf", 24)
        font_label   = pygame.font.Font("Caveat-Regular.ttf", 20)
    except:
        font_title   = pygame.font.SysFont('Arial', 32, bold=True)
        font_btn     = pygame.font.SysFont('Arial', 22, bold=True)
        font_hint    = pygame.font.SysFont('Arial', 18)
        font_predict = pygame.font.SysFont('Arial', 100, bold=True)
        font_conf    = pygame.font.SysFont('Arial', 22)
        font_label   = pygame.font.SysFont('Arial', 18)

    canvas = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
    canvas.fill((0, 0, 0))

    canvas_border_pts = gen_sketchy_pts(
        CANVAS_X - 3, CANVAS_Y - 3,
        CANVAS_SIZE + 6, CANVAS_SIZE + 6, wobble=3
    )
    canvas_shadow_pts = gen_sketchy_pts(
        CANVAS_X + 1, CANVAS_Y + 4,
        CANVAS_SIZE + 6, CANVAS_SIZE + 6, wobble=2
    )

    btn_clear   = (100, 468, 120, 42)
    btn_predict = (250, 468, 120, 42)
    btn_upload  = (400, 468, 120, 42)

    drawing           = False
    last_pos          = None
    prediction_letter = None
    prediction_conf   = None

    clock = pygame.time.Clock()

    while True:
        screen.fill(BG)
        draw_notebook_lines(screen, 0, 0, WIDTH, HEIGHT)

        # Title
        title_surf = font_title.render("✏  Handwriting Reader — CNN  ✏", True, CHARCOAL)
        screen.blit(title_surf, (WIDTH // 2 - title_surf.get_width() // 2, 20))

        # Canvas shadow
        pygame.draw.polygon(screen, SHADOW_CANVAS, canvas_shadow_pts)
        canvas_bg = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
        canvas_bg.fill(CANVAS_COL)
        draw_notebook_lines(canvas_bg, 0, 0, CANVAS_SIZE, CANVAS_SIZE)
        screen.blit(canvas_bg, (CANVAS_X, CANVAS_Y))
        screen.blit(canvas, (CANVAS_X, CANVAS_Y))
        draw_sketchy_rect(screen, CHARCOAL, canvas_border_pts, width=3)

        # Hint
        hint = font_hint.render("draw a letter above  ↑", True, MUTED)
        screen.blit(hint, (WIDTH // 2 - hint.get_width() // 2, CANVAS_Y + CANVAS_SIZE + 8))

        # Buttons
        mx, my = pygame.mouse.get_pos()
        draw_doodle_button(screen, "Clear",    btn_clear,   RED,    font_btn,
                           hover=pygame.Rect(*btn_clear).collidepoint(mx, my))
        draw_doodle_button(screen, "Predict!", btn_predict, GREEN,  font_btn,
                           hover=pygame.Rect(*btn_predict).collidepoint(mx, my))
        draw_doodle_button(screen, "Upload",   btn_upload,  YELLOW, font_btn,
                           hover=pygame.Rect(*btn_upload).collidepoint(mx, my))

        # Prediction display — below buttons
        if prediction_letter:
            pred_y = 525

            label_surf = font_label.render("Prediction:", True, MUTED)
            screen.blit(label_surf, (40, pred_y))

            letter_surf = font_predict.render(prediction_letter, True, NAVY)
            screen.blit(letter_surf, (40, pred_y + 20))

            draw_squiggle(screen, BLUE,
                          40, pred_y + 20 + letter_surf.get_height() - 8,
                          letter_surf.get_width())

            conf_surf = font_conf.render(f"{prediction_conf:.1f}% confidence", True, CHARCOAL)
            screen.blit(conf_surf, (40 + letter_surf.get_width() + 16, pred_y + 40))

            if prediction_conf > 90:
                star_surf = font_hint.render("⭐ confident!", True, YELLOW)
                screen.blit(star_surf, (40 + letter_surf.get_width() + 16, pred_y + 70))

        # Cursor
        if (CANVAS_X <= mx <= CANVAS_X + CANVAS_SIZE and
                CANVAS_Y <= my <= CANVAS_Y + CANVAS_SIZE):
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_CROSSHAIR)
        else:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                ex, ey = event.pos
                cx, cy = ex - CANVAS_X, ey - CANVAS_Y

                if 0 <= cx <= CANVAS_SIZE and 0 <= cy <= CANVAS_SIZE:
                    drawing  = True
                    last_pos = (cx, cy)

                if pygame.Rect(*btn_clear).collidepoint(ex, ey):
                    canvas.fill((0, 0, 0))
                    prediction_letter = None
                    prediction_conf   = None

                elif pygame.Rect(*btn_predict).collidepoint(ex, ey):
                    tensor = preprocess_drawn(canvas)
                    if tensor is not None:
                        prediction_letter, prediction_conf = predict(tensor, model)
                    else:
                        prediction_letter = "?"
                        prediction_conf   = 0.0

                elif pygame.Rect(*btn_upload).collidepoint(ex, ey):
                    path = open_file()
                    if path:
                        tensor = preprocess_image(path)
                        if tensor is not None:
                            prediction_letter, prediction_conf = predict(tensor, model)
                        else:
                            prediction_letter = "?"
                            prediction_conf   = 0.0

            elif event.type == pygame.MOUSEBUTTONUP:
                drawing  = False
                last_pos = None

            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    ex, ey = event.pos
                    cx, cy = ex - CANVAS_X, ey - CANVAS_Y
                    if 0 <= cx <= CANVAS_SIZE and 0 <= cy <= CANVAS_SIZE:
                        if last_pos:
                            pygame.draw.line(canvas, (255, 255, 255),
                                             last_pos, (cx, cy), 18)
                        last_pos = (cx, cy)
                    else:
                        last_pos = None

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()