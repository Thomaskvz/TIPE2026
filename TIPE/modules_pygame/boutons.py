import pygame

class Button:
    def __init__(self, x, y, width, height, text, font,
                 color=(100, 100, 100),
                 border_color=(0, 0, 0),
                 hover_border_color=(255, 255, 255),
                 click_color=(80, 80, 80),
                 press_offset=3):

        self.base_rect = pygame.Rect(x, y, width, height)
        self.rect = self.base_rect.copy()

        self.font = font

        self.base_color = color
        self.click_color = click_color
        self.current_color = color

        self.border_color = border_color
        self.hover_border_color = hover_border_color
        self.current_border_color = border_color

        self.press_offset = press_offset
        self.was_pressed = False
        
        self.armed = False

        self.set_text(text)

    def set_text(self, new_text):
        self.text = new_text
        self.text_surface = self.font.render(self.text, True, (255, 255, 255))
        self.text_rect = self.text_surface.get_rect(center=self.rect.center)

    def draw(self, screen):
        pygame.draw.rect(screen, self.current_color, self.rect)
        pygame.draw.rect(screen, self.current_border_color, self.rect, 2)
        screen.blit(self.text_surface, self.text_rect)

    def update(self, mouse_pos, mouse_pressed):
        hovering = self.base_rect.collidepoint(mouse_pos)

        # --- Border ---
        self.current_border_color = (
            self.hover_border_color if hovering else self.border_color
        )

        # --- Fill color ---
        if hovering and mouse_pressed:
            self.current_color = self.click_color
            self.rect.y = self.base_rect.y + self.press_offset
        else:
            self.current_color = self.base_color
            self.rect.y = self.base_rect.y

        # Keep text centered
        self.text_rect.center = self.rect.center

    def is_clicked(self, mouse_pos, mouse_pressed):
        hovering = self.base_rect.collidepoint(mouse_pos)
        clicked = False

        # Mouse button pressed down inside button
        if hovering and mouse_pressed and not self.was_pressed:
            self.armed = True

        # Mouse button released
        if not mouse_pressed and self.was_pressed:
            if hovering and self.armed:
                clicked = True
            self.armed = False  # reset in all cases

        self.was_pressed = mouse_pressed
        return clicked

