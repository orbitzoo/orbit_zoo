import pygame

from play3d.three_d import Camera

base_step = 0.1


def handle_camera_with_keys():
    keys = pygame.key.get_pressed()

    camera_position = Camera.get_instance()
    if keys[pygame.K_UP]:
        camera_position['y'] += base_step
        # print('y', camera_position['y'])

    if keys[pygame.K_DOWN]:
        camera_position['y'] -= base_step
        # print('y', camera_position['y'])

    if keys[ord('w')]:
        camera_position['z'] -= base_step
        # print('z', camera_position['z'])

    if keys[ord('s')]:
        camera_position['z'] += base_step
        # print('z', camera_position['z'])

    if keys[ord('d')]:
        camera_position['x'] += base_step
        # print('x', camera_position['x'])

    if keys[ord('a')]:
        camera_position['x'] -= base_step
        # print('x', camera_position['x'])

    if keys[ord('q')]:
        camera_position.rotate('y', -2)

    if keys[ord('e')]:
        camera_position.rotate('y', 2)

    if keys[ord('r')]:
        camera_position.rotate('x', 2)

    if keys[ord('t')]:
        camera_position.rotate('x', -2)

    # if keys[pygame.K_UP] or keys[pygame.K_DOWN] or keys[ord('w')] or keys[ord('s')] or keys[ord('d')] or keys[ord('a')] or keys[ord('q')] or keys[ord('e')] or keys[ord('r')] or keys[ord('t')]:
    #     print(f'{camera_position['x']}, {camera_position['y']}, {camera_position['z']}')