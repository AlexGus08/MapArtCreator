import cv2
import numpy as np
import os
from collections import defaultdict
BLOCK_SIZE = 16
DEFAULT_MAP_SIZE = 64
LIST_BLOCKS = list()
def load_color_palettes(palette_dir="palettes"):
    palettes = defaultdict(dict)
    for palette_name in os.listdir(palette_dir):
        palette_path = os.path.join(palette_dir, palette_name)
        if os.path.isdir(palette_path):
            for shade_subdir in os.listdir(palette_path):
                shade_path = os.path.join(palette_path, shade_subdir)
                if os.path.isdir(shade_path):
                    for block_image_name in os.listdir(shade_path):
                        block_image_path = os.path.join(shade_path, block_image_name)
                        if block_image_path.endswith(".png"):
                            block_image = cv2.imread(block_image_path)
                            avg_color = cv2.mean(block_image)[:3]
                            if shade_subdir not in palettes[palette_name]:
                                palettes[palette_name][shade_subdir] = []
                            palettes[palette_name][shade_subdir].append((block_image_path, avg_color))
    return palettes

def find_closest_palette_color(palettes, color, blacklist):
    closest_color = None
    min_distance = float("inf")
    for palette_name, shades in palettes.items():
        for shade, blocks in shades.items():
            for block_image_path, avg_color in blocks:
                if block_image_path not in blacklist:
                    distance = np.sqrt(np.sum((np.array(avg_color) - np.array(color)) ** 2))
                    if distance < min_distance:
                        min_distance = distance
                        closest_color = (block_image_path, avg_color)
    return closest_color

def create_blacklist_interface(palettes):
    font = cv2.FONT_HERSHEY_SIMPLEX
    blacklisted = set()

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for block_image_path, (block_x, block_y) in display_positions:
                if block_x < x < block_x + 16 and block_y < y < block_y + 16:
                    if block_image_path in blacklisted:
                        blacklisted.remove(block_image_path)
                        cv2.putText(display_image, "O", (block_x, block_y - 5), font, 0.7, (0, 255, 0), 2)
                    else:
                        blacklisted.add(block_image_path)
                        cv2.putText(display_image, "O", (block_x, block_y - 5), font, 0.7, (0, 0, 255), 2)

    display_image = np.ones((800, 2000, 3), dtype=np.uint8) * 255
    display_positions = []

    y_offset = 20
    x_offset = 20
    max_width = display_image.shape[1] - 40  # Учитываем отступы
    group_counter = 0  # Счетчик групп

    for palette_name, shades in palettes.items():
        # Проверка, нужно ли смещать
        if group_counter > 0 and group_counter % 12 == 0:
            x_offset += 350  # Сдвинуть вправо на 200 пикселей
            y_offset = 20    # Вернуться к началу по вертикали

        cv2.putText(display_image, palette_name, (x_offset, y_offset), font, 0.7, (0, 0, 0), 2)
        y_offset += 30
        group_counter += 1  # Увеличиваем счетчик после добавления названия палитры

        for shade, blocks in shades.items():
            for block_image_path, avg_color in blocks:
                if not os.path.isfile(block_image_path):
                    print(f"Ошибка: Файл не существует - {block_image_path}")
                    continue

                img = cv2.imread(block_image_path)

                if img is None:
                    print(f"Ошибка: Не удалось загрузить изображение {block_image_path}. Проверьте путь.")
                    continue

                img = cv2.resize(img, (16, 16))

                # Проверка высоты
                if y_offset + 16 > display_image.shape[0] - 20:  # Если предел высоты превышен
                    print("Ошибка: Достигнут предел высоты для размещения блоков. Блоки не могут быть добавлены.")
                    break

                # Отображение изображения
                display_image[y_offset:y_offset + 16, x_offset:x_offset + 16] = img
                display_positions.append((block_image_path, (x_offset, y_offset)))

                x_offset += 20  # Сдвигаем вправо для следующего блока
                # Отображаем статус (X или O)
                status_x = x_offset - 10

            y_offset += 30  # Отступ после каждого оттенка
            x_offset = 20 + 350 * (group_counter // 12) if group_counter % 12 != 0 else x_offset  # Сбросить x_offset для нового оттенка, только если не кратно 12

        for shade, blocks in shades.items():
            for block_image_path in blacklisted:
                if block_image_path in blacklisted:
                    cv2.putText(display_image, "O", (status_x, y_offset - 5), font, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_image, "O", (status_x, y_offset - 5), font, 0.7, (0, 0, 0), 2)
    # Установка коллбэка и отображение окна
    cv2.namedWindow("Blacklist Selector")
    cv2.setMouseCallback("Blacklist Selector", click_event)

    while True:
        cv2.imshow("Blacklist Selector", display_image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    #cv2.destroyAllWindows()

    return blacklisted


import cv2
import numpy as np
import os


def create_mapart(image_path, output_dir, palette_dir="palettes", blocks_dir="blocks", map_width=DEFAULT_MAP_SIZE,
                  map_height=DEFAULT_MAP_SIZE):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")

    palettes = load_color_palettes(palette_dir)
    blacklist = create_blacklist_interface(palettes)

    # Подгоняем изображение под нужный размер
    img_height, img_width = image.shape[:2]
    aspect_ratio = img_width / img_height
    new_width = map_width * BLOCK_SIZE
    new_height = int(new_width / aspect_ratio)
    if new_height > map_height * BLOCK_SIZE:
        new_height = map_height * BLOCK_SIZE
        new_width = int(new_height * aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height))

    # Добавляем черные полосы для подгонки под размер карты
    top = (map_height * BLOCK_SIZE - new_height) // 2
    bottom = map_height * BLOCK_SIZE - new_height - top
    left = (map_width * BLOCK_SIZE - new_width) // 2
    right = map_width * BLOCK_SIZE - new_width - left

    resized_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    output_image = np.zeros((map_height * BLOCK_SIZE, map_width * BLOCK_SIZE, 3), dtype=np.uint8)
    block_image_window = np.zeros_like(output_image)

    for x in range(0, map_width * BLOCK_SIZE, BLOCK_SIZE):
        for y in range(0, map_height * BLOCK_SIZE, BLOCK_SIZE):
            block = resized_image[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
            avg_color = tuple(map(int, cv2.mean(block)[:3]))
            closest_color = find_closest_palette_color(palettes, avg_color, blacklist)

            if closest_color:
                block_image_path, avg_color = closest_color
                block_image = cv2.imread(block_image_path)
                block_image = cv2.resize(block_image, (BLOCK_SIZE, BLOCK_SIZE))
                output_image[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE] = block_image

                # Сравнение пути для получения соответствующего изображения из blocks
                path_parts = block_image_path.split("/")
                subgroup = path_parts[-3]  # вторая часть пути
                corresponding_block_path = os.path.join(blocks_dir, subgroup)
                print(corresponding_block_path)
                # Извлекаем первое изображение из соответствующей папки
                if os.path.exists(corresponding_block_path):
                    block_images = [img for img in os.listdir(corresponding_block_path) if img.endswith('.png')]
                    if block_images:
                        first_block_image_path = os.path.join(corresponding_block_path, block_images[0])
                        corresponding_image = cv2.imread(first_block_image_path)
                        corresponding_image = cv2.resize(corresponding_image, (BLOCK_SIZE, BLOCK_SIZE))
                        block_image_window[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE] = corresponding_image

                print(
                    f"Pixel ({x // BLOCK_SIZE + 1},{y // BLOCK_SIZE + 1}): Color {avg_color} -> Block {block_image_path}")

    # Создаем окна с результатом
    cv2.imshow("MapArt showMap", output_image)
    cv2.imshow("Corresponding Blocks", block_image_window)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(os.path.join(output_dir, "mapart.png"), output_image)
    cv2.imwrite(os.path.join(output_dir, "mapart_of_block.png"), block_image_window)

def display_image(image_path, window_name="Image"):
    """Отображает изображение из файла по указанному пути в отдельном окне."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")

    cv2.imshow(window_name, image)
    cv2.waitKey(0)

# display_image("image.png", "list_of_block")
# display_image("image2.png", "list_of_block")
# display_image("image3.png", "list_of_block")
# display_image("image4.png", "list_of_block")



create_mapart("image52.jpeg", "output_directory", "palettes", map_width=256, map_height=256)

