import numpy as np


def getBackground(images, labels, img_size):
    """
        Separate the background_image to 16 blocks
    """
    background_images = []
    background_blocks = []
    input_h, input_w = img_size[0], img_size[1]
    block_h, block_w = input_h // 4, input_w // 4
    i_block = 0
    current_h = 0
    current_w = 0
    back_img = np.full((input_h, input_w, 3), 114, dtype=np.uint8)

    for image, label in zip(images, labels):

        xmin = label[0][:, 0].min()
        ymin = label[0][:, 1].min()
        xmax = label[0][:, 2].max()
        ymax = label[0][:, 3].max()
        (h, w, c) = image.shape[:3]
        if i_block == 0:
            back_img = np.full((input_h, input_w, c), 114, dtype=np.uint8)
            current_w = i_block * block_w
            current_h = i_block * block_h
        if xmin > block_w and ymin > block_h:
            back_img[current_h: (current_h+block_h), current_w: (current_w+block_w)] = image[:block_h, :block_w]
            background_blocks.append(image[:block_h, :block_w])
            i_block += 1
            current_w = (i_block % 4) * block_w
            current_h = (i_block // 4) * block_h
        elif (w - xmax) > block_w and (h - ymax) > block_h:
            back_img[current_h: (current_h + block_h), current_w: (current_w + block_w)] = image[-block_h:, -block_w:]
            background_blocks.append(image[:block_h, :block_w])
            i_block += 1
            current_w = (i_block % 4) * block_w
            current_h = (i_block // 4) * block_h
        if i_block == 15:
            background_images.append(back_img)
            i_block = 0
    return background_images, background_blocks
