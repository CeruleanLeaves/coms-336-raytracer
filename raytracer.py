import numpy as np
from PIL import Image

def main():

    width = 400
    height = 225

    image = np.zeros(shape=(height, width, 3), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            r = x / (width - 1)
            g = y / (height - 1)
            b = .5
            image[y, x, 0] = r
            image[y, x, 1] = g
            image[y, x, 2] = b

    image_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)

    rendered_image = Image.fromarray(image_uint8, mode='RGB')
    rendered_image.save('output.png')
    print('yayy')

if __name__ == '__main__':
    main()