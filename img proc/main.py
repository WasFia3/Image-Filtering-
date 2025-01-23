import cv2
import numpy as np


# This program should take an image and
# Implement edge detection Algorithms which are:
# a. Roberts Operator
# b. Sobel Operator
# c. Prewitt operators



print('Welcome to edge detection program (it will take couple seconds)');


def roberts_detection(image):
    # Get the dimensions of the image
    height= len(image) # the number of rows (length)
    width = len(image[0]) # len returns length

    # Initialize an empty list to hold the rows of the output image
    output_image = []

    # Loop height times to create rows
    for _ in range(height):
        row = [0] * width
        output_image.append(row)
        # new image created filled with zeros

    # Now creating Masks
    mask_x = [[1, 0], [0, -1]]
    mask_y = [[0, 1], [-1, 0]]

    # Loop over the image (exclude border pixels)
    for i in range(1, height - 1):
        for j in range(1, width - 1):

            # Calculate gradient in x direction
            gx = 0
            # Loop on the mask array
            for m in range(2):
                for n in range(2):
                    # Multiply the pixel value by the corresponding mask value and add it to the gradient
                    gx += image[i + m][j + n] * mask_x[m][n]

            # Calculate gradient in y direction
            gy = 0
            for m in range(2):
                for n in range(2):
                    # Multiply the pixel value by the corresponding mask value and add it to the gradient
                    gy += image[i + m][j + n] * mask_y[m][n]

            # Compute gradient value
            gradient_value = abs(gx) + abs(gy)

            # Loop and assign the computed value to the corresponding pixel
            output_image[i][j] = gradient_value

    return output_image


def sobel_operator(image):
    # Kernels
    kernel_x = [[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]]

    kernel_y = [[-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]]

    # Get the dimensions of the image
    height = len(image)  # the number of rows (length)
    width = len(image[0])  # len returns length

    # Initialize an empty list to hold the rows of the output image
    output_image = []

    # Loop height times to create rows
    for _ in range(height):
        row = [0] * width
        output_image.append(row)
        # new image created filled with zeros

    # Apply Sobel operator
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Initialize gradients in x and y directions
            gx = 0
            gy = 0

            # Calculate gradient in x direction
            for m in range(3):  # Iterate over rows of the kernel
                for n in range(3):  # Iterate over columns of the kernel
                    # Multiply pixel value by corresponding kernel value and accumulate
                    gx += image[i + m - 1][j + n - 1] * kernel_x[m][n]

            # Calculate gradient in y direction
            for m in range(3):  # Iterate over rows of the kernel
                for n in range(3):  # Iterate over columns of the kernel
                    # Multiply pixel value by corresponding kernel value and accumulate
                    gy += image[i + m - 1][j + n - 1] * kernel_y[m][n]

            # Compute gradient magnitude (approximation)
            gradient_value = abs(gx) + abs(gy)

            # Assign gradient magnitude to corresponding pixel
            output_image[i][j] = gradient_value

    return output_image


def prewitt_operator(image):
    # Prewitt kernels
    kernel_x = [[-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1]]

    kernel_y = [[-1, -1, -1],
                [0, 0, 0],
                [1, 1, 1]]

    # Get the dimensions of the image
    height = len(image)  # the number of rows (length)
    width = len(image[0])  # len returns length

    # Initialize an empty list to hold the rows of the output image
    output_image = []

    # Loop height times to create rows
    for _ in range(height):
        row = [0] * width
        output_image.append(row)
        # new image created filled with zeros


    # Apply Prewitt operator
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Initialize gradients in x and y directions
            gx = 0
            gy = 0

            # Calculate gradient in x direction
            # Loop over kernel
            for m in range(3):
                for n in range(3):
                    # Multiply pixel value by corresponding kernel value and accumulate
                    gx += image[i + m - 1][j + n - 1] * kernel_x[m][n]

            # Calculate gradient in y direction
            # Loop over kernel
            for m in range(3):
                for n in range(3):
                    # Multiply pixel value by corresponding kernel value and accumulate
                    gy += image[i + m - 1][j + n - 1] * kernel_y[m][n]

            # Compute gradient value
            gradient_value = abs(gx) + abs(gy)

            # Assign gradient magnitude to corresponding pixel
            output_image[i][j] = gradient_value

    return output_image


# Load the image in grayscale
img = cv2.imread("sharp.jpg", cv2.IMREAD_GRAYSCALE)
img_hand = cv2.imread("hand.jpg", cv2.IMREAD_GRAYSCALE)
img_cat = cv2.imread("orange_cat_because_i_like_em.PNG", cv2.IMREAD_GRAYSCALE)
# Check if the image is loaded correctly
if img is None:
    print("Error: Unable to load the image.")
else:
    # Apply edge detection functions!
    robert = roberts_detection(img)
    sobel = sobel_operator(img)
    prewitt = prewitt_operator(img)

    robert_hand = roberts_detection(img_hand)
    sobel_hand = sobel_operator(img_hand)
    prewitt_hand = prewitt_operator(img_hand)

    robert_orange_cat = roberts_detection(img_cat)
    sobel_orange_cat = sobel_operator(img_cat)
    prewitt_orange_cat = prewitt_operator(img_cat)


    # Convert the resulting edges to a NumPy array
    edges_robert_array = np.array(robert)
    edges_sobel_array = np.array(sobel)
    edges_prewitt_array = np.array(prewitt)

    edges_robert_hand = np.array(robert_hand)
    edges_sobel_hand = np.array(sobel_hand)
    edges_prewitt_hand = np.array(prewitt_hand)

    edges_robert_orange_cat = np.array(robert_orange_cat)
    edges_sobel_orange_cat = np.array(sobel_orange_cat)
    edges_prewitt_orange_cat = np.array(prewitt_orange_cat)

    # Save the resulting images
    cv2.imwrite('output_robert.jpg', edges_robert_array)
    cv2.imwrite('output_sobel.jpg', edges_sobel_array)
    cv2.imwrite('output_prewitt.jpg', edges_prewitt_array)

    cv2.imwrite('output_robert_hand.jpg', edges_robert_hand)
    cv2.imwrite('output_sobel_hand.jpg', edges_sobel_hand)
    cv2.imwrite('output_prewitt_hand.jpg', edges_prewitt_hand)

    cv2.imwrite('output_robert_orange_cat.jpg', edges_robert_orange_cat)
    cv2.imwrite('output_sobel_orange_cat.jpg', edges_sobel_orange_cat)
    cv2.imwrite('output_prewitt_orange_cat.jpg', edges_prewitt_orange_cat)

    print("Image edited using roberts_detection saved successfully.")
    print("Image edited using sobel_operator saved successfully.")
    print("Image edited using prewitt_operator saved successfully.")

    print("Hand Image edited using roberts_detection saved successfully.")
    print("Hand Image edited using sobel_operator saved successfully.")
    print("Hand Image edited using prewitt_operator saved successfully.")

    print("Orange Image edited using roberts_detection saved successfully.")
    print("Sobel Image edited using roberts_detection saved successfully.")
    print("Sobel Image edited using sobel operator saved successfully.")