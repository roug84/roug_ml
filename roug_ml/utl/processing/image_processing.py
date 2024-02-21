import cv2
import numpy as np
from matplotlib import pyplot as plt


class ImageProcessing:
    def __init__(self):
        pass

    def crop_image(self, image: np.ndarray, in_show: bool = False) -> np.ndarray:
        """
        Crop a region of interest from the input image.

        :param image: The input image
        :param in_show: True to display image
        :return: The cropped region of the image.
        """
        top = 30
        bottom = 160  # 120
        left = 40  # 50
        right = 160

        image = image[top:bottom, left:right]

        if in_show:
            plt.imshow(image, cmap="gray")
            plt.title("crop_image")
            plt.show()
        return image

    def normalize_image(self, image: np.ndarray, in_show: bool = False) -> np.ndarray:
        """
        Normalize pixel values in the input image to the range [0, 255].

        :param image: The input image
        :param in_show: True to display image
        :return: The normalized image
        """
        image_float = image.astype(float)
        min_val = image_float.min()
        max_val = image_float.max()
        image_normalized = 255 * (image_float - min_val) / (max_val - min_val)
        image_normalized = image_normalized.astype(np.uint8)
        if in_show:
            plt.imshow(image_normalized, cmap="gray")
            plt.title("normalize_image")
            plt.show()
        return image_normalized

    def adjust_contrast_brightness(
        self, image: np.ndarray, alpha: float, beta: float, in_show: bool = False
    ) -> np.ndarray:
        """
        Adjust contrast and brightness of the input image.

        :param image: The input image
        :param alpha: Contrast control (1.0 is the original image)
        :param beta: Brightness control (0 is the original image)
        :param in_show: True to display image
        :return: The adjusted image
        """
        enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        if in_show:
            plt.imshow(enhanced_image, cmap="gray")
            plt.title("adjust_contrast_brightness")
            plt.show()
        return enhanced_image

    def equalize_histogram(
        self, image: np.ndarray, in_show: bool = False
    ) -> np.ndarray:
        """
        Apply histogram equalization to the input image.

        :param image: The input image
        :param in_show: True to display image
        :return: The equalized image
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        equalized_image = cv2.equalizeHist(image)
        if in_show:
            plt.imshow(equalized_image, cmap="gray")
            plt.title("equalize_histogram")
            plt.show()
        return equalized_image

    def gamma_correction(
        self, image: np.ndarray, gamma: float, in_show: bool = False
    ) -> np.ndarray:
        """
        Apply gamma correction to the input image.

        :param image: The input image
        :param gamma: Gamma correction factor
        :param in_show: True to display image
        :return: The gamma-corrected image
        """
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        lut = np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8
        )
        gamma_corrected_image = cv2.LUT(image, lut)
        if in_show:
            plt.imshow(gamma_corrected_image, cmap="gray")
            plt.title("gamma_correction")
            plt.show()
        return gamma_corrected_image

    def erode_image(
        self,
        image: np.ndarray,
        kernel_size: int,
        iterations: int,
        in_show: bool = False,
    ) -> np.ndarray:
        """
        Apply erosion to the input image.

        :param image: The input image
        :param kernel_size: Size of the erosion kernel
        :param iterations: Number of iterations
        :param in_show: True to display image
        :return: The eroded image
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded_image = cv2.erode(image, kernel, iterations=iterations)
        if in_show:
            plt.imshow(eroded_image, cmap="gray")
            plt.title("erode_image")
            plt.show()
        return eroded_image

    def dilate_image(
        self,
        image: np.ndarray,
        kernel_size: int,
        iterations: int,
        in_show: bool = False,
    ) -> np.ndarray:
        """
        Apply dilation to the input image.

        :param image: The input image
        :param kernel_size: Size of the dilation kernel
        :param iterations: Number of iterations
        :param in_show: True to display image
        :return: The dilated image
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_image = cv2.dilate(image, kernel, iterations=iterations)
        if in_show:
            plt.imshow(dilated_image, cmap="gray")
            plt.title("dilate_image")
            plt.show()
        return dilated_image

    def threshold_image(
        self, image: np.ndarray, threshold: int = 0, in_show: bool = False
    ) -> np.ndarray:
        """
        Apply thresholding to the input image.

        :param image: The input image
        :param threshold: Threshold value (default is 0)
        :param in_show: True to display image
        :return: The thresholded image
        """
        _, thresholded = cv2.threshold(
            image, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if in_show:
            plt.imshow(thresholded, cmap="gray")
            plt.title("threshold_image")
            plt.show()
        return thresholded
