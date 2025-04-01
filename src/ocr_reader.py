import string

import cv2
import easyocr

# Mapping dictionaries for character conversion
dict_char_to_int = {"O": "0", "I": "1", "J": "3", "A": "4", "G": "6", "S": "5"}

dict_int_to_char = {"0": "O", "1": "I", "3": "J", "4": "A", "6": "G", "5": "S"}


class OCRReader:
    def __init__(self, gpu):
        # Initialize the OCR reader
        self.reader = easyocr.Reader(["en"], gpu=gpu)

    def read_license_plate(self, license_plate_crop):
        """
        Read the license plate text from the given cropped image.

        Args:
            license_plate_crop (np.array): Cropped image containing the license plate.

        Returns:
            tuple: Tuple containing the formatted license plate text and its confidence score.
        """
        # Pre-process License Plate
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        _, license_plate_crop_thresh = cv2.threshold(
            license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
        )

        # Read License Plate
        detections = self.reader.readtext(license_plate_crop_thresh)

        # Extract License Plate Text
        for detection in detections:
            # Extract Text and Score
            _, text, score = detection

            text = text.upper().replace(" ", "")

            if self.license_complies_format(text):
                # Format License Plate
                license_plate_txt = self.format_license(text), score
                return license_plate_txt, score

        # Return None if no license plate is detected
        return None, None

    def license_complies_format(self, text):
        """
        Check if the license plate text complies with the required format.

        Args:
            text (str): License plate text.

        Returns:
            bool: True if the license plate complies with the format, False otherwise.
        """
        # Check if the license plate text complies with the format
        if len(text) != 7:
            return False

        # Check if the license plate text complies with the format
        if (
            (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys())
            and (
                text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()
            )
            and (
                text[2] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[2] in dict_char_to_int.keys()
            )
            and (
                text[3] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[3] in dict_char_to_int.keys()
            )
            and (
                text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()
            )
            and (
                text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()
            )
            and (
                text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()
            )
        ):
            return True
        else:
            return False

    def format_license(self, text):
        """
        Format the license plate text by converting characters using the mapping dictionaries.

        Args:
            text (str): License plate text.

        Returns:
            str: Formatted license plate text.
        """
        license_plate_txt = ""
        mapping = {
            0: dict_int_to_char,
            1: dict_int_to_char,
            4: dict_int_to_char,
            5: dict_int_to_char,
            6: dict_int_to_char,
            2: dict_char_to_int,
            3: dict_char_to_int,
        }
        for j in [0, 1, 2, 3, 4, 5, 6]:
            if text[j] in mapping[j].keys():
                license_plate_txt += mapping[j][text[j]]
            else:
                license_plate_txt += text[j]

        return license_plate_txt
