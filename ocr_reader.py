from typing import Self

from easyocr import Reader
from easyocr.utils import get_paragraph
from unicodedata import combining, normalize
from PIL import Image
import cv2


class OCRReader():

    def __init__(self, img: Image, language: list[str] | str, gpu: bool, new_img_path: str) -> None:
        self.image: cv2.Mat = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.reader = Reader(language, gpu=gpu, verbose=False)
        self.file_path: str = new_img_path

    def easy_ocr(self: Self) -> str | None:

        if self.reader:
            res_all = self.reader.readtext(image=self.image)

            for (box, text, _) in res_all:

                (tl, tr, br, bl) = box
                tl = (int(tl[0]), int(tl[1]))
                tr = (int(tr[0]), int(tr[1]))
                br = (int(br[0]), int(br[1]))
                bl = (int(bl[0]), int(bl[1]))

                overlay = self.image.copy()
                cv2.rectangle(overlay, tl, br, (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, self.image,
                                0.4, 0, self.image)

                # Rectangle border
                cv2.rectangle(self.image, tl, br, (255, 200, 0), 1)

                text = u"".join([c for c in normalize('NFKD', text)
                                 if not combining(c)])

                cv2.putText(self.image, text, (tl[0] + 4, tl[1] + 17),
                            cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 200, 0), 1)

                cv2.imwrite(filename=self.file_path, img=self.image)

            text = "\n".join([item[1] for item in get_paragraph(res_all)])

            if text:
                return text
