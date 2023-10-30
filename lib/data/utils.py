import numpy as np
import os
import sys
import io
import pickle
import time

from typing import Any, List, Tuple, Union
import cv2
from PIL import Image

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


class Logger(object):
    """Redirect stderr to stdout, optionally print stdout to a file, and optionally force flushing on both stdout and the file."""

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: Union[str, bytes]) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if isinstance(text, bytes):
            text = text.decode()
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()
            self.file = None


def print_stats(input, name="input", scientific=False):
    dims = input.shape[-1]
    template = "{} {} dim={}: min={:.3e}, mean={:.3e}, max={:.3e}, std={:.3e}" if scientific else "{} {} dim={}: min={:.3f}, mean={:.3f}, max={:.3f}, std={:.3f}"
    for i in range(dims):
        x = input[..., i]
        print(template.format(
            name, input.shape, i, x.min(), x.mean(), x.max(), x.std()
        ))


def stitch_images(images, squeeze_width):

    H, W, C = images[0].shape

    new_width = W - 2 * squeeze_width
    canvas_width = (len(images) - 1) * new_width + W

    stitched = np.ones([H, canvas_width, C], dtype=np.uint8) * 255

    for i in range(len(images)):

        canvas = np.ones([H, canvas_width, C], dtype=np.uint8) * 255
        canvas[:, i*new_width:i*new_width+W] = images[i]
        stitched = np.minimum(stitched, canvas)

    return stitched
