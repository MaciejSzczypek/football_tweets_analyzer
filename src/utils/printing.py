from typing import Callable


def section_printing_decorator(printing_function: Callable):
    def _wrapper(*args, **kwargs):
        print("=" * 100)
        result = printing_function(*args, **kwargs)
        print("=" * 100)
        return result

    return _wrapper


def new_line_appendix_decorator(printing_function: Callable):
    def _wrapper(*args, **kwargs):
        printing_function(*args, **kwargs)
        print()

    return _wrapper
