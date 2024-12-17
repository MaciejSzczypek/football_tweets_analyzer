from typing import Callable


def section_printing_decorator(section_name: str ):
    def _inner(section: Callable):
        def _wrapper(*args, **kwargs):
            print("=" * 50, f"Section: {section_name}", "=" * 50)
            print()
            result = section(*args, **kwargs)
            print("=" * 100)
            return result
        return _wrapper
    return _inner


def new_line_appendix_decorator(printing_function: Callable):
    def _wrapper(*args, **kwargs):
        printing_function(*args, **kwargs)
        print()
    return _wrapper
