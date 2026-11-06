from importlib import import_module


NodeDropPooling = import_module('.ndrp-new', __name__).NodeDropPooling

__all__ = ['NodeDropPooling']
