import inspect


def add_to_class(Class):
    def wrapper(obj):
        """
        Wrapper function returns nothing meaning that the decorated function is no longer available for use
        outside of the 'Class' it has been wrapped with
        """
        setattr(Class, obj.__name__, obj)
        
    return wrapper 


def save_hyperparameters(self, ignore=[]):
    """Save function arguments into class attributes."""
    frame = inspect.currentframe().f_back
    _, _, _, local_vars = inspect.getargvalues(frame)
    self.hparams = {k:v for k, v in local_vars.items()
                    if k not in set(ignore+['self']) and not k.startswith('_')}
    for k, v in self.hparams.items():

        setattr(self, k, v)

        setattr(self, k, v)
