# model_factory.py  (NEW)
from ClassesML.TabularMLP import TabularMLP

def build_model(hyperparameters, embedding_sizes):
    """
    Returns a brand-new TabularMLP instance.
    All kwargs map one-to-one to TabularMLP.__init__.
    """
    return TabularMLP(
        hyperparameters,
        embedding_sizes
    )
