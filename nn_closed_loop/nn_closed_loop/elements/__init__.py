from .Element import Element
from .GuidedElement import GuidedElement
from .OptGuidedElement import OptGuidedElement

element_dict = {
    "Guided": GuidedElement,
    "OptGuided": OptGuidedElement,
    "Uniform": Element,
}
