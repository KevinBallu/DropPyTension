#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Initialize the package by importing key classes and functions
from .utils import setup_environment, setup_constants
from .processor import ImageProcessor

__all__ = [
    'ImageProcessor',
    'setup_environment',
    'setup_constants'
]