import pytest
from src.some_module import some_function  # Örnek fonksiyon

def test_some_function():
    result = some_function(2, 3)
    assert result == 5  # Beklenen değer
