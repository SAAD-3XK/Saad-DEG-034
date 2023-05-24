from my_proj.sum_func import add_positive, add
import pytest


# first testing add_positive function for correct outputs
def test_add_all_positive():    
    assert add_positive(5, 9) == 14

def test_add_all_negative():
    assert add_positive(-4, -3) == None
    
def test_add_positive_and_negative():
    assert add_positive(9, -3) == None
    
def test_add_negative_and_positive():
    assert add_positive(-4, 3) == None

def test_add_both_zero():
    assert add_positive(0, 0) == None

def test_add_zero():
    assert add_positive(0, 7) == None

# now testing add_positive function whether it throws errors on string inputs
def test_add_with_one_string_one_int():
    with pytest.raises(TypeError):
        add_positive("2", 3)

def test_add_with_one_int_one_string():
    with pytest.raises(TypeError):
        add_positive(2, "3")

def test_add_with_all_strings():
    with pytest.raises(TypeError):
        add_positive("2", "3")

# now testing add function whether it throws errors on string inputs
def test_first_add_with_one_string_one_int():
    with pytest.raises(TypeError):
        add("2", 3)

def test_first_add_with_one_int_one_string():
    with pytest.raises(TypeError):
        add(2, "3")

'''
    this test should fail as the add function will NOT throw an error
    it will instead give a result "23"
'''
def test_first_add_with_all_strings():
    with pytest.raises(TypeError):
        add("2", "3")