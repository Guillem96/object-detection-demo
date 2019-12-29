
def assert_shapes(shape, other):
    assert len(shape) == len(other), "Dimensions are different"
    for s, o in zip(shape, other):
        if s is not None and o is not None:
            assert s == o, "Shapes {} and {} are not equal".format(shape, other)
