"""Independent exact regressions for the reused grouped Laurent validator.

The intentionally scalar reference below is only a correctness oracle on small
fixtures. Production must keep qhsolve's grouped Sage multiplication strategy.
"""
from fractions import Fraction
import random
import unittest
from unittest.mock import patch

from sage.all import QQ, ZZ, LaurentPolynomialRing

from hologuess import _qhsolve as core
from hologuess.qsequence import QSequence, _residual, guess_qsequence


def rational(value):
    numerator = getattr(value, 'numerator', None)
    denominator = getattr(value, 'denominator', None)
    if callable(numerator) and callable(denominator):
        return Fraction(int(numerator()), int(denominator()))
    return Fraction(value)


def reference(operator, values, start=0):
    """Scalar characteristic-zero residual, independent of Sage multiplication."""
    operator = {key:rational(c) for key,c in operator.items() if c}
    if not operator:
        raise ValueError('A validation witness must be nonzero.')
    order = max(b for b,j,e in operator)
    windows = len(values)-order
    if windows <= 0:
        return False, 0
    for n in range(windows):
        terms = {}
        for (b,j,e), c in operator.items():
            for exponent, value in values[n+b].items():
                power = int(exponent)+e+j*(n+start)
                terms[power] = terms.get(power, Fraction(0)) + c*rational(value)
        if any(terms.values()):
            return False, n
    return True, windows


def first_order_values(start, count, *, rational_data=False):
    factor = Fraction(1, 3) if rational_data else 1
    values = []
    for n in range(start,start+count):
        shift = n*(n-1)//2-2*n
        values.append({shift-4:factor, shift+2:-2*factor})
    return values


def multiply_operator(operator, scalar):
    result = {}
    for (b,j,e), c in operator.items():
        for exponent, coefficient in scalar.items():
            key = (b,j,e+exponent)
            result[key] = result.get(key,0)+c*coefficient
    return {key:c for key,c in result.items() if c}


class GroupedValidation(unittest.TestCase):
    def test_integer_laurent_ring_selection(self):
        ring, cells = core.to_sage({(0,0,-3):ZZ(7),(1,2,4):-2})
        self.assertEqual(ring.base_ring(),ZZ)
        self.assertEqual(cells[(0,0)][-3],7)

    def test_rational_operator_ring_selection(self):
        ring, cells = core.to_sage({(0,1,-3):Fraction(2,7),(1,0,0):QQ(1)/2})
        self.assertEqual(ring.base_ring(),QQ)
        self.assertEqual(cells[(0,1)][-3],QQ(2)/7)
        self.assertEqual(cells[(1,0)][0],QQ(1)/2)

    def test_integral_sage_rationals_keep_integer_ring(self):
        ring, _ = core.to_sage({(0,0,0):QQ(2),(1,0,1):QQ(-3)})
        self.assertEqual(ring.base_ring(),ZZ)

    def test_nonzero_and_negative_index_origins(self):
        operator = multiply_operator({(1,0,0):1,(0,1,-2):-1},{-3:2,1:-1})
        for start in [-4,0,7]:
            with self.subTest(start=start):
                values = first_order_values(start,14)
                self.assertEqual(core.annihilates_exactly(operator,values,start=start),
                                 reference(operator,values,start))

    def test_rational_laurent_data_with_integer_operator(self):
        operator = {(1,0,0):1,(0,1,-2):-1}
        values = first_order_values(3,12,rational_data=True)
        self.assertEqual(core.annihilates_exactly(operator,values,start=3),(True,11))

    def test_rational_data_and_operator_coefficients(self):
        operator = multiply_operator({(1,0,0):1,(0,1,-2):-1},
                                     {-3:Fraction(1,2),2:Fraction(2,7)})
        values = first_order_values(-2,13,rational_data=True)
        self.assertEqual(core.annihilates_exactly(operator,values,start=-2),
                         reference(operator,values,-2))

    def test_multiple_q_cells_are_combined_before_each_shift_product(self):
        ring = LaurentPolynomialRing(QQ,'q'); q = ring.gen()
        start = -3
        sequence = [q**-3+QQ(1)/2, 1-q+q**4]
        for n in range(start,start+11):
            sequence.append((1+q**n)*sequence[-1]-q**-1*sequence[-2])
        values = [p.dict() for p in sequence]
        operator = multiply_operator({(2,0,0):1,(1,0,0):-1,(1,1,0):-1,(0,0,-1):1},
                                     {-2:Fraction(2,3),3:Fraction(-1,7)})
        self.assertEqual(core.annihilates_exactly(operator,values,start=start),(True,11))
        self.assertEqual(core.annihilates_exactly(operator,values,start=start),
                         reference(operator,values,start))

    def test_nonzero_operator_on_zero_sequence_is_valid(self):
        self.assertEqual(core.annihilates_exactly({(1,1,-2):1},[{}, {}, {}]),(True,2))

    def test_corrupt_term_reports_first_bad_window(self):
        operator = multiply_operator({(1,0,0):1,(0,1,-2):-1},{-1:1,2:2})
        values = first_order_values(5,12,rational_data=True)
        values[4][0] = values[4].get(0,0)+Fraction(1,11)
        self.assertEqual(core.annihilates_exactly(operator,values,start=5),(False,3))
        self.assertEqual(core.annihilates_exactly(operator,values,start=5),
                         reference(operator,values,5))

    def test_random_small_inputs_match_scalar_reference(self):
        rng = random.Random(98127)
        for trial in range(40):
            values = [{e:Fraction(rng.randint(-4,4),rng.randint(1,5))
                       for e in range(-3,4)} for _ in range(8)]
            operator = {(b,j,e):Fraction(rng.randint(-4,4),rng.randint(1,5))
                        for b in range(3) for j in range(3) for e in [-2,0,3]}
            start = rng.randint(-3,7)
            self.assertEqual(core.annihilates_exactly(operator,values,start=start),
                             reference(operator,values,start),trial)

    def test_empty_and_zero_operator_cannot_validate(self):
        for operator in [{},{(0,0,0):0}]:
            with self.subTest(operator=operator):
                with self.assertRaises(ValueError):
                    core.annihilates_exactly(operator,[{0:1},{1:1}])

    def test_no_complete_window_is_not_success(self):
        for values in [[],[{0:1}],[{0:1},{1:1}]]:
            self.assertEqual(core.annihilates_exactly({(2,0,0):1},values),(False,0))

    def test_explicit_integer_ring_never_truncates_rationals(self):
        with self.assertRaises((TypeError,ValueError)):
            core.to_sage({(0,0,0):Fraction(1,2)},base_ring=ZZ)
        with self.assertRaises((TypeError,ValueError)):
            core.annihilates_exactly({(0,0,0):1},[{0:Fraction(1,2)}],base_ring=ZZ)

    def test_private_compatibility_wrapper_delegates_to_core(self):
        operator = {(1,0,0):1,(0,1,-2):-1}
        values = first_order_values(4,10,rational_data=True)
        with patch.object(core,'annihilates_exactly',wraps=core.annihilates_exactly) as checked:
            self.assertEqual(_residual(operator,values,4),(True,9))
        checked.assert_called_once()
        self.assertEqual(checked.call_args.kwargs.get('start'),4)

    def test_public_guess_uses_grouped_validator(self):
        values = [{n:QQ(1)/3} for n in range(20)]
        with patch.object(core,'annihilates_exactly',wraps=core.annihilates_exactly) as checked:
            result = guess_qsequence(values,max_order=1,max_degree=0,max_points=256)
        self.assertTrue(result.metadata['finite_window_valid'])
        self.assertEqual(checked.call_count,1)

    def test_rational_sequences_retain_degree_planning_data(self):
        sequence = QSequence([{n-3:Fraction(1,n+2)} for n in range(12)])
        self.assertIsNotNone(sequence.polys)
        self.assertEqual([sorted(p) for p in sequence.polys],[[n-3] for n in range(12)])

    def test_rational_sequences_use_vectorized_modular_evaluation(self):
        sequence = QSequence([{n-3:Fraction(1,n+2)} for n in range(12)])
        with patch.object(core,'eval_laurent',wraps=core.eval_laurent) as evaluated:
            points, actual = sequence.at(101,4)
            more_points, more = sequence.at(101,7)
        self.assertEqual(evaluated.call_count,2)
        for n in range(12):
            for k,p in enumerate(points):
                expected = pow(int(p),n-3,101)*pow(n+2,-1,101)%101
                self.assertEqual(int(actual[n,k]),expected)
        self.assertEqual(actual.tolist(),more[:,:4].tolist())
        self.assertEqual(points.tolist(),more_points[:4].tolist())

    def test_integral_values_stay_integral_in_the_wrapper(self):
        sequence = QSequence([{n:QQ(3)} for n in range(12)])
        self.assertTrue(all(isinstance(c,int) for value in sequence.values for c in value.values()))


if __name__ == '__main__': unittest.main()
