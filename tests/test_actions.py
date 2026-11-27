import unittest
from sage.all import QQ, PolynomialRing, matrix, vector
from hologuess.actions import OreAction, act, module_problem


class ModuleActions(unittest.TestCase):
    def test_twisted_matrices_are_not_ordinary_powers(self):
        R = PolynomialRing(QQ, 'z'); z = R.gen()
        B = matrix(R, [[z]])
        action = OreAction(B, sigma=lambda a: a(z+1))
        v = vector(R, [1])
        self.assertEqual(act([action], [2], v)[0], z*(z+1))
        self.assertNotEqual(act([action], [2], v), B**2*v)

    def test_derivation_is_applied_at_every_iteration(self):
        R = PolynomialRing(QQ, 'z'); z = R.gen()
        action = OreAction(matrix(R, [[z]]), delta=lambda a: a.derivative())
        self.assertEqual(act([action], [2], vector(R,[1]))[0], z*z+1)

    def test_generic_module_relation(self):
        from hologuess.module import solve_module
        B = matrix(QQ, [[0,1],[1,1]])
        action = OreAction(B)
        problem = module_problem([action], vector(QQ,[1,0]),
                                 [(1,(i,)) for i in range(3)])
        result = solve_module(problem)
        self.assertTrue(result.basis)
        self.assertTrue(all(sum(c*(B**i*vector(QQ,[1,0]))
                                for i,c in enumerate(v)) == 0 for v in result.basis))


if __name__ == '__main__': unittest.main()
