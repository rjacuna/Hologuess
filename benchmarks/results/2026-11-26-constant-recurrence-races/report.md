# Cold constant-coefficient recurrence races

On the two SymPy test inputs, Hologuess with its default FLINT bridge won all
six pairs against native Ore. SymPy and Wolfram's built-in recurrence guesser
each won all six of their pairs. These cases have constant coefficients and
orders two and three. [Samples and medians](summary.json)

| Input | Competitor | Hologuess median (ms) | Competitor median (ms) | Exact pairs |
|---|---|---:|---:|---:|
| Fibonacci | Native Ore | 54.233 | 110.163 | 3/3 |
| Fibonacci | SymPy | 54.843 | 0.385 | 3/3 |
| Fibonacci | Wolfram | 52.554 | 15.405 | 3/3 |
| Order three | Native Ore | 54.633 | 111.407 | 3/3 |
| Order three | SymPy | 56.209 | 0.901 | 3/3 |
| Order three | Wolfram | 53.241 | 15.422 | 3/3 |

The Fibonacci fit uses the twelve values from the SymPy suite, with four
further values held out. The order-three fit uses the eighteen values from
the suite, with five further values held out. The independent rational kernels
have dimension one and give $E^2-E-1$ and $E^3-5E^2+3E-8$, with $E$ the
forward shift. Both operators satisfy every fit and withheld window. All 36
candidate spaces matched those full kernels exactly. The benchmark workers
received the fit values only.
[Fibonacci test](https://github.com/sympy/sympy/blob/16fa855354eb7bcabd3fe10993841e03b1382692/sympy/concrete/tests/test_guess.py#L21)
· [Order-three test](https://github.com/sympy/sympy/blob/16fa855354eb7bcabd3fe10993841e03b1382692/sympy/concrete/tests/test_guess.py#L35)

One $N=500$ ms was fixed for the degree-zero class. Each pair used fresh
processes with a shared start barrier after library and fit-data setup. The
solver clock includes the public call and monic output conversion. Common
controller verification and persistence follow that clock. The $\infty$ tag
denotes a worker stopped once the grace interval elapsed and a competitor had
a verified result. All 18 pairs completed here.

The engines were Hologuess 0.3.0 with FLINT 3.4.0, native
`ore_algebra.guess` 0.5 under SageMath 10.10, SymPy
`find_simple_recurrence_vector` 1.14.0, and Wolfram
`FindLinearRecurrence` 14.3.0. Hologuess and Ore received the fixed order and
degree box. Wolfram received the same order as its upper bound. SymPy's API
infers the order from the fit values. The installed RISC `Guess` package
reported an unexpected loading error. Maple and FriCAS executables were
unavailable on this host, so this comparison has no timings for those three
engines. [Wolfram API](https://reference.wolfram.com/language/ref/FindLinearRecurrence.html)
· [RISC Guess](https://risc.jku.at/sw/guess/)

The [quickstart](../../../examples/quickstart.ipynb) reconstructs the Fibonacci recurrence. The [worked examples](../../../examples/worked_examples.ipynb) give the order-three recurrence.
