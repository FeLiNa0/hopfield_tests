Hopfield Network Performance Tests
==================================

Hopfield Networks are recursive neural networks that model human memory. Although commonly referenced for their clear theory, see the bibliography for that. Instead, the programs here can be used to test the recall effectiveness of Hopfield Networks. A sample dataset composed of the Walsh vectors, an orthogonal system is included by default.

How to install
--------------
Python, numpy, matplotlib/pyplot are needed.
See [here](http://matplotlib.org/users/installing.html) for a description of the process.

How to run
----------

```
python3 tests.py --nvectors --size --samples --latex-ouput
```

Sample Results
--------------
```
python3 tests.py --nvectors 3, 5, 10 --samples 100 --size 5

vector size 5, vectors in memory 3 ---  randomly flipped 0% of bits
	matched, nearly matched, mean iterations to convergence
	(100.00 ±   0.00)%, (100.00 ±   0.00)% -- 1.00

  5,   3 ---  0.1%
	(100.00 ±   0.00)%, (100.00 ±   0.00)% -- 2.00

  5,   3 ---  0.2%
	( 70.00 ±  20.00)%, ( 95.62 ±  14.79)% -- 2.00

  5,   5 ---  0.0%
	(100.00 ±   0.00)%, (100.00 ±   0.00)% -- 1.00

  5,   5 ---  0.1%
	(100.00 ±   0.00)%, (100.00 ±   0.00)% -- 2.00

  5,   5 ---  0.2%
	( 50.00 ±  20.98)%, ( 92.50 ±  20.16)% -- 2.67

  5,  10 ---  0.0%
	(100.00 ±   0.00)%, (100.00 ±   0.00)% -- 1.00

  5,  10 ---  0.1%
	( 47.27 ±  16.18)%, ( 96.70 ±  11.00)% -- 1.47

  5,  10 ---  0.2%
	(  1.82 ±   6.03)%, ( 79.72 ±  20.75)% -- 1.84
```

Bibliography
------------
[The Hopfield Model](http://page.mi.fu-berlin.de/rojas/neural/chapter/K13.pdf)
[Hopfield Networks](http://www.comp.leeds.ac.uk/ai23/reading/Hopfield.pdf)

