#!/usr/bin/env sh
set -e
core=24
tag=ompi
N=1
n=$N
t=26
[ "$t" -gt "$core" ] && t=$core
c=`expr $t + 0`
[ "$c" -gt "$core" ] && c=$core

LOG=examples/mnist/result-${tag}-$N-$t-$c.log
SNAPSHOT=examples/mnist/lenet_iter_0.solverstate
yhrun -n $n -c $c -N $N ./build/tools/caffe train \
		--solver=examples/mnist/lenet_solver.prototxt \
		--thread=$t \
		--snapshot=$SNAPSHOT \
		&>$LOG

echo "yhrun -n $n -c $c -N $N ./build/tools/caffe train \
		--solver=examples/mnist/lenet_solver.prototxt \
		--thread=$t \
		--snapshot=$SNAPSHOT \
		&>$LOG" >>$LOG