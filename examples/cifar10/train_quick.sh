#!/usr/bin/env sh
set -e
core=24
tag=ompi
N=1
n=$N
t=16
c=$t

LOG=examples/cifar10/quick-${tag}-$N-$n-$t-$c.log
SNAPSHOT=examples/cifar10/cifar10_quick_iter_0.solverstate.h5

TOOLS=./build/tools


echo "yhrun -n $n -c $c -N $N $TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_solver.prototxt \
  --thread=$t \
  --snapshot=$SNAPSHOT \
  &>>$LOG" >$LOG

echo "yhrun -n $n -c $c -N $N $TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_solver_lr1.prototxt \
  --snapshot=examples/cifar10/cifar10_quick_iter_4000.solverstate.h5
  --thread=$t \
  &>>$LOG" >> $LOG

yhrun -n $n -c $c -N $N $TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_solver.prototxt \
  --thread=$t \
  --snapshot=$SNAPSHOT \
  &>>$LOG

# reduce learning rate by factor of 10 after 8 epochs
yhrun -n $n -c $c -N $N $TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_solver_lr1.prototxt \
  --snapshot=examples/cifar10/cifar10_quick_iter_4000.solverstate.h5 \
  --thread=$t \
  &>>$LOG
