#!/usr/bin/env sh
set -e
core=24
ver=ompi
tag=full
N=1
n=$N
t=10
c=$t

LOG=examples/cifar10/${tag}-${ver}-$N-$n-$t-$c.log
SNAPSHOT=examples/cifar10/cifar10_full_iter_0.solverstate.h5

TOOLS=./build/tools


echo "yhrun -n $n -c $c -N $N $TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_full_solver.prototxt \
  --thread=$t \
  --snapshot=$SNAPSHOT \
  &>>$LOG" >$LOG

echo "yhrun -n $n -c $c -N $N $TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_full_solver_lr1.prototxt \
  --snapshot=examples/cifar10/cifar10_full_iter_60000.solverstate.h5
  --thread=$t \
  &>>$LOG" >> $LOG

echo "yhrun -n $n -c $c -N $N $TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_full_solver_lr2.prototxt \
  --snapshot=examples/cifar10/cifar10_full_iter_65000.solverstate.h5
  --thread=$t \
  &>>$LOG" >> $LOG


yhrun -n $n -c $c -N $N $TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_full_solver.prototxt \
  --thread=$t \
  --snapshot=$SNAPSHOT \
  &>>$LOG

# reduce learning rate by factor of 10 after 8 epochs
yhrun -n $n -c $c -N $N $TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_full_solver_lr1.prototxt \
  --snapshot=examples/cifar10/cifar10_full_iter_60000.solverstate.h5 \
  --thread=$t \
  &>>$LOG

yhrun -n $n -c $c -N $N $TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_full_solver_lr2.prototxt \
  --snapshot=examples/cifar10/cifar10_full_iter_65000.solverstate.h5 \
  --thread=$t \
  &>>$LOG
