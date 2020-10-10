#!/bin/bash

p=`pwd`
cd bad
time ./go.sh
cd $p
cd bad2
time ./go.sh
cd $p
