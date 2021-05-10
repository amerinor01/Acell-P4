#!/bin/bash

PROG_CPU="./vectores.exe"
PROG_1="ej1/paralelo.exe"
PROG_2="ej2/paralelo.exe"
PROG_3="ej3/paralelo.exe"
SEED=3
N=(256 2048 16384 131072 1048576 8388608 134217728) #,8388608,134217728,1073741824,8589934592,6871476738)
BLOCKSIZE=(4 16 64 256 1024)

for i in ${N[@]}
do
	for j in ${BLOCKSIZE[@]}
	do
		for k in {1..10}
		do
			echo "It: " $k " BlockSize: " $j " N: " $i >> test\/paralelo1.txt
			echo "It: " $k " BlockSize: " $j " N: " $i >> test\/paralelo2.txt
			echo "It: " $k " BlockSize: " $j " N: " $i >> test\/paralelo3.txt

			$PROG_1 $j $i >> test\/paralelo1.txt
			$PROG_2 $j $i >> test\/paralelo2.txt
			$PROG_3 $j $SEED $i >> test\/paralelo3.txt

			echo "" >> test\/paralelo1.txt
			echo "" >> test\/paralelo2.txt
			echo "" >> test\/paralelo3.txt
		done
	done
done

for i in ${N[@]}
do
	for j in {1..10}
	do
		echo "It: " $j " N: " $i >> test\/secuencial.txt
		$PROG_CPU $i >> test\/secuencial.txt
		echo "" >> test\/secuencial.txt
	done
done
