#!/bin/bash

PROG_CPU="./vectores.exe"
PROG_1="ej1/paralelo.exe"
PROG_2="ej2/paralelo.exe"
PROG_3="ej3/paralelo.exe"
SEED=3
N=(262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728)
BLOCKSIZE=(4 16 64 256 1024)

for i in ${N[@]}
do
	for j in ${BLOCKSIZE[@]}
	do
		for k in {1..10}
		do
			#echo "It:" $k "BlockSize:" $j "N:" $i >> test\/paralelo1.txt
			echo "It:" $k "BlockSize:" $j "N:" $i >> test\/paralelo2.txt
			#echo "It:" $k "BlockSize:" $j "N:" $i >> test\/paralelo3.txt

			#$PROG_1 $j $i >> test\/paralelo1.txt
			$PROG_2 $j $i >> test\/paralelo2.txt
			#$PROG_3 $j $SEED $i >> test\/paralelo3.txt

			#echo "" >> test\/paralelo1.txt
			echo "" >> test\/paralelo2.txt
			#echo "" >> test\/paralelo3.txt
		done
	done
done
#
#for i in ${N[@]}
#do
#	for j in {1..10}
#	do
#		echo "It:" $j "N:" $i >> test\/secuencial.txt
#		$PROG_CPU $i >> test\/secuencial.txt
#		echo "" >> test\/secuencial.txt
#	done
#done
