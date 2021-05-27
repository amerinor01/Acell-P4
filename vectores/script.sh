#!/bin/bash

PROG_CPU="build/vectores.exe"
PROG_1="build/paralelo_1.out"
PROG_2="build/paralelo_2.out"
PROG_3="build/paralelo_3.out"
SEED=3
N=(262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728)
BLOCKSIZE=(16 64 256 1024)
STREAM=(64 128 256 1024)
for i in ${N[@]}
do
	for l in ${STREAM[@]}
	do
		for j in ${BLOCKSIZE[@]}
		do
			for k in {1..10}
			do
			#	echo "It:" $k "BlockSize:" $j "N:" $i "STREAM:" $l >> test\/paralelo1.txt
				#echo "It:" $k "BlockSize:" $j "N:" $i "STREAM:" $l >> test\/paralelo2.txt
				echo "It:" $k "BlockSize:" $j "N:" $i "STREAM:" $l >> test\/paralelo3.txt

				#$PROG_1 $j  $l $i 0 >> test\/paralelo1.txt
				#$PROG_2 $j  $l $i 0 >> test\/paralelo2.txt
				$PROG_3 $j  $l $i $SEED 0 >> test\/paralelo3.txt

				#echo "" >> test\/paralelo1.txt
			 	#echo "" >> test\/paralelo2.txt
				echo "" >> test\/paralelo3.txt
			done
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
