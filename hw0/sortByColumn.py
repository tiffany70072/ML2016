import numpy as np
import sys

column = int(sys.argv[1])
data = []
column_value = []

for line in open(sys.argv[2], 'r').readlines():
	data.append(line.split())

for i in range(len(data)):
	column_value.append(data[i][column])
	
column_value.sort(key = float)

for i in range(len(column_value)-1):
	#sys.stdout.write(string(column_value[i]))
	#print '\b%s'%column_value[i],
	
	print column_value[i],
	sys.stdout.softspace = False
	print ',',
	sys.stdout.softspace = False

print column_value[len(column_value)-1],
sys.stdout.softspace = False